"""Batched Feature Matrix Extractor with Disk Checkpointing (.npz)."""
from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Any, List, Optional, Tuple

from .data import PreferenceExample
from .logger import get_logger

logger = get_logger("PDD.FeatureExtractor")


@dataclass
class FeatureMatrices:
    """Example-level feature matrices for retained preference examples."""

    example_ids: np.ndarray             # (N,)
    P_max: np.ndarray                   # (N, d_sae)
    P_freq: np.ndarray                  # (N, d_sae)
    C_max: np.ndarray                   # (N, d_sae)
    C_freq: np.ndarray                  # (N, d_sae)
    R_max: np.ndarray                   # (N, d_sae)
    R_freq: np.ndarray                  # (N, d_sae)

    @property
    def D_max(self) -> np.ndarray:
        return self.C_max - self.R_max

    @property
    def D_freq(self) -> np.ndarray:
        return self.C_freq - self.R_freq

    def save_npz(self, filepath: str) -> None:
        """Save feature matrices to disk as an uncompressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_filepath = filepath.replace(".npz", "_tmp.npz")
        np.savez(
            tmp_filepath,
            example_ids=self.example_ids,
            P_max=self.P_max,
            P_freq=self.P_freq,
            C_max=self.C_max,
            C_freq=self.C_freq,
            R_max=self.R_max,
            R_freq=self.R_freq,
        )
        os.replace(tmp_filepath, filepath)

    @classmethod
    def load_npz(cls, filepath: str) -> FeatureMatrices:
        """Load feature matrices from disk .npz archive."""
        data = np.load(filepath)
        return cls(
            example_ids=data["example_ids"],
            P_max=data["P_max"],
            P_freq=data["P_freq"],
            C_max=data["C_max"],
            C_freq=data["C_freq"],
            R_max=data["R_max"],
            R_freq=data["R_freq"],
        )


class FeatureMatrixExtractor:
    """Optimized batched SAE feature extractor with VRAM management & disk caching."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        sae: Any,
        hook_layer: int,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.hook_layer = hook_layer
        self.device = device
        self.batch_size = batch_size

    def extract(
        self,
        examples: List[PreferenceExample],
        checkpoint_path: Optional[str] = None,
        use_checkpoint: bool = True,
    ) -> FeatureMatrices:
        """Extract or load feature matrices."""
        if use_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading cached feature matrices from checkpoint: {checkpoint_path}")
            return FeatureMatrices.load_npz(checkpoint_path)


        logger.info(f"Extracting SAE feature matrices for {len(examples)} examples (batch_size={self.batch_size})...")
        matrices = self._extract_batched(
            examples=examples,
            checkpoint_path=checkpoint_path,
            use_checkpoint=use_checkpoint,
        )

        if checkpoint_path:
            logger.info(f"Saving feature matrices checkpoint to {checkpoint_path}...")
            matrices.save_npz(checkpoint_path)

        return matrices

    @torch.inference_mode()
    def _extract_batched(
        self,
        examples: List[PreferenceExample],
        checkpoint_path: Optional[str] = None,
        use_checkpoint: bool = True,
        save_every_batches: int = 1250,
    ) -> FeatureMatrices:
        self.model.eval()

        d_sae = self.sae.cfg.d_sae
        N = len(examples)
        example_ids = np.array([ex.example_id for ex in examples], dtype=np.int64)

        P_max = np.zeros((N, d_sae), dtype=np.float32)
        P_freq = np.zeros((N, d_sae), dtype=np.float32)
        C_max = np.zeros((N, d_sae), dtype=np.float32)
        C_freq = np.zeros((N, d_sae), dtype=np.float32)
        R_max = np.zeros((N, d_sae), dtype=np.float32)
        R_freq = np.zeros((N, d_sae), dtype=np.float32)

        partial_ckpt = checkpoint_path.replace(".npz", "_partial.npz") if checkpoint_path else None
        start_batch_idx = 0

        if use_checkpoint and partial_ckpt and os.path.exists(partial_ckpt):
            try:
                logger.info(f"Found partial batch checkpoint '{partial_ckpt}'. Loading intermediate state...")
                partial_data = np.load(partial_ckpt)
                p_m_part = partial_data["P_max"]
                n_part = len(p_m_part)
                P_max[:n_part] = p_m_part
                P_freq[:n_part] = partial_data["P_freq"]
                C_max[:n_part] = partial_data["C_max"]
                C_freq[:n_part] = partial_data["C_freq"]
                R_max[:n_part] = partial_data["R_max"]
                R_freq[:n_part] = partial_data["R_freq"]
                start_batch_idx = int(partial_data["last_batch_idx"]) + 1
                logger.info(f"Resuming SAE feature extraction from batch {start_batch_idx} ({n_part}/{N} examples pre-loaded)...")
            except Exception as e:
                logger.warning(f"Could not load partial checkpoint '{partial_ckpt}': {e}. Starting from batch 0...")
                start_batch_idx = 0

        # Hook target layer
        residual_container = []

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            residual_container.append(hidden.detach())

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            target_layer = self.model.model.layers[self.hook_layer]
        elif hasattr(self.model, "layers"):
            target_layer = self.model.layers[self.hook_layer]
        else:
            raise ValueError(f"Unable to locate decoder layer {self.hook_layer} in model architecture.")

        handle = target_layer.register_forward_hook(hook_fn)

        try:
            num_batches = (N + self.batch_size - 1) // self.batch_size
            for b_idx in tqdm(range(start_batch_idx, num_batches), desc="Batched SAE extraction", initial=start_batch_idx, total=num_batches):
                start_i = b_idx * self.batch_size
                end_i = min(start_i + self.batch_size, N)
                batch_exs = examples[start_i:end_i]

                # 1. Process Prompts (applying chat template if tokenizer supports it)
                prompts = [ex.prompt for ex in batch_exs]
                p_m, p_f = self._process_span_batch(prompts, target_layer, residual_container, is_prompt=True)
                P_max[start_i:end_i] = p_m
                P_freq[start_i:end_i] = p_f

                # 2. Process Chosen Responses
                chosens = [ex.chosen for ex in batch_exs]
                c_m, c_f = self._process_span_batch(chosens, target_layer, residual_container, is_prompt=False)
                C_max[start_i:end_i] = c_m
                C_freq[start_i:end_i] = c_f

                # 3. Process Rejected Responses
                rejecteds = [ex.rejected for ex in batch_exs]
                r_m, r_f = self._process_span_batch(rejecteds, target_layer, residual_container, is_prompt=False)
                R_max[start_i:end_i] = r_m
                R_freq[start_i:end_i] = r_f

                # Incrementally save partial progress checkpoint (sliced up to end_i for fast uncompressed saving)
                if partial_ckpt and ((b_idx + 1) % save_every_batches == 0 or b_idx == num_batches - 1):
                    tmp_ckpt = partial_ckpt.replace(".npz", "_tmp.npz")
                    np.savez(
                        tmp_ckpt,
                        P_max=P_max[:end_i],
                        P_freq=P_freq[:end_i],
                        C_max=C_max[:end_i],
                        C_freq=C_freq[:end_i],
                        R_max=R_max[:end_i],
                        R_freq=R_freq[:end_i],
                        last_batch_idx=b_idx,
                    )
                    os.replace(tmp_ckpt, partial_ckpt)

            if partial_ckpt and os.path.exists(partial_ckpt):
                try:
                    os.remove(partial_ckpt)
                except Exception:
                    pass



        finally:
            handle.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return FeatureMatrices(
            example_ids=example_ids,
            P_max=P_max,
            P_freq=P_freq,
            C_max=C_max,
            C_freq=C_freq,
            R_max=R_max,
            R_freq=R_freq,
        )

    def _process_span_batch(
        self,
        texts: List[str],
        target_layer: Any,
        residual_container: List[torch.Tensor],
        is_prompt: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        residual_container.clear()

        # Apply chat template formatting to user prompts if chat_template exists
        if is_prompt and getattr(self.tokenizer, "chat_template", None) is not None:
            formatted_texts = []
            for text in texts:
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    formatted_texts.append(formatted)
                except Exception:
                    formatted_texts.append(text)
            texts = formatted_texts

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(self.device)


        with torch.inference_mode():
            self.model(**inputs)

        resid = residual_container[0]  # (B, L, d_in)
        mask = inputs.attention_mask.unsqueeze(-1).to(self.sae.device) # (B, L, 1)

        # Encode with SAE
        acts = self.sae.encode(resid.to(self.sae.device)).to(torch.float32) # (B, L, d_sae)

        # Mask out padding tokens before computing max / mean frequency
        acts_masked = acts * mask

        # Span Max: (B, d_sae)
        span_max = torch.max(acts_masked, dim=1).values.detach().cpu().numpy()

        # Span Frequency: sum over valid tokens / valid token count
        token_counts = torch.sum(inputs.attention_mask, dim=1, keepdim=True).to(self.sae.device) # (B, 1)
        token_counts = torch.clamp(token_counts, min=1)
        span_freq = (torch.sum((acts_masked > 0).float(), dim=1) / token_counts).detach().cpu().numpy()

        return span_max, span_freq

