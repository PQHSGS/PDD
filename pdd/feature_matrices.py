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


import scipy.sparse as sp


def _to_csr(mat: Any) -> sp.csr_matrix:
    if sp.isspmatrix_csr(mat):
        return mat
    elif sp.issparse(mat):
        return mat.tocsr()
    else:
        return sp.csr_matrix(mat, dtype=np.float32)


@dataclass
class FeatureMatrices:
    """Example-level sparse feature matrices for retained preference examples."""

    example_ids: np.ndarray             # (N,)
    P_max: Any                          # (N, d_sae) sp.csr_matrix
    P_freq: Any                         # (N, d_sae) sp.csr_matrix
    C_max: Any                          # (N, d_sae) sp.csr_matrix
    C_freq: Any                         # (N, d_sae) sp.csr_matrix
    R_max: Any                          # (N, d_sae) sp.csr_matrix
    R_freq: Any                         # (N, d_sae) sp.csr_matrix

    def __post_init__(self):
        self.P_max = _to_csr(self.P_max)
        self.P_freq = _to_csr(self.P_freq)
        self.C_max = _to_csr(self.C_max)
        self.C_freq = _to_csr(self.C_freq)
        self.R_max = _to_csr(self.R_max)
        self.R_freq = _to_csr(self.R_freq)

    @property
    def D_max(self) -> sp.csr_matrix:
        return self.C_max - self.R_max

    @property
    def D_freq(self) -> sp.csr_matrix:
        return self.C_freq - self.R_freq

    def save_npz(self, filepath: str, last_batch_idx: Optional[int] = None) -> None:
        """Save sparse feature matrices to disk as a compact .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_filepath = filepath.replace(".npz", "_tmp.npz")

        kwargs = {"example_ids": self.example_ids}
        if last_batch_idx is not None:
            kwargs["last_batch_idx"] = np.array([last_batch_idx], dtype=np.int64)

        matrix_names = ["P_max", "P_freq", "C_max", "C_freq", "R_max", "R_freq"]
        for name in matrix_names:
            csr = _to_csr(getattr(self, name))
            kwargs[f"{name}_data"] = csr.data
            kwargs[f"{name}_indices"] = csr.indices
            kwargs[f"{name}_indptr"] = csr.indptr
            kwargs[f"{name}_shape"] = np.array(csr.shape, dtype=np.int64)

        np.savez(tmp_filepath, **kwargs)
        os.replace(tmp_filepath, filepath)

    @classmethod
    def load_npz(cls, filepath: str) -> FeatureMatrices:
        """Load feature matrices from disk .npz archive (supports sparse CSR & dense legacy formats)."""
        data = np.load(filepath)
        matrix_names = ["P_max", "P_freq", "C_max", "C_freq", "R_max", "R_freq"]

        if "P_max_data" in data:
            mats = {}
            for name in matrix_names:
                d = data[f"{name}_data"]
                ind = data[f"{name}_indices"]
                ptr = data[f"{name}_indptr"]
                shp = tuple(data[f"{name}_shape"])
                mats[name] = sp.csr_matrix((d, ind, ptr), shape=shp)
            if "example_ids" in data:
                ex_ids = data["example_ids"]
            else:
                logger.warning(f"Key 'example_ids' missing in '{filepath}'; generated default sequence IDs [0..{mats['P_max'].shape[0]-1}].")
                ex_ids = np.arange(mats["P_max"].shape[0], dtype=np.int64)
            return cls(
                example_ids=ex_ids,
                P_max=mats["P_max"],
                P_freq=mats["P_freq"],
                C_max=mats["C_max"],
                C_freq=mats["C_freq"],
                R_max=mats["R_max"],
                R_freq=mats["R_freq"],
            )
        else:
            logger.warning(f"File '{filepath}' contains legacy dense numpy format. Converting to sparse CSR matrix...")
            p_m = data["P_max"]
            if "example_ids" in data:
                ex_ids = data["example_ids"]
            else:
                logger.warning(f"Key 'example_ids' missing in legacy file '{filepath}'; generated default sequence IDs [0..{p_m.shape[0]-1}].")
                ex_ids = np.arange(p_m.shape[0], dtype=np.int64)
            return cls(
                example_ids=ex_ids,
                P_max=_to_csr(p_m),
                P_freq=_to_csr(data["P_freq"]),
                C_max=_to_csr(data["C_max"]),
                C_freq=_to_csr(data["C_freq"]),
                R_max=_to_csr(data["R_max"]),
                R_freq=_to_csr(data["R_freq"]),
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
        save_every_batches: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.hook_layer = hook_layer
        self.device = device
        self.batch_size = batch_size
        self.save_every_batches = save_every_batches

    def extract(
        self,
        examples: List[PreferenceExample],
        checkpoint_path: Optional[str] = None,
        use_checkpoint: bool = True,
        save_every_batches: Optional[int] = None,
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
            save_every_batches=save_every_batches or self.save_every_batches,
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
        save_every_batches: Optional[int] = None,
    ) -> FeatureMatrices:
        self.model.eval()

        if save_every_batches is None:
            save_every_batches = self.save_every_batches
        if save_every_batches is None:
            save_every_batches = max(1, 1000 // max(1, self.batch_size))

        d_sae = self.sae.cfg.d_sae
        N = len(examples)
        example_ids = np.array([ex.example_id for ex in examples], dtype=np.int64)

        chunks_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "chunks") if checkpoint_path else None
        partial_ckpt = checkpoint_path.replace(".npz", "_partial.npz") if checkpoint_path else None

        processed_samples = 0
        chunk_files: List[str] = []

        if use_checkpoint and chunks_dir and os.path.exists(chunks_dir):
            existing_chunks = sorted([f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".npz")])
            if existing_chunks:
                try:
                    for fname in existing_chunks:
                        fpath = os.path.join(chunks_dir, fname)
                        with np.load(fpath, mmap_mode="r") as d_hdr:
                            if "P_max_shape" in d_hdr:
                                n_c = int(d_hdr["P_max_shape"][0])
                            elif "example_ids" in d_hdr:
                                n_c = len(d_hdr["example_ids"])
                            else:
                                n_c = 0
                        processed_samples += n_c
                        chunk_files.append(fname)
                    logger.info(f"Found chunked checkpoint directory: {processed_samples:,} samples pre-extracted across {len(chunk_files)} chunks [ZERO RAM loaded].")
                except Exception as e:
                    logger.warning(f"Could not read chunk directory '{chunks_dir}': {e}. Starting fresh...")
                    processed_samples = 0
                    chunk_files.clear()
            elif partial_ckpt and os.path.exists(partial_ckpt):
                try:
                    logger.info(f"Converting legacy partial checkpoint '{partial_ckpt}' to incremental chunk format...")
                    os.makedirs(chunks_dir, exist_ok=True)
                    part_mats = FeatureMatrices.load_npz(partial_ckpt)
                    n_part = part_mats.P_max.shape[0]
                    legacy_chunk_file = os.path.join(chunks_dir, "chunk_0000000.npz")
                    part_mats.save_npz(legacy_chunk_file)
                    
                    processed_samples = n_part
                    chunk_files = ["chunk_0000000.npz"]
                    del part_mats
                    import gc
                    gc.collect()
                    logger.info(f"Converted legacy partial checkpoint cleanly. Processed samples: {processed_samples:,}.")
                except Exception as e:
                    logger.warning(f"Could not convert legacy partial checkpoint '{partial_ckpt}': {e}. Starting fresh...")
                    processed_samples = 0

        start_batch_idx = processed_samples // self.batch_size
        start_i = start_batch_idx * self.batch_size

        if start_i > 0:
            logger.info(f"Resuming SAE extraction from sample index {start_i:,} (batch {start_batch_idx:,}, batch_size={self.batch_size}) [ZERO RAM loaded].")

        # In-memory accumulator for CURRENT CHUNK ONLY (keeps RAM footprint <100 MB)
        curr_chunk_P_max: List[sp.csr_matrix] = []
        curr_chunk_P_freq: List[sp.csr_matrix] = []
        curr_chunk_C_max: List[sp.csr_matrix] = []
        curr_chunk_C_freq: List[sp.csr_matrix] = []
        curr_chunk_R_max: List[sp.csr_matrix] = []
        curr_chunk_R_freq: List[sp.csr_matrix] = []
        curr_chunk_ex_ids: List[int] = []

        chunk_start_i = start_i

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
                b_start = b_idx * self.batch_size
                b_end = min(b_start + self.batch_size, N)
                batch_exs = examples[b_start:b_end]

                # 2-pass preference batch extraction (Prompt+Chosen & Prompt+Rejected)
                p_m, p_f, c_m, c_f, r_m, r_f = self._process_preference_batch(batch_exs, target_layer, residual_container)

                curr_chunk_P_max.append(sp.csr_matrix(p_m, dtype=np.float32))
                curr_chunk_P_freq.append(sp.csr_matrix(p_f, dtype=np.float32))
                curr_chunk_C_max.append(sp.csr_matrix(c_m, dtype=np.float32))
                curr_chunk_C_freq.append(sp.csr_matrix(c_f, dtype=np.float32))
                curr_chunk_R_max.append(sp.csr_matrix(r_m, dtype=np.float32))
                curr_chunk_R_freq.append(sp.csr_matrix(r_f, dtype=np.float32))
                curr_chunk_ex_ids.extend([ex.example_id for ex in batch_exs])

                # Save current chunk incrementally every save_every_batches or at final batch
                if chunks_dir and ((b_idx + 1) % save_every_batches == 0 or b_idx == num_batches - 1):
                    if len(curr_chunk_P_max) > 0:
                        os.makedirs(chunks_dir, exist_ok=True)
                        chunk_filename = f"chunk_{chunk_start_i:07d}.npz"
                        chunk_filepath = os.path.join(chunks_dir, chunk_filename)
                        
                        chunk_mats = FeatureMatrices(
                            example_ids=np.array(curr_chunk_ex_ids, dtype=np.int64),
                            P_max=sp.vstack(curr_chunk_P_max, format="csr"),
                            P_freq=sp.vstack(curr_chunk_P_freq, format="csr"),
                            C_max=sp.vstack(curr_chunk_C_max, format="csr"),
                            C_freq=sp.vstack(curr_chunk_C_freq, format="csr"),
                            R_max=sp.vstack(curr_chunk_R_max, format="csr"),
                            R_freq=sp.vstack(curr_chunk_R_freq, format="csr"),
                        )
                        chunk_mats.save_npz(chunk_filepath)

                        if chunk_filename not in chunk_files:
                            chunk_files.append(chunk_filename)

                        # Reset in-memory chunk arrays to keep RAM < 100 MB
                        curr_chunk_P_max.clear()
                        curr_chunk_P_freq.clear()
                        curr_chunk_C_max.clear()
                        curr_chunk_C_freq.clear()
                        curr_chunk_R_max.clear()
                        curr_chunk_R_freq.clear()
                        curr_chunk_ex_ids.clear()
                        chunk_start_i = b_end
                        import gc
                        gc.collect()

        finally:
            handle.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Merge all chunks into final consolidated FeatureMatrices
        logger.info(f"Consolidating extraction chunks from '{chunks_dir}'...")
        all_P_max: List[sp.csr_matrix] = []
        all_P_freq: List[sp.csr_matrix] = []
        all_C_max: List[sp.csr_matrix] = []
        all_C_freq: List[sp.csr_matrix] = []
        all_R_max: List[sp.csr_matrix] = []
        all_R_freq: List[sp.csr_matrix] = []
        all_ex_ids: List[np.ndarray] = []

        if chunks_dir and os.path.exists(chunks_dir):
            existing_chunks = sorted([f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".npz")])
            for fname in existing_chunks:
                fpath = os.path.join(chunks_dir, fname)
                if os.path.exists(fpath):
                    m = FeatureMatrices.load_npz(fpath)
                    all_ex_ids.append(m.example_ids)
                    all_P_max.append(m.P_max)
                    all_P_freq.append(m.P_freq)
                    all_C_max.append(m.C_max)
                    all_C_freq.append(m.C_freq)
                    all_R_max.append(m.R_max)
                    all_R_freq.append(m.R_freq)

        final_matrices = FeatureMatrices(
            example_ids=np.concatenate(all_ex_ids) if all_ex_ids else example_ids,
            P_max=sp.vstack(all_P_max, format="csr") if all_P_max else sp.csr_matrix((N, d_sae), dtype=np.float32),
            P_freq=sp.vstack(all_P_freq, format="csr") if all_P_freq else sp.csr_matrix((N, d_sae), dtype=np.float32),
            C_max=sp.vstack(all_C_max, format="csr") if all_C_max else sp.csr_matrix((N, d_sae), dtype=np.float32),
            C_freq=sp.vstack(all_C_freq, format="csr") if all_C_freq else sp.csr_matrix((N, d_sae), dtype=np.float32),
            R_max=sp.vstack(all_R_max, format="csr") if all_R_max else sp.csr_matrix((N, d_sae), dtype=np.float32),
            R_freq=sp.vstack(all_R_freq, format="csr") if all_R_freq else sp.csr_matrix((N, d_sae), dtype=np.float32),
        )

        # Clean up temporary chunks directory after successful final matrix consolidation
        if chunks_dir and os.path.exists(chunks_dir):
            try:
                import shutil
                shutil.rmtree(chunks_dir)
                logger.info(f"Cleaned up temporary chunk directory '{chunks_dir}'.")
            except Exception as e:
                logger.warning(f"Could not remove temporary chunk directory '{chunks_dir}': {e}")

        return final_matrices

    def _process_preference_batch(
        self,
        batch_exs: List[PreferenceExample],
        target_layer: Any,
        residual_container: List[torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        residual_container.clear()
        B = len(batch_exs)
        d_sae = self.sae.cfg.d_sae

        # 1. Format prompts with chat template if present
        formatted_prompts = []
        for ex in batch_exs:
            if getattr(self.tokenizer, "chat_template", None) is not None:
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": ex.prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    formatted_prompts.append(formatted)
                except Exception:
                    formatted_prompts.append(ex.prompt)
            else:
                formatted_prompts.append(ex.prompt)

        # Tokenize prompts alone to extract exact prompt token lengths
        prompt_inputs = self.tokenizer(
            formatted_prompts,
            padding=False,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        p_lens = [len(ids) for ids in prompt_inputs["input_ids"]]

        # 2. Process Prompt + Chosen sequences
        chosen_texts = [p + ex.chosen for p, ex in zip(formatted_prompts, batch_exs)]
        c_inputs = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            self.model(**c_inputs)

        c_resid = residual_container[0]
        residual_container.clear()
        c_acts = self.sae.encode(c_resid.to(self.sae.device)).to(torch.float32)
        c_attn_mask = c_inputs.attention_mask.to(self.sae.device)

        p_m = np.zeros((B, d_sae), dtype=np.float32)
        p_f = np.zeros((B, d_sae), dtype=np.float32)
        c_m = np.zeros((B, d_sae), dtype=np.float32)
        c_f = np.zeros((B, d_sae), dtype=np.float32)

        for i in range(B):
            valid_len = int(c_attn_mask[i].sum().item())
            p_len = min(p_lens[i], valid_len)

            if p_len > 0:
                p_span = c_acts[i, :p_len]
                p_m[i] = torch.max(p_span, dim=0).values.detach().cpu().numpy()
                p_f[i] = (torch.sum((p_span > 0).float(), dim=0) / float(p_len)).detach().cpu().numpy()

            c_len = max(0, valid_len - p_len)
            if c_len > 0:
                c_span = c_acts[i, p_len:valid_len]
                c_m[i] = torch.max(c_span, dim=0).values.detach().cpu().numpy()
                c_f[i] = (torch.sum((c_span > 0).float(), dim=0) / float(c_len)).detach().cpu().numpy()

        # 3. Process Prompt + Rejected sequences
        rejected_texts = [p + ex.rejected for p, ex in zip(formatted_prompts, batch_exs)]
        r_inputs = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            self.model(**r_inputs)

        r_resid = residual_container[0]
        residual_container.clear()
        r_acts = self.sae.encode(r_resid.to(self.sae.device)).to(torch.float32)
        r_attn_mask = r_inputs.attention_mask.to(self.sae.device)

        r_m = np.zeros((B, d_sae), dtype=np.float32)
        r_f = np.zeros((B, d_sae), dtype=np.float32)

        for i in range(B):
            valid_len = int(r_attn_mask[i].sum().item())
            p_len = min(p_lens[i], valid_len)

            r_len = max(0, valid_len - p_len)
            if r_len > 0:
                r_span = r_acts[i, p_len:valid_len]
                r_m[i] = torch.max(r_span, dim=0).values.detach().cpu().numpy()
                r_f[i] = (torch.sum((r_span > 0).float(), dim=0) / float(r_len)).detach().cpu().numpy()

        return p_m, p_f, c_m, c_f, r_m, r_f

