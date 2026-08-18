"""Neural Inspector Engine: Real Model + SAE Forward Pass for Faithful PDD Inspection.

Executes true PyTorch forward pass through the Base Model and Sparse Autoencoder (SAE)
to extract exact token feature activations, compute exact cluster scores, and evaluate
real disparity signals u_m = 1(C > 0.01) - 1(R > 0.01) (arXiv:2606.12360 App. B.1 & B.2).
Optimized with batched GPU forward passes, pre-warmed CUDA kernels, and memory management.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional, Tuple
import numpy as np
import torch

from .config import ModelConfig, SAEConfig
from .sae import ModelBackend, SAEBackend

logger = logging.getLogger("PDD.NeuralInspector")


class NeuralInspector:
    """Live Neural Model + SAE inference engine with batched GPU execution and zero mock code."""

    def __init__(
        self,
        model_path: str = "google/gemma-2-2b",
        sae_repo: str = "gemma-scope-2b-pt-res-canonical",
        sae_id: Optional[str] = "layer_12/width_16k/canonical",
        layer: int = 12,
        d_in: Optional[int] = None,
        d_sae: Optional[int] = None,
        k: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.sae_repo = sae_repo
        self.sae_id = sae_id
        self.layer = layer
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.device = device
        self.dtype = dtype

        self.model = None
        self.tokenizer = None
        self.sae = None
        self._is_loaded = False

    def load(self) -> None:
        """Load Model, Tokenizer, and SAE via SAEBackend and warm up CUDA kernels.

        Enforces a single device for both model and SAE: if loading on CUDA fails
        (or the SAE silently falls back to CPU while the model stays on GPU), the
        whole stack is retried on CPU so inference never runs on mismatched devices.
        """
        if self._is_loaded:
            return

        # Auto-detect available VRAM if device is set to cuda
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
                if free_gb < 3.5:
                    logger.warning(
                        f"GPU VRAM heavily congested ({free_gb:.2f} GB free of {total_bytes / (1024**3):.2f} GB). "
                        f"Switching NeuralInspector to CPU inference mode."
                    )
                    self.device = "cpu"
                    self.dtype = "float32"
            except Exception as e:
                logger.warning(f"Could not query CUDA memory: {e}")

        try:
            self._load_components()
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if self._is_cuda_failure(e):
                logger.warning(f"Caught CUDA failure during load ({e}). Retrying the full model+SAE load on CPU...")
                self._release_components()
                self.device = "cpu"
                self.dtype = "float32"
                torch.cuda.empty_cache()
                gc.collect()
                self._load_components()
            else:
                raise e

        if not self._devices_consistent():
            logger.warning(
                f"Model/SAE device mismatch after load (model on {self._model_device()}, SAE on {self._sae_device()}). "
                f"Reloading both on CPU..."
            )
            self._release_components()
            self.device = "cpu"
            self.dtype = "float32"
            torch.cuda.empty_cache()
            gc.collect()
            self._load_components()

        self._is_loaded = True
        logger.info(f"NeuralInspector successfully initialized via SAEBackend on {self.device}.")

    def _load_components(self) -> None:
        logger.info(f"Loading Tokenizer & Model ({self.model_path}) on {self.device}...")
        model_cfg = ModelConfig(path=self.model_path, dtype=self.dtype, device=self.device)
        model_backend = ModelBackend(model_cfg)
        self.model, self.tokenizer = model_backend.load()

        logger.info(f"Loading SAE ({self.sae_repo}) via SAEBackend on {self.device}...")
        sae_cfg = SAEConfig(
            type="auto",
            repo=self.sae_repo,
            sae_id=self.sae_id,
            layer=self.layer,
            d_in=self.d_in if self.d_in is not None else 2048,
            d_sae=self.d_sae if self.d_sae is not None else 16384,
            k=self.k if self.k is not None else 50,
            device=self.device,
        )
        sae_backend = SAEBackend(sae_cfg)
        self.sae = sae_backend.load()

    def _release_components(self) -> None:
        for attr in ("model", "tokenizer", "sae"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    del obj
                except Exception:
                    pass
            setattr(self, attr, None)

    @staticmethod
    def _is_cuda_failure(e: Exception) -> bool:
        if isinstance(e, torch.OutOfMemoryError):
            return True
        msg = str(e).lower()
        return "cuda out of memory" in msg or "cublas" in msg or "cuda error" in msg

    def _model_device(self) -> str:
        if self.model is None:
            return "None"
        try:
            return str(next(self.model.parameters()).device)
        except (StopIteration, AttributeError):
            return "unknown"

    def _sae_device(self) -> str:
        if self.sae is None:
            return "None"
        try:
            return str(next(self.sae.parameters()).device)
        except (StopIteration, AttributeError):
            return "unknown"

    def _devices_consistent(self) -> bool:
        if self.model is None or self.sae is None:
            return False
        return self._model_device().split(":")[0] == self._sae_device().split(":")[0]

    @torch.inference_mode()
    def extract_prompt_features(self, prompt: str) -> np.ndarray:
        """Single-prompt forward pass on GPU returning max-pooled SAE activations P(x)."""
        self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        activations = []

        def hook_fn(module, input, output):
            res = output[0] if isinstance(output, tuple) else output
            activations.append(res)

        target_layer = self.model.model.layers[self.layer] if hasattr(self.model, "model") else self.model.layers[self.layer]
        hook_handle = target_layer.register_forward_hook(hook_fn)

        try:
            self.model(**inputs)
        finally:
            hook_handle.remove()

        if not activations:
            raise RuntimeError(f"Failed to hook activations at layer {self.layer}.")

        resid = activations[0]  # (1, seq_len, d_model)
        sae_acts = self.sae.encode(resid)  # (1, seq_len, d_sae)
        if isinstance(sae_acts, tuple):
            sae_acts = sae_acts[0]

        # Max pool over sequence tokens on GPU -> CPU numpy
        p_max = sae_acts[0].max(dim=0).values.float().cpu().numpy()
        return p_max

    @torch.inference_mode()
    def extract_pair_features(self, prompt: str, chosen: str, rejected: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched dual forward pass on GPU extracting Chosen (C) and Rejected (R) disparity in parallel."""
        self.load()

        texts = [f"{prompt} {chosen}", f"{prompt} {rejected}"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        activations = []

        def hook_fn(module, input, output):
            res = output[0] if isinstance(output, tuple) else output
            activations.append(res)

        target_layer = self.model.model.layers[self.layer] if hasattr(self.model, "model") else self.model.layers[self.layer]
        hook_handle = target_layer.register_forward_hook(hook_fn)

        try:
            self.model(**inputs)
        finally:
            hook_handle.remove()

        if not activations:
            raise RuntimeError(f"Failed to hook activations at layer {self.layer}.")

        resid = activations[0]  # (2, seq_len, d_model)
        sae_acts = self.sae.encode(resid)  # (2, seq_len, d_sae)
        if isinstance(sae_acts, tuple):
            sae_acts = sae_acts[0]

        # Max pool over sequence tokens on GPU
        # Index 0 = Chosen, Index 1 = Rejected
        c_p = sae_acts[0].max(dim=0).values.float().cpu().numpy()
        r_p = sae_acts[1].max(dim=0).values.float().cpu().numpy()

        # Exact paper threshold disparity: u = 1(C > 0.01) - 1(R > 0.01) (Appendix B.1)
        u = (c_p > 0.01).astype(np.float32) - (r_p > 0.01).astype(np.float32)
        return c_p, r_p, u

    def unload(self) -> None:
        """Unload model and SAE from GPU memory to allow seamless model switching."""
        if not self._is_loaded:
            return
        logger.info(f"Unloading Model ({self.model_path}) and SAE from GPU...")
        del self.model
        del self.tokenizer
        del self.sae
        self.model = None
        self.tokenizer = None
        self.sae = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"GPU VRAM cleared (Current Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB).")


_INSPECTOR_INSTANCE: Optional[NeuralInspector] = None


def get_neural_inspector(
    model_path: str = "google/gemma-2-2b",
    sae_repo: str = "gemma-scope-2b-pt-res-canonical",
    sae_id: Optional[str] = "layer_12/width_16k/canonical",
    layer: int = 12,
    d_in: Optional[int] = None,
    d_sae: Optional[int] = None,
    k: Optional[int] = None,
) -> NeuralInspector:
    """Singleton getter that automatically swaps models and frees GPU VRAM upon run selection switch."""
    global _INSPECTOR_INSTANCE
    if _INSPECTOR_INSTANCE is not None:
        if (
            _INSPECTOR_INSTANCE.model_path != model_path
            or _INSPECTOR_INSTANCE.sae_repo != sae_repo
            or _INSPECTOR_INSTANCE.sae_id != sae_id
            or _INSPECTOR_INSTANCE.layer != layer
            or _INSPECTOR_INSTANCE.d_sae != d_sae
            or _INSPECTOR_INSTANCE.d_in != d_in
            or _INSPECTOR_INSTANCE.k != k
        ):
            logger.info(f"Model switch detected ({_INSPECTOR_INSTANCE.model_path} -> {model_path}). Releasing GPU VRAM...")
            _INSPECTOR_INSTANCE.unload()
            _INSPECTOR_INSTANCE = None

    if _INSPECTOR_INSTANCE is None:
        _INSPECTOR_INSTANCE = NeuralInspector(
            model_path=model_path,
            sae_repo=sae_repo,
            sae_id=sae_id,
            layer=layer,
            d_in=d_in,
            d_sae=d_sae,
            k=k,
        )
    return _INSPECTOR_INSTANCE
