"""SAE & Model Backends: Clean OOP loading for Transformer Models & Sparse Autoencoders."""
from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Tuple, Optional

from .config import ModelConfig, SAEConfig
from .logger import get_logger

logger = get_logger("PDD.SAE")

# sae-lens hook format
HOOK_FORMAT = "blocks.{layer}.hook_resid_stream"


class ModelBackend:
    """Encapsulates Hugging Face causal LM model and tokenizer."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def load(self) -> Tuple[Any, Any]:
        """Load tokenizer and model with specified device and precision."""
        logger.info(f"Loading tokenizer from {self.cfg.path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token



        logger.info(f"Loading model from {self.cfg.path} (dtype={self.cfg.dtype}, device={self.cfg.device})...")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16) if self.cfg.device != "cpu" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if self.cfg.device != "cpu":
            self.model = self.model.to(self.cfg.device)

        self.model.eval()
        return self.model, self.tokenizer


class SAEBackend:
    """Unified Sparse Autoencoder Backend with Automatic Fallback."""

    def __init__(self, cfg: SAEConfig):
        self.cfg = cfg
        self.sae = None

    def load(self) -> Any:
        """Load SAE instance using auto-detection or explicit configuration."""
        logger.info(f"Loading SAE (type={self.cfg.type}, repo={self.cfg.repo}, layer={self.cfg.layer})...")

        if self.cfg.type == "qwen_scope":
            self.sae = self._load_qwen_scope()
        elif self.cfg.type == "sae_lens":
            self.sae = self._load_sae_lens()
        elif self.cfg.type == "auto":
            # Auto-detection: check if Qwen repo format or try sae_lens from_pretrained
            if "qwen" in self.cfg.repo.lower() and "sae-res" in self.cfg.repo.lower():
                logger.info("Auto-detected Qwen-Scope single-file (.sae.pt) repository layout. Using Qwen-Scope loader.")
                self.sae = self._load_qwen_scope()
            else:
                try:
                    logger.info("Attempting standard sae_lens from_pretrained loader...")
                    self.sae = self._load_sae_lens()
                except Exception as e:
                    logger.warning(f"sae_lens from_pretrained failed ({e}). Falling back to custom Qwen-Scope loader...")
                    self.sae = self._load_qwen_scope()
        else:
            raise ValueError(f"Unsupported SAE type: '{self.cfg.type}'")

        if self.cfg.device != "cpu":
            self.sae.to(self.cfg.device)
            self.sae.cfg.device = self.cfg.device

        return self.sae


    def _load_qwen_scope(self) -> Any:
        from sae_lens import SAE

        path = hf_hub_download(self.cfg.repo, f"layer{self.cfg.layer}.sae.pt")
        state = torch.load(path, map_location="cpu", weights_only=True)

        weights = {
            "W_enc": state["W_enc"].T.contiguous(),
            "W_dec": state["W_dec"].T.contiguous(),
            "b_enc": state["b_enc"].contiguous(),
            "b_dec": state["b_dec"].contiguous(),
        }
        k_val = self.cfg.k if self.cfg.k is not None else 50
        cfg_dict = {
            "architecture": "topk",
            "d_in": self.cfg.d_in,
            "d_sae": self.cfg.d_sae,
            "activation_fn_str": "topk",
            "activation_fn_kwargs": {"k": k_val},
            "apply_b_dec_to_input": True,
            "finetuning_scaling_factor": False,
            "context_size": 2048,
            "model_name": "qwen3-1.7b-base",
            "hook_name": HOOK_FORMAT.format(layer=self.cfg.layer),
            "hook_layer": self.cfg.layer,
            "hook_head_index": None,
            "prepend_bos": False,
            "dataset_path": "",
            "dataset_trust_remote_code": False,
            "normalize_activations": "none",
            "dtype": "float32",
            "device": self.cfg.device,
            "sae_lens_training_version": "qwen-scope-0.0.1",
            "neuronpedia_id": None,
            **weights,
        }
        return SAE.from_dict(cfg_dict)

    def _load_sae_lens(self) -> Any:
        from sae_lens import SAE

        sae_id = self.cfg.sae_id or f"layer_{self.cfg.layer}/width_16k/canonical"
        sae, _, _ = SAE.from_pretrained(
            release=self.cfg.repo,
            sae_id=sae_id,
            device=self.cfg.device,
        )
        return sae