"""SAE & Model Backends: Clean OOP loading for Transformer Models & Sparse Autoencoders."""
from __future__ import annotations

import sys
# Kaggle/Colab compatibility: block torchvision/torchaudio from being imported by transformers.
# PDD is a pure NLP pipeline and never uses these packages.
# IMPORTANT: do NOT call __import__ here — on the local server torchvision/torchaudio ARE installed
# and __import__ would fully load their C++ extension (_C.so), which registers fake ops via
# torch.library.register_fake and corrupts PyTorch 2.4's Autograd dispatcher, breaking loss.backward().
# Only block packages that aren't already in sys.modules (i.e. haven't been imported yet).
for _pkg in ("torchvision", "torchaudio"):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = None
        sys.modules[f"{_pkg}.io"] = None
        sys.modules[f"{_pkg}.ops"] = None

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Tuple

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

        kwargs = {}
        if "gemma" in self.cfg.path.lower():
            kwargs["attn_implementation"] = "eager"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **kwargs,
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
        elif self.cfg.type == "batch_topk":
            self.sae = self._load_batch_topk()
        elif self.cfg.type == "sae_lens":
            self.sae = self._load_sae_lens()
        elif self.cfg.type == "auto":
            # Auto-detection: check if Qwen repo format or try sae_lens from_pretrained
            if "adamkarvonen" in self.cfg.repo.lower() or "batch_top_k" in self.cfg.repo.lower() or "batchtopk" in self.cfg.repo.lower():
                logger.info("Auto-detected BatchTopK / dictionary_learning repository layout. Using BatchTopK loader.")
                self.sae = self._load_batch_topk()
            elif "qwen" in self.cfg.repo.lower() and "sae-res" in self.cfg.repo.lower():
                logger.info("Auto-detected Qwen-Scope single-file (.sae.pt) repository layout. Using Qwen-Scope loader.")
                self.sae = self._load_qwen_scope()
            else:
                try:
                    logger.info("Attempting standard sae_lens from_pretrained loader...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.sae = self._load_sae_lens()
                except Exception as e:
                    logger.warning(f"sae_lens from_pretrained failed on GPU ({e}). Retrying sae_lens loader on CPU...")
                    saved_device = self.cfg.device
                    self.cfg.device = "cpu"
                    try:
                        self.sae = self._load_sae_lens()
                    finally:
                        self.cfg.device = saved_device
        else:
            raise ValueError(f"Unsupported SAE type: '{self.cfg.type}'")

        if hasattr(self.sae, "cfg"):
            if self.cfg.d_in is None:
                self.cfg.d_in = getattr(self.sae.cfg, "d_in", None)
            if self.cfg.d_sae is None:
                self.cfg.d_sae = getattr(self.sae.cfg, "d_sae", None)

        target_device = "cpu" if self.cfg.sae_cpu else self.cfg.device
        logger.info(f"Setting SAE execution device to '{target_device}' (sae_cpu={self.cfg.sae_cpu})...")
        self.sae.to(target_device)
        self.sae.cfg.device = target_device

        return self.sae


    def _hf_download(self, repo: str, filename: str) -> str:
        """Resolve a hub file locally first; fall back to a network download (logged)."""
        try:
            return hf_hub_download(repo, filename, local_files_only=True)
        except Exception:
            logger.info(f"SAE weights not found in local HF cache ({repo}/{filename}); downloading from hub...")
            return hf_hub_download(repo, filename)

    def _load_qwen_scope(self) -> Any:
        from sae_lens import SAE

        path = self._hf_download(self.cfg.repo, f"layer{self.cfg.layer}.sae.pt")
        state = torch.load(path, map_location="cpu", weights_only=True)

        weights = {
            "W_enc": state["W_enc"].T.contiguous(),
            "W_dec": state["W_dec"].T.contiguous(),
            "b_enc": state["b_enc"].contiguous(),
            "b_dec": state["b_dec"].contiguous(),
        }
        d_in = int(weights["W_enc"].shape[0])
        d_sae = int(weights["W_enc"].shape[1])
        k_val = self.cfg.k if self.cfg.k is not None else 50
        target_device = "cpu" if self.cfg.sae_cpu else self.cfg.device
        cfg_dict = {
            "architecture": "topk",
            "d_in": d_in,
            "d_sae": d_sae,
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
            "device": target_device,
            "sae_lens_training_version": "qwen-scope-0.0.1",
            "neuronpedia_id": None,
            **weights,
        }
        return SAE.from_dict(cfg_dict)

    def _load_batch_topk(self) -> Any:
        from sae_lens import SAE

        subpath = (
            self.cfg.sae_id
            or f"saes_Qwen_Qwen3-1.7B_batch_top_k/resid_post_layer_{self.cfg.layer}/trainer_0/ae.pt"
        )
        if not subpath.endswith(".pt"):
            subpath = f"{subpath}/ae.pt"

        path = self._hf_download(self.cfg.repo, subpath)
        state = torch.load(path, map_location="cpu", weights_only=True)

        if "encoder.weight" in state:
            weights = {
                "W_enc": state["encoder.weight"].T.contiguous(),
                "W_dec": state["decoder.weight"].T.contiguous(),
                "b_enc": state["encoder.bias"].contiguous(),
                "b_dec": state["b_dec"].contiguous(),
            }
        else:
            d_in_ref = self.cfg.d_in or 2048
            w_enc = state["W_enc"]
            w_dec = state["W_dec"]
            weights = {
                "W_enc": w_enc.T.contiguous() if w_enc.shape[0] != d_in_ref else w_enc.contiguous(),
                "W_dec": w_dec.T.contiguous() if w_dec.shape[1] != d_in_ref else w_dec.contiguous(),
                "b_enc": state["b_enc"].contiguous(),
                "b_dec": state["b_dec"].contiguous(),
            }

        d_in = int(weights["W_enc"].shape[0])
        d_sae = int(weights["W_enc"].shape[1])
        k_val = int(state.get("k", self.cfg.k or 80))
        target_device = "cpu" if self.cfg.sae_cpu else self.cfg.device

        cfg_dict = {
            "architecture": "topk",
            "d_in": d_in,
            "d_sae": d_sae,
            "activation_fn_str": "topk",
            "activation_fn_kwargs": {"k": k_val},
            "apply_b_dec_to_input": True,
            "finetuning_scaling_factor": False,
            "context_size": 2048,
            "model_name": "qwen3-1.7b",
            "hook_name": HOOK_FORMAT.format(layer=self.cfg.layer),
            "hook_layer": self.cfg.layer,
            "hook_head_index": None,
            "prepend_bos": False,
            "dataset_path": "",
            "dataset_trust_remote_code": False,
            "normalize_activations": "none",
            "dtype": "float32",
            "device": target_device,
            "sae_lens_training_version": "0.0.1",
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