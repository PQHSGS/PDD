"""Configuration dataclasses with JSON loading/saving and validation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    path: str = "Qwen/Qwen3-1.7B-Base"
    dtype: str = "bfloat16"
    device: str = "cuda"

    def validate(self) -> None:
        if not self.path:
            raise ValueError("ModelConfig.path cannot be empty.")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"Unsupported dtype: '{self.dtype}'. Allowed: ['bfloat16', 'float16', 'float32'].")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"Unsupported device: '{self.device}'. Allowed: ['cuda', 'cpu'].")


@dataclass
class SAEConfig:
    type: str = "auto"                 # "auto", "qwen_scope", "batch_topk", or "sae_lens"
    repo: str = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
    sae_id: Optional[str] = None      # Optional ID for sae_lens releases
    layer: int = 14
    d_in: Optional[int] = None        # Auto-inferred from checkpoint if None
    d_sae: Optional[int] = None       # Auto-inferred from checkpoint if None
    k: Optional[int] = None           # Auto-inferred from checkpoint if None
    device: str = "cuda"
    sae_cpu: bool = False

    def validate(self) -> None:
        if self.type not in ("auto", "qwen_scope", "batch_topk", "sae_lens"):
            raise ValueError(f"Unsupported SAE type: '{self.type}'. Must be 'auto', 'qwen_scope', 'batch_topk', or 'sae_lens'.")
        if not self.repo:
            raise ValueError("SAEConfig.repo cannot be empty.")
        if self.layer < 0:
            raise ValueError(f"Invalid SAE layer index: {self.layer}.")



@dataclass
class DataConfig:
    path: str = "allenai/Dolci-Instruct-DPO"
    split: str = "train"
    prompt_col: str = "prompt"
    chosen_col: str = "chosen"
    rejected_col: str = "rejected"
    max_samples: int = -1
    batch_size: int = 8
    datasets_dir: str = "datasets"
    save_every_batches: int = 100

    def validate(self) -> None:
        if not self.path:
            raise ValueError("DataConfig.path cannot be empty.")
        if self.batch_size <= 0:
            raise ValueError(f"DataConfig.batch_size must be positive, got {self.batch_size}.")
        if self.save_every_batches <= 0:
            raise ValueError(f"DataConfig.save_every_batches must be positive, got {self.save_every_batches}.")



@dataclass
class FeatureConditionedConfig:
    tau: float = 0.01
    silent_pct: float = 5.0
    n_data_clusters: int = 512
    min_feat_cluster_size: int = 10
    min_data_cluster_size: int = 25
    split_half_eps: float = 1e-6
    weighted_disparity: bool = False

    def validate(self) -> None:
        if self.tau < 0:
            raise ValueError("tau must be non-negative.")
        if not (0.0 <= self.silent_pct <= 100.0):
            raise ValueError("silent_pct must be between 0 and 100.")


@dataclass
class PromptConditionedConfig:
    min_prompt_count: int = 200
    min_resp_count: int = 200
    min_resp_sigma: float = 1e-3
    min_resp_abs_delta: float = 1e-4
    n_sample_emb: int = 30000
    n_svd: int = 128
    n_prompt_clusters: int = 512
    n_resp_clusters: int = 512

    def validate(self) -> None:
        if self.n_svd <= 0:
            raise ValueError("n_svd must be positive.")
        if self.n_prompt_clusters <= 0:
            raise ValueError("n_prompt_clusters must be positive.")
        if self.n_resp_clusters <= 0:
            raise ValueError("n_resp_clusters must be positive.")


@dataclass
class AutoLabelConfig:
    """Final pipeline stage: auto-interpretation of every cluster level for the viewer.

    Pass 1 labels data clusters B_k (LLM on real centroid/random sampled prompts),
    Pass 2 labels SAE feature clusters T_m (LLM on real response examples firing
    each cluster), Pass 3 maps prompt clusters A_k / response-delta clusters R_m to
    their strongest real examples. All artifacts land in ``<run>/`` (cluster_labels.json,
    feature_cluster_labels.json, prompt_conditioned_cluster_examples.json).
    """
    enabled: bool = True
    label_model: str = "google/gemma-3-4b-it" # Local chat model for LLM labels
    heuristic: bool = False                   # Keyword labels instead of the local LLM
    num_clusters: int = -1                    # Data clusters B_k to label (-1 = all active)
    skip_feature_clusters: bool = False       # Skip Pass 2 (T_m whole-cluster labels)
    skip_pc_examples: bool = False            # Skip Pass 3 (A_k / R_m example indices)
    pc_n_top: int = 15                        # Examples per A_k / R_m in Pass 3
    max_prompt_chars: int = 600               # Max text length shown to the LLM
    max_examples: int = 15                    # Max examples per cluster shown to the LLM

    def validate(self) -> None:
        if not self.label_model:
            raise ValueError("AutoLabelConfig.label_model cannot be empty.")
        if self.pc_n_top <= 0:
            raise ValueError("pc_n_top must be positive.")
        if self.max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive.")
        if self.max_examples <= 0:
            raise ValueError("max_examples must be positive.")


@dataclass
class FeatureClusterConfig:
    """SAE feature clustering (binary MI graph + Leiden) settings.

    Goodfire/paper defaults (arXiv:2606.12360, App. "SAE Feature Clusters"):
    top 1% of off-diagonal MI pairs, communities with fewer than 4 features
    filtered out.
    """
    top_pct: float = 1.0               # Keep the top top_pct% of positive-MI pairs (paper: 1.0)
    min_community_size: int = 4        # Drop Leiden communities with fewer features (paper: 4)
    min_firing_freq: float = 1e-4      # Restrict MI graph to features firing >= this fraction of rows
    block_size: int = 2048             # MI co-occurrence block size (performance, not a paper knob)
    resolution_parameter: float = 1.5  # Leiden RBConfiguration resolution (unset in paper)

    def validate(self) -> None:
        if not (0.0 < self.top_pct <= 100.0):
            raise ValueError("top_pct must be in (0, 100].")
        if self.min_community_size < 1:
            raise ValueError("min_community_size must be >= 1.")
        if self.min_firing_freq < 0.0:
            raise ValueError("min_firing_freq must be non-negative.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.resolution_parameter <= 0.0:
            raise ValueError("resolution_parameter must be positive.")


@dataclass
class DPOValidationConfig:
    """Experiment 4 DPO Training & Rollout Validation settings (paper Table 8 & §4)."""
    train_samples: int = -1             # Preference pairs for DPO training (-1 = full dataset)
    eval_prompts: int = 500             # Held-out evaluation prompts for text rollouts
    batch_size: int = 4                 # Micro-batch size per GPU step (high throughput, ~3.2 GB VRAM)
    grad_accum: int = 16                # Gradient accumulation steps (eff_bs = 64)
    lr: float = 5e-5                    # Learning rate (5e-5 for LoRA, 1e-6 for full finetuning)
    beta: float = 2.0                   # DPO beta parameter (author uses 2.0 on Dolci, Table 8)
    epochs: int = 1                     # DPO training epochs (paper: 1 epoch)
    lora_rank: int = 16                 # LoRA rank (16 = low-VRAM < 3.5 GB safe for shared GPUs)
    warmup_ratio: float = 0.1           # Linear LR warmup ratio (paper: 0.1)
    temperature: float = 0.0            # Rollout decoding temperature (0.0 = greedy, noise-free)
    max_length: int = 512               # Max sequence length (VRAM safe)
    num_features: int = 50              # Top feature clusters for R^2 evaluation

    def validate(self) -> None:
        if self.train_samples <= 0 and self.train_samples != -1:
            raise ValueError("train_samples must be positive or -1 (for full dataset).")
        if self.eval_prompts <= 0:
            raise ValueError("eval_prompts must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.grad_accum <= 0:
            raise ValueError("grad_accum must be positive.")


@dataclass
class PipelineConfig:
    name: str = "qwen3_dolci_default"
    seed: int = 0
    output_dir: str = "runs/default"
    checkpoint_dir: str = "checkpoints"
    use_checkpoint: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    feature_clusters: FeatureClusterConfig = field(default_factory=FeatureClusterConfig)
    feature_conditioned: FeatureConditionedConfig = field(default_factory=FeatureConditionedConfig)
    prompt_conditioned: PromptConditionedConfig = field(default_factory=PromptConditionedConfig)
    auto_label: AutoLabelConfig = field(default_factory=AutoLabelConfig)
    dpo_validation: DPOValidationConfig = field(default_factory=DPOValidationConfig)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("PipelineConfig.name cannot be empty.")
        self.model.validate()
        self.sae.validate()
        self.data.validate()
        self.feature_clusters.validate()
        self.feature_conditioned.validate()
        self.prompt_conditioned.validate()
        self.auto_label.validate()
        self.dpo_validation.validate()


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, json_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        model_cfg = ModelConfig(**data.get("model", {}))
        sae_cfg = SAEConfig(**data.get("sae", {}))
        data_cfg = DataConfig(**data.get("data", {}))
        fc_cfg = FeatureClusterConfig(**data.get("feature_clusters", {}))
        fcd_cfg = FeatureConditionedConfig(**data.get("feature_conditioned", {}))
        pc_cfg = PromptConditionedConfig(**data.get("prompt_conditioned", {}))
        al_cfg = AutoLabelConfig(**data.get("auto_label", {}))
        val_cfg = DPOValidationConfig(**data.get("dpo_validation", {}))

        top_kwargs = {k: v for k, v in data.items() if k not in ("model", "sae", "data", "feature_clusters", "feature_conditioned", "prompt_conditioned", "auto_label", "dpo_validation")}

        config = cls(
            model=model_cfg,
            sae=sae_cfg,
            data=data_cfg,
            feature_clusters=fc_cfg,
            feature_conditioned=fcd_cfg,
            prompt_conditioned=pc_cfg,
            auto_label=al_cfg,
            dpo_validation=val_cfg,
            **top_kwargs
        )
        config.validate()
        return config

    @classmethod
    def load_json(cls, json_path: str) -> PipelineConfig:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)