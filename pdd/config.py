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
    type: str = "auto"                 # "auto", "qwen_scope", or "sae_lens"
    repo: str = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
    sae_id: Optional[str] = None      # Optional ID for sae_lens releases
    layer: int = 14
    d_in: int = 2048
    d_sae: int = 32768
    k: Optional[int] = 50
    device: str = "cuda"

    def validate(self) -> None:
        if self.type not in ("auto", "qwen_scope", "sae_lens"):
            raise ValueError(f"Unsupported SAE type: '{self.type}'. Must be 'auto', 'qwen_scope', or 'sae_lens'.")
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

    def validate(self) -> None:
        if not self.path:
            raise ValueError("DataConfig.path cannot be empty.")
        if self.batch_size <= 0:
            raise ValueError(f"DataConfig.batch_size must be positive, got {self.batch_size}.")


@dataclass
class FeatureConditionedConfig:
    tau: float = 0.01
    silent_pct: float = 5.0
    n_data_clusters: int = 512
    min_feat_cluster_size: int = 10
    min_data_cluster_size: int = 25
    split_half_eps: float = 1e-6

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
    feature_conditioned: FeatureConditionedConfig = field(default_factory=FeatureConditionedConfig)
    prompt_conditioned: PromptConditionedConfig = field(default_factory=PromptConditionedConfig)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("PipelineConfig.name cannot be empty.")
        self.model.validate()
        self.sae.validate()
        self.data.validate()
        self.feature_conditioned.validate()
        self.prompt_conditioned.validate()


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
        fc_cfg = FeatureConditionedConfig(**data.get("feature_conditioned", {}))
        pc_cfg = PromptConditionedConfig(**data.get("prompt_conditioned", {}))

        top_kwargs = {k: v for k, v in data.items() if k not in ("model", "sae", "data", "feature_conditioned", "prompt_conditioned")}

        config = cls(
            model=model_cfg,
            sae=sae_cfg,
            data=data_cfg,
            feature_conditioned=fc_cfg,
            prompt_conditioned=pc_cfg,
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