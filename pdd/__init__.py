"""Predictive Data Debugging (PDD) package.

Kept import-light: only the config/logging surface is loaded eagerly so the
viewer (and any `import pdd.*`) starts fast without pulling torch/datasets/
scipy/sklearn. Heavy submodules are imported lazily on attribute access.
"""
from .config import (
    PipelineConfig,
    ModelConfig,
    SAEConfig,
    DataConfig,
    FeatureConditionedConfig,
    PromptConditionedConfig,
)
from .logger import get_logger

__all__ = [
    "PipelineConfig",
    "ModelConfig",
    "SAEConfig",
    "DataConfig",
    "FeatureConditionedConfig",
    "PromptConditionedConfig",
    "DatasetLoader",
    "PreferenceExample",
    "FeatureMatrices",
    "FeatureMatrixExtractor",
    "FeatureClusterMap",
    "LeidenFeatureClusterer",
    "FeatureConditionedPipeline",
    "FeatureConditionedResult",
    "PromptConditionedPipeline",
    "PromptConditionedResult",
    "PDDPipeline",
    "ModelBackend",
    "SAEBackend",
    "ValidationMetrics",
    "compute_prediction_validation_metrics",
    "DatasetInoculator",
    "LossReweighter",
    "FeatureSteerer",
    "ClusterAutoLabeler",
    "ClusterLabel",
    "AutoLabelConfig",
    "get_logger",
]

_LAZY = {
    "DatasetLoader": ("data", "DatasetLoader"),
    "PreferenceExample": ("data", "PreferenceExample"),
    "FeatureMatrices": ("feature_matrices", "FeatureMatrices"),
    "FeatureMatrixExtractor": ("feature_matrices", "FeatureMatrixExtractor"),
    "FeatureClusterMap": ("feature_clusters", "FeatureClusterMap"),
    "LeidenFeatureClusterer": ("feature_clusters", "LeidenFeatureClusterer"),
    "FeatureConditionedPipeline": ("feature_conditioned", "FeatureConditionedPipeline"),
    "FeatureConditionedResult": ("feature_conditioned", "FeatureConditionedResult"),
    "PromptConditionedPipeline": ("prompt_conditioned", "PromptConditionedPipeline"),
    "PromptConditionedResult": ("prompt_conditioned", "PromptConditionedResult"),
    "PDDPipeline": ("pipeline", "PDDPipeline"),
    "ModelBackend": ("sae", "ModelBackend"),
    "SAEBackend": ("sae", "SAEBackend"),
    "ValidationMetrics": ("validation", "ValidationMetrics"),
    "compute_prediction_validation_metrics": ("validation", "compute_prediction_validation_metrics"),
    "DatasetInoculator": ("interventions", "DatasetInoculator"),
    "LossReweighter": ("interventions", "LossReweighter"),
    "FeatureSteerer": ("interventions", "FeatureSteerer"),
    "ClusterAutoLabeler": ("autolabel", "ClusterAutoLabeler"),
    "ClusterLabel": ("autolabel", "ClusterLabel"),
    "AutoLabelConfig": ("config", "AutoLabelConfig"),
}


def __getattr__(name: str):
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module 'pdd' has no attribute '{name}'")
    module_name, attr = entry
    import importlib
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value