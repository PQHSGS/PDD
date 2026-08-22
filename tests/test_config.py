"""Config dataclass validation + JSON round-trip tests."""
import pytest

from pdd.config import (
    AutoLabelConfig,
    DataConfig,
    FeatureClusterConfig,
    FeatureConditionedConfig,
    ModelConfig,
    PipelineConfig,
    PromptConditionedConfig,
    SAEConfig,
)


def test_defaults_validate():
    PipelineConfig().validate()


@pytest.mark.parametrize("bad_dtype", ["fp64", "int8", ""])
def test_model_config_rejects_bad_dtype(bad_dtype):
    with pytest.raises(ValueError):
        ModelConfig(dtype=bad_dtype).validate()


def test_sae_config_rejects_unknown_type_and_negative_layer():
    with pytest.raises(ValueError):
        SAEConfig(type="magic").validate()
    with pytest.raises(ValueError):
        SAEConfig(layer=-1).validate()


def test_data_config_rejects_nonpositive_batch():
    with pytest.raises(ValueError):
        DataConfig(batch_size=0).validate()
    with pytest.raises(ValueError):
        DataConfig(save_every_batches=-2).validate()


def test_fc_config_tau_bounds():
    FeatureConditionedConfig(tau=0.0).validate()
    with pytest.raises(ValueError):
        FeatureConditionedConfig(tau=-0.1).validate()
    with pytest.raises(ValueError):
        FeatureConditionedConfig(silent_pct=150.0).validate()


def test_feature_cluster_config_bounds():
    with pytest.raises(ValueError):
        FeatureClusterConfig(top_pct=0.0).validate()
    with pytest.raises(ValueError):
        FeatureClusterConfig(resolution_parameter=0.0).validate()


def test_autolabel_config_positive_integers():
    AutoLabelConfig().validate()
    for field in ("pc_n_top", "max_prompt_chars", "max_examples", "max_new_tokens", "batch_size"):
        kwargs = {field: 0}
        with pytest.raises(ValueError):
            AutoLabelConfig(**kwargs).validate()


def test_pipeline_from_dict_roundtrip():
    cfg = PipelineConfig(name="unit", seed=7)
    cfg.data.max_samples = 123
    cfg.auto_label.batch_size = 3
    restored = PipelineConfig.from_dict(cfg.to_dict())
    assert restored.name == "unit" and restored.seed == 7
    assert restored.data.max_samples == 123
    assert restored.auto_label.batch_size == 3


def test_prompt_conditioned_config_validates_counts():
    PromptConditionedConfig().validate()
    with pytest.raises(ValueError):
        PromptConditionedConfig(n_svd=0).validate()
