"""Predictive Data Interventions module (Paper Sec. 5).

Provides 3 actionable debugging mechanisms:
1. Data Filtering / Inoculation: Purges preference pairs in response-topic cluster B_k causing negative shifts.
2. DPO Loss Reweighting / Masking: Calculates per-example loss weights w_i based on cluster disparity u_{i,m}.
3. SAE Feature Steering: Extracts decoder steering vectors v_steer for feature cluster T_m.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from .data import PreferenceExample
from .feature_clusters import FeatureClusterMap
from .feature_conditioned import FeatureConditionedResult, HypothesisPair
from .logger import get_logger

logger = get_logger("PDD.Interventions")


class DatasetInoculator:
    """Filter out dataset pairs that trigger unwanted behavior shifts."""

    @staticmethod
    def filter_examples_by_cluster(
        examples: List[PreferenceExample],
        cluster_assignments: np.ndarray,      # (N,) array of assignments in 0..K_data
        purge_cluster_ids: List[int],
    ) -> List[PreferenceExample]:
        """Purge preference examples belonging to target cluster IDs."""
        purge_set = set(purge_cluster_ids)
        retained: List[PreferenceExample] = []

        for idx, ex in enumerate(examples):
            if cluster_assignments[idx] not in purge_set:
                retained.append(ex)

        logger.info(f"Inoculated dataset: Purged {len(examples) - len(retained)} / {len(examples)} examples across clusters {purge_cluster_ids}.")
        return retained


class LossReweighter:
    """Compute per-example loss weights for DPO reward shaping."""

    @staticmethod
    def compute_disparity_weights(
        u_matrix: np.ndarray,                  # (N, K_r)
        target_cluster_idx: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Compute example weights w_i = sigmoid(u_{i, m} / temp)."""
        u_col = u_matrix[:, target_cluster_idx]
        weights = 1.0 / (1.0 + np.exp(- u_col / temperature))
        logger.info(f"Computed loss weights for target feature cluster {target_cluster_idx}: mean weight = {np.mean(weights):.4f}")
        return weights


class FeatureSteerer:
    """Extract SAE feature activation steering vectors from decoder weights."""

    @staticmethod
    def compute_steering_vector(
        sae: Any,
        feature_indices: List[int],
        weights_scale: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Compute steering direction vector v_steer = sum_g scale_g * W_dec[g]."""
        # sae.W_dec has shape (d_sae, d_in) in sae_lens convention
        w_dec = sae.W_dec.detach()
        d_in = w_dec.shape[1]

        v_steer = torch.zeros(d_in, device=w_dec.device, dtype=w_dec.dtype)
        scales = weights_scale if weights_scale is not None else [1.0] * len(feature_indices)

        for feat, scale in zip(feature_indices, scales):
            v_steer += scale * w_dec[feat]

        v_norm = torch.norm(v_steer)
        if v_norm > 0:
            v_steer = v_steer / v_norm

        logger.info(f"Computed unit steering vector for {len(feature_indices)} SAE features (norm={v_norm:.4f}).")
        return v_steer
