"""Leiden Feature Clusterer with disk checkpointing (.json)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
import igraph as ig
import leidenalg as la
import numpy as np
from typing import Dict, List, Optional, Tuple

from .logger import get_logger

logger = get_logger("PDD.FeatureClusterer")


@dataclass
class FeatureClusterMap:
    """Mapping of feature clusters T_m and feature assignments."""

    clusters: Dict[int, List[int]]         # m (1..K_r) -> list of SAE feature indices T_m
    feature_to_cluster: Dict[int, int]    # feature index g -> cluster_id m (0 if unassigned)

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)

    def save_json(self, filepath: str) -> None:
        """Save cluster mapping to disk as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "clusters": {str(k): v for k, v in self.clusters.items()},
            "feature_to_cluster": {str(k): v for k, v in self.feature_to_cluster.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> FeatureClusterMap:
        """Load cluster mapping from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        clusters = {int(k): v for k, v in data["clusters"].items()}
        feature_to_cluster = {int(k): v for k, v in data["feature_to_cluster"].items()}
        return cls(clusters=clusters, feature_to_cluster=feature_to_cluster)


class LeidenFeatureClusterer:
    """Normalized binary mutual-information graph builder & Leiden community detector."""

    def __init__(
        self,
        top_pct: float = 1.0,
        min_community_size: int = 4,
        min_firing_freq: float = 1e-5,
    ):
        self.top_pct = top_pct
        self.min_community_size = min_community_size
        self.min_firing_freq = min_firing_freq

    def cluster(
        self,
        binary_activations: np.ndarray,       # (N, d_sae)
        seed: int = 0,
        checkpoint_path: Optional[str] = None,
        use_checkpoint: bool = True,
    ) -> FeatureClusterMap:
        """Cluster SAE features using binary MI graph and Leiden algorithm."""
        if use_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading cached feature clusters from checkpoint: {checkpoint_path}")
            return FeatureClusterMap.load_json(checkpoint_path)


        logger.info(f"Building normalized binary MI graph for {binary_activations.shape[1]} SAE features...")
        cluster_map = self._build_clusters(binary_activations, seed)

        if checkpoint_path:
            logger.info(f"Saving feature cluster mapping to checkpoint: {checkpoint_path}")
            cluster_map.save_json(checkpoint_path)

        return cluster_map

    def _build_clusters(self, binary_activations: np.ndarray, seed: int) -> FeatureClusterMap:
        N, d_sae = binary_activations.shape
        p1 = binary_activations.mean(axis=0)
        p0 = 1.0 - p1

        active_indices = np.where(p1 > self.min_firing_freq)[0]
        D_act = len(active_indices)

        if D_act < self.min_community_size:
            logger.warning(f"Only {D_act} active features found. Returning empty cluster map.")
            return FeatureClusterMap(clusters={}, feature_to_cluster={g: 0 for g in range(d_sae)})

        act_matrix = binary_activations[:, active_indices]
        p1_act = p1[active_indices]
        p0_act = p0[active_indices]

        H = - np.where(p0_act > 0, p0_act * np.log(p0_act + 1e-12), 0.0) \
            - np.where(p1_act > 0, p1_act * np.log(p1_act + 1e-12), 0.0)

        p11 = (act_matrix.T @ act_matrix) / N
        p10 = p1_act[:, None] - p11
        p01 = p1_act[None, :] - p11
        p00 = 1.0 - p11 - p10 - p01

        p11 = np.clip(p11, 1e-12, 1.0)
        p10 = np.clip(p10, 1e-12, 1.0)
        p01 = np.clip(p01, 1e-12, 1.0)
        p00 = np.clip(p00, 1e-12, 1.0)

        MI = p11 * np.log(p11 / (p1_act[:, None] * p1_act[None, :] + 1e-12)) \
           + p10 * np.log(p10 / (p1_act[:, None] * p0_act[None, :] + 1e-12)) \
           + p01 * np.log(p01 / (p0_act[None, :] * p1_act[None, :] + 1e-12)) \
           + p00 * np.log(p00 / (p0_act[None, :] * p0_act[None, :] + 1e-12))

        denom = np.sqrt(np.outer(H, H)) + 1e-12
        norm_MI = MI / denom
        np.fill_diagonal(norm_MI, 0.0)

        triu_i, triu_j = np.triu_indices_from(norm_MI, k=1)
        weights = norm_MI[triu_i, triu_j]

        non_zero_mask = weights > 0
        if not np.any(non_zero_mask):
            return FeatureClusterMap(clusters={}, feature_to_cluster={g: 0 for g in range(d_sae)})

        triu_i = triu_i[non_zero_mask]
        triu_j = triu_j[non_zero_mask]
        weights = weights[non_zero_mask]

        cutoff = np.percentile(weights, 100.0 - self.top_pct)
        top_mask = weights >= cutoff

        # Map back to global feature indices
        global_i = active_indices[triu_i[top_mask]]
        global_j = active_indices[triu_j[top_mask]]

        edges = list(zip(global_i.tolist(), global_j.tolist()))
        edge_weights = weights[top_mask].tolist()

        g = ig.Graph(n=d_sae, edges=edges, edge_attrs={"weight": edge_weights})

        logger.info(f"Running Leiden community detection on graph with {len(edges)} edges...")
        partition = la.find_partition(
            g,
            la.ModularityVertexPartition,
            weights="weight",
            seed=seed,
        )

        clusters: Dict[int, List[int]] = {}
        feature_to_cluster: Dict[int, int] = {feat: 0 for feat in range(d_sae)}

        cluster_id_counter = 1
        for comm in partition:
            if len(comm) >= self.min_community_size:
                clusters[cluster_id_counter] = list(comm)
                for feat in comm:
                    feature_to_cluster[feat] = cluster_id_counter
                cluster_id_counter += 1

        logger.info(f"Extracted {len(clusters)} retained Leiden feature communities.")
        return FeatureClusterMap(clusters=clusters, feature_to_cluster=feature_to_cluster)
