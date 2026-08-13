"""Feature-Conditioned Pipeline (Appendix B.1) OOP implementation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from .config import FeatureConditionedConfig
from .feature_clusters import FeatureClusterMap
from .feature_matrices import FeatureMatrices
from .logger import get_logger

logger = get_logger("PDD.FeatureConditioned")


@dataclass
class HypothesisPair:
    k: int                              # Data cluster index (1..K_data)
    m: int                              # Response feature cluster index (1..K_r)
    n_k: int                            # Data cluster size
    t_m: int                            # Feature cluster size |T_m|
    u_in: float                         # Inside mean
    u_out: float                        # Outside mean
    delta: float                        # Signed effect \Delta_{k,m}
    z_score: float                      # Welch z_{k,m}
    cohens_d: float                     # Cohen's d_{k,m}
    delta_A: float                      # Split-half A effect
    delta_B: float                      # Split-half B effect
    sign_consistent: bool               # SC_{k,m} flag
    delta_min: float                    # Conservative effect score \Delta^{min}_{k,m}
    is_chosen_leaning: bool             # True if \Delta_{k,m} > 0, False if < 0


@dataclass
class FeatureConditionedResult:
    s_matrix: np.ndarray                # (N, K_r)
    u_matrix: np.ndarray                # (N, K_r)
    v_matrix: np.ndarray                # (N, K_r)
    cluster_assignments: np.ndarray     # (N,) in 0..K_data
    silent_mask: np.ndarray             # (N,) bool
    hypotheses: List[HypothesisPair]     # Filtered & ranked hypothesis pairs

    def save_summary(self, filepath: str) -> None:
        """Save hypotheses summary to disk as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "total_hypotheses": len(self.hypotheses),
            "hypotheses": [asdict(h) for h in self.hypotheses],
        }
        tmp_filepath = filepath + ".tmp"
        with open(tmp_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_filepath, filepath)


class FeatureConditionedPipeline:
    """Runner for Appendix B.1 Feature-Conditioned Hypothesis Generation Pipeline."""

    def __init__(self, cfg: FeatureConditionedConfig):
        self.cfg = cfg

    def run(
        self,
        matrices: FeatureMatrices,
        cluster_map: FeatureClusterMap,
        seed: int = 0,
    ) -> FeatureConditionedResult:
        """Execute Feature-Conditioned Pipeline."""
        N = len(matrices.example_ids)
        K_r = cluster_map.num_clusters

        if K_r == 0:
            logger.warning("No feature clusters provided. Returning empty FeatureConditionedResult.")
            return FeatureConditionedResult(
                s_matrix=np.zeros((N, 0), dtype=np.float32),
                u_matrix=np.zeros((N, 0), dtype=np.float32),
                v_matrix=np.zeros((N, 0), dtype=np.float32),
                cluster_assignments=np.zeros(N, dtype=np.int32),
                silent_mask=np.ones(N, dtype=bool),
                hypotheses=[],
            )

        logger.info(f"Computing per-pair primitives (s, u, v) over {K_r} feature clusters...")
        s_matrix = np.zeros((N, K_r), dtype=np.float32)
        u_matrix = np.zeros((N, K_r), dtype=np.float32)
        v_matrix = np.zeros((N, K_r), dtype=np.float32)

        cluster_ids = sorted(cluster_map.clusters.keys())
        cluster_sizes = [len(cluster_map.clusters[cid]) for cid in cluster_ids]

        for col_idx, cid in enumerate(cluster_ids):
            feats = cluster_map.clusters[cid]
            c_freq = matrices.C_freq[:, feats]
            r_freq = matrices.R_freq[:, feats]
            if sp.issparse(c_freq):
                c_freq = c_freq.toarray()
            if sp.issparse(r_freq):
                r_freq = r_freq.toarray()

            s_matrix[:, col_idx] = np.sum(c_freq + r_freq, axis=1)
            b_tau = (c_freq > self.cfg.tau).astype(np.float32) - (r_freq > self.cfg.tau).astype(np.float32)
            u_matrix[:, col_idx] = np.mean(b_tau, axis=1)
            v_matrix[:, col_idx] = np.sum(c_freq - r_freq, axis=1)

        # Silent bucket B_0
        s_norms = np.linalg.norm(s_matrix, axis=1)
        gamma = np.percentile(s_norms, self.cfg.silent_pct)
        silent_mask = s_norms < gamma
        active_indices = np.where(~silent_mask)[0]
        N_act = len(active_indices)
        logger.info(f"Silent bucket B_0 cutoff gamma={gamma:.4f} (silent examples: {np.sum(silent_mask)} / {N}).")

        cluster_assignments = np.zeros(N, dtype=np.int32)
        if N_act > 0 and self.cfg.n_data_clusters > 0:
            n_clusters = min(self.cfg.n_data_clusters, max(2, N_act // 10))
            if n_clusters < self.cfg.n_data_clusters:
                logger.info(f"Adjusted data clusters K={n_clusters} (from configured {self.cfg.n_data_clusters}) for active sample size N_act={N_act}.")
            logger.info(f"Running Spherical K-Means (K={n_clusters}) on {N_act} active examples...")
            s_act = s_matrix[active_indices]
            s_act_norms = np.linalg.norm(s_act, axis=1, keepdims=True)
            s_act_norms[s_act_norms == 0] = 1e-12
            s_act_normed = s_act / s_act_norms

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(1024, N_act),
                random_state=seed,
                n_init="auto",
            )
            kmeans.fit(s_act_normed)
            cluster_assignments[active_indices] = kmeans.labels_ + 1

        logger.info("Computing Welch inside-vs-outside statistics and split-half validation...")
        hypotheses: List[HypothesisPair] = []
        N_pool = N_act

        if N_pool < 2:
            return FeatureConditionedResult(
                s_matrix=s_matrix,
                u_matrix=u_matrix,
                v_matrix=v_matrix,
                cluster_assignments=cluster_assignments,
                silent_mask=silent_mask,
                hypotheses=[],
            )

        half_A_mask = (np.arange(N) % 2 == 0) & (~silent_mask)
        half_B_mask = (np.arange(N) % 2 == 1) & (~silent_mask)

        unique_clusters = set(cluster_assignments)
        unique_clusters.discard(0)
        num_active_clusters = max(1, len(unique_clusters))
        effective_min_cluster_size = min(self.cfg.min_data_cluster_size, max(2, N_act // (num_active_clusters * 4)))

        for k in tqdm(sorted(unique_clusters), desc="Testing feature-data pairs"):
            in_mask = (cluster_assignments == k)
            n_k = np.sum(in_mask)

            out_mask = (~silent_mask) & (~in_mask)
            n_out = N_pool - n_k

            if n_k < 2 or n_out < 2:
                continue

            in_A_mask = in_mask & half_A_mask
            out_A_mask = out_mask & half_A_mask
            n_in_A = np.sum(in_A_mask)
            n_out_A = np.sum(out_A_mask)

            in_B_mask = in_mask & half_B_mask
            out_B_mask = out_mask & half_B_mask
            n_in_B = np.sum(in_B_mask)
            n_out_B = np.sum(out_B_mask)

            u_in_all = u_matrix[in_mask].mean(axis=0)
            u_out_all = u_matrix[out_mask].mean(axis=0)
            delta_all = u_in_all - u_out_all

            var_in_all = u_matrix[in_mask].var(axis=0, ddof=1)
            var_out_all = u_matrix[out_mask].var(axis=0, ddof=1)

            se_all = np.sqrt((var_in_all / n_k) + (var_out_all / n_out) + 1e-12)
            z_score_all = delta_all / se_all

            s_pooled_sq_all = (((n_k - 1) * var_in_all) + ((n_out - 1) * var_out_all)) / max(1, (N_pool - 2))
            cohens_d_all = delta_all / np.sqrt(s_pooled_sq_all + 1e-12)

            delta_A_all = np.zeros(K_r, dtype=np.float32)
            if n_in_A > 0 and n_out_A > 0:
                delta_A_all = u_matrix[in_A_mask].mean(axis=0) - u_matrix[out_A_mask].mean(axis=0)

            delta_B_all = np.zeros(K_r, dtype=np.float32)
            if n_in_B > 0 and n_out_B > 0:
                delta_B_all = u_matrix[in_B_mask].mean(axis=0) - u_matrix[out_B_mask].mean(axis=0)

            sc_all = (
                (np.abs(delta_A_all) > self.cfg.split_half_eps)
                & (np.abs(delta_B_all) > self.cfg.split_half_eps)
                & (np.sign(delta_A_all) == np.sign(delta_B_all))
            )
            delta_min_all = np.minimum(np.abs(delta_A_all), np.abs(delta_B_all))

            for col_idx, cid in enumerate(cluster_ids):
                t_m = cluster_sizes[col_idx]
                sc = sc_all[col_idx]
                if t_m >= self.cfg.min_feat_cluster_size and n_k >= effective_min_cluster_size and sc:
                    hypotheses.append(
                        HypothesisPair(
                            k=int(k),
                            m=int(cid),
                            n_k=int(n_k),
                            t_m=int(t_m),
                            u_in=float(u_in_all[col_idx]),
                            u_out=float(u_out_all[col_idx]),
                            delta=float(delta_all[col_idx]),
                            z_score=float(z_score_all[col_idx]),
                            cohens_d=float(cohens_d_all[col_idx]),
                            delta_A=float(delta_A_all[col_idx]),
                            delta_B=float(delta_B_all[col_idx]),
                            sign_consistent=bool(sc),
                            delta_min=float(delta_min_all[col_idx]),
                            is_chosen_leaning=bool(delta_all[col_idx] > 0),
                        )
                    )

        hypotheses.sort(key=lambda h: h.delta_min, reverse=True)
        logger.info(f"Extracted {len(hypotheses)} verified hypotheses passing split-half validation.")

        return FeatureConditionedResult(
            s_matrix=s_matrix,
            u_matrix=u_matrix,
            v_matrix=v_matrix,
            cluster_assignments=cluster_assignments,
            silent_mask=silent_mask,
            hypotheses=hypotheses,
        )
