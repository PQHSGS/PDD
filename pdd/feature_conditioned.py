"""Feature-Conditioned Pipeline (Appendix B.1) OOP implementation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import gc
import json
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import List

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
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(self, filepath: str) -> None:
        """Persist intermediate matrices and hypotheses to fast npz checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        hypo_json = json.dumps([asdict(h) for h in self.hypotheses])
        tmp_path = filepath + ".tmp"
        np.savez(
            tmp_path,
            s_matrix=self.s_matrix,
            u_matrix=self.u_matrix,
            v_matrix=self.v_matrix,
            cluster_assignments=self.cluster_assignments,
            silent_mask=self.silent_mask,
            hypotheses_json=np.array(hypo_json),
        )
        os.replace(tmp_path, filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str) -> FeatureConditionedResult:
        """Load intermediate matrices and hypotheses via fast zero-copy memory-mapping."""
        with np.load(filepath, mmap_mode="r", allow_pickle=True) as data:
            hypo_data = json.loads(str(data["hypotheses_json"]))
            hypotheses = [HypothesisPair(**h) for h in hypo_data]
            return cls(
                s_matrix=np.array(data["s_matrix"], copy=False),
                u_matrix=np.array(data["u_matrix"], copy=False),
                v_matrix=np.array(data["v_matrix"], copy=False),
                cluster_assignments=np.array(data["cluster_assignments"], copy=False),
                silent_mask=np.array(data["silent_mask"], copy=False),
                hypotheses=hypotheses,
            )


class FeatureConditionedPipeline:
    """Runner for Appendix B.1 Feature-Conditioned Hypothesis Generation Pipeline."""

    def __init__(self, cfg: FeatureConditionedConfig):
        self.cfg = cfg

    def run(
        self,
        matrices: FeatureMatrices,
        cluster_map: FeatureClusterMap,
        seed: int = 0,
        checkpoint_dir: Optional[str] = None,
        use_checkpoint: bool = True,
    ) -> FeatureConditionedResult:
        """Execute Feature-Conditioned Pipeline."""
        if checkpoint_dir and use_checkpoint:
            ckpt_file = os.path.join(checkpoint_dir, "feature_conditioned.npz")
            if os.path.exists(ckpt_file):
                logger.info(f"Found cached FeatureConditioned result at '{ckpt_file}'. Skipping recomputation!")
                try:
                    return FeatureConditionedResult.load_checkpoint(ckpt_file)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint '{ckpt_file}' ({e}). Recomputing...")
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

        # Per-cluster primitives computed with bounded-memory sparse-dense matmuls
        # against an indicator matrix A[f, k] = 1 iff feature f is in cluster k.
        # s = C@A + R@A (sum of activations), v = C@A - R@A, and u counts activations
        # > tau chunk-wise (never materializing the full ~15GB C/R or a full bool
        # mask). Peak RAM stays at the (N, K) outputs + one row-chunk.
        c_csr = matrices.C_freq
        r_csr = matrices.R_freq
        d_sae = c_csr.shape[1]
        A = np.zeros((d_sae, K_r), dtype=np.float32)
        for k, cid in enumerate(cluster_ids):
            A[cluster_map.clusters[cid], k] = 1.0

        s_C = c_csr @ A
        s_R = r_csr @ A
        s_matrix = s_C + s_R
        v_matrix = s_C - s_R

        tau = self.cfg.tau
        u_matrix = np.zeros((N, K_r), dtype=np.float32)
        chunk = 8192
        for r0 in tqdm(range(0, N, chunk), desc="Streaming per-cluster activation counts (u)"):
            r1 = min(r0 + chunk, N)
            u_matrix[r0:r1] = (c_csr[r0:r1] > tau) @ A - (r_csr[r0:r1] > tau) @ A
        u_matrix /= np.asarray(cluster_sizes, dtype=np.float32)[None, :]
        del c_csr, r_csr, A, s_C, s_R
        gc.collect()

        # Silent bucket B_0
        s_norms = np.linalg.norm(s_matrix, axis=1)
        gamma = np.percentile(s_norms, self.cfg.silent_pct)
        silent_mask = s_norms < gamma
        active_indices = np.where(~silent_mask)[0]
        N_act = len(active_indices)
        logger.info(f"Silent bucket B_0 cutoff gamma={gamma:.4f} (silent examples: {np.sum(silent_mask)} / {N}).")

        cluster_assignments = np.zeros(N, dtype=np.int32)
        if N_act > 0 and self.cfg.n_data_clusters > 0:
            # Strictly honor the configured number of data clusters. Only a hard
            # feasibility clamp (cannot cluster more groups than active samples)
            # applies; otherwise the user's config decides.
            n_clusters = min(self.cfg.n_data_clusters, N_act)
            if n_clusters < self.cfg.n_data_clusters:
                logger.info(
                    f"Clamped data clusters K={n_clusters} (configured {self.cfg.n_data_clusters} "
                    f"exceeds active sample count N_act={N_act}); no auto-tuning applied."
                )
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
        # Strict paper filter: only hypotheses on data clusters with n_k >= the
        # configured minimum survive. No auto-cap relaxes this for small datasets.
        effective_min_cluster_size = self.cfg.min_data_cluster_size
        logger.info(
            f"Using strict data cluster min size n_k >= {effective_min_cluster_size} "
            f"(paper filter; no auto-cap) across {num_active_clusters} active data clusters."
        )

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

        result = FeatureConditionedResult(
            s_matrix=s_matrix,
            u_matrix=u_matrix,
            v_matrix=v_matrix,
            cluster_assignments=cluster_assignments,
            silent_mask=silent_mask,
            hypotheses=hypotheses,
        )

        if checkpoint_dir and use_checkpoint:
            ckpt_file = os.path.join(checkpoint_dir, "feature_conditioned.npz")
            try:
                result.save_checkpoint(ckpt_file)
                logger.info(f"Saved FeatureConditioned checkpoint to '{ckpt_file}'.")
            except Exception as e:
                logger.warning(f"Failed to save FeatureConditioned checkpoint ({e}).")

        return result
