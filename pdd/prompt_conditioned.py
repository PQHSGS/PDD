"""Prompt-Conditioned Pipeline (Appendix B.2) OOP implementation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from .config import PromptConditionedConfig
from .feature_matrices import FeatureMatrices
from .logger import get_logger

logger = get_logger("PDD.PromptConditioned")


@dataclass
class PromptConditionedHypothesis:
    k: int                              # Prompt cluster index A_k
    m: int                              # Response cluster index R_m
    n_prompt_feats: int                 # Number of features in A_k
    n_resp_feats: int                   # Number of features in R_m
    u_in: float                         # Mean response delta inside top prompt group
    u_out: float                        # Mean response delta outside
    delta: float                        # Signed effect \Delta
    z_score: float                      # Welch z-score
    cohens_d: float                     # Cohen's d


@dataclass
class PromptConditionedResult:
    prompt_clusters: Dict[int, List[int]]
    resp_clusters: Dict[int, List[int]]
    c_matrix: np.ndarray
    u_matrix: np.ndarray
    hypotheses: List[PromptConditionedHypothesis]

    def save_summary(self, filepath: str) -> None:
        """Save prompt-conditioned hypotheses summary to disk as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "total_hypotheses": len(self.hypotheses),
            "hypotheses": [asdict(h) for h in self.hypotheses],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class PromptConditionedPipeline:
    """Runner for Appendix B.2 Prompt-Conditioned Hypothesis Generation Pipeline."""

    def __init__(self, cfg: PromptConditionedConfig):
        self.cfg = cfg

    def run(
        self,
        matrices: FeatureMatrices,
        use_max_statistic: bool = False,
        seed: int = 0,
        checkpoint_dir: Optional[str] = None,
    ) -> PromptConditionedResult:
        """Execute Prompt-Conditioned Pipeline.

        Memory-safe rewrite: never materializes the full D = C - R matrix nor the
        full (C>0)+(R>0) bool union. Per-column statistics are reduced directly
        from the (mmap'd) CSR matrices, D_sample is built from two row-slices, and
        c/u matrices come from sparse matmuls against indicator matrices.
        """
        import gc
        import scipy.sparse as sp
        from .feature_matrices import _to_csr

        use_max = use_max_statistic
        P = _to_csr(matrices.P_max if use_max else matrices.P_freq)
        Cd = _to_csr(matrices.C_max if use_max else matrices.C_freq)
        Rd = _to_csr(matrices.R_max if use_max else matrices.R_freq)
        Cf = _to_csr(matrices.C_freq)
        Rf = _to_csr(matrices.R_freq)
        N, d_sae = P.shape

        effective_min_p = min(self.cfg.min_prompt_count, max(2, int(0.01 * N)))
        effective_min_r = min(self.cfg.min_resp_count, max(2, int(0.01 * N)))
        logger.info(f"Filtering features for prompt-conditioned pipeline (effective min_prompt_count={effective_min_p}, min_resp_count={effective_min_r})...")

        if sp.issparse(P):
            p_counts = np.bincount(P.indices, minlength=d_sae).astype(np.int64)
        else:
            p_counts = np.sum(P > 0, axis=0)
        retained_p_indices = np.where(p_counts >= effective_min_p)[0]

        # r_counts = per-column count of examples where the feature fires in the
        # chosen OR rejected completion. Reuse the clustering pipeline's union
        # firing probabilities (p1 * N) when available at a matching extraction
        # state — avoids a ~6GB sparse elementwise multiply — else compute via
        # the overlap trick (Cf.multiply(Rf) is nonzero only where BOTH fire).
        from .feature_matrices import state_valid
        p1 = getattr(matrices, "_union_p1", None)
        if p1 is None and checkpoint_dir and state_valid(matrices, checkpoint_dir):
            p1_path = os.path.join(checkpoint_dir, "union_p1.npz")
            if os.path.exists(p1_path):
                p1 = np.load(p1_path)["p1"]
        if p1 is not None and len(p1) == d_sae:
            r_counts = (np.asarray(p1, dtype=np.float64) * N).astype(np.int64)
            del Cf, Rf
        else:
            overlap_f = Cf.multiply(Rf)
            r_counts = (
                np.bincount(Cf.indices, minlength=d_sae)
                + np.bincount(Rf.indices, minlength=d_sae)
                - np.bincount(overlap_f.indices, minlength=d_sae)
            ).astype(np.int64)
            del overlap_f, Cf, Rf
        gc.collect()

        # d_means/d_stds from column reductions of D = Cd - Rd (no materialization):
        #   mean_D = mean_C - mean_R
        #   E[D^2] = E[C^2] + E[R^2] - 2*E[C*R]
        colSum_C = np.asarray(Cd.sum(axis=0)).ravel()
        colSum_R = np.asarray(Rd.sum(axis=0)).ravel()
        d_means = (colSum_C - colSum_R) / float(N)

        colSum_C2 = np.bincount(Cd.indices, weights=Cd.data.astype(np.float64) ** 2, minlength=d_sae)
        colSum_R2 = np.bincount(Rd.indices, weights=Rd.data.astype(np.float64) ** 2, minlength=d_sae)
        CR = Cd.multiply(Rd)
        colSum_CR = np.asarray(CR.sum(axis=0)).ravel()
        del CR
        gc.collect()
        E_D2 = (colSum_C2 + colSum_R2 - 2.0 * colSum_CR) / float(N)
        d_stds = np.sqrt(np.maximum(0, E_D2 - d_means ** 2))

        retained_r_indices = np.where(
            (r_counts >= effective_min_r)
            & (d_stds >= self.cfg.min_resp_sigma)
            & (np.abs(d_means) >= self.cfg.min_resp_abs_delta)
        )[0]

        logger.info(f"Retained {len(retained_p_indices)} prompt features and {len(retained_r_indices)} response features.")


        if len(retained_p_indices) < 2 or len(retained_r_indices) < 2:
            logger.warning("Insufficient features retained for prompt-conditioned pipeline. Returning empty result.")
            return PromptConditionedResult(
                prompt_clusters={},
                resp_clusters={},
                c_matrix=np.zeros((N, 0), dtype=np.float32),
                u_matrix=np.zeros((N, 0), dtype=np.float32),
                hypotheses=[],
            )

        sample_size = min(self.cfg.n_sample_emb, N)
        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(N, size=sample_size, replace=False)

        # 1. Prompt Feature Embeddings (SVD-128)
        logger.info(f"Computing SVD-{self.cfg.n_svd} embeddings for prompt features...")
        P_sample = P[sample_idx][:, retained_p_indices].T
        if sp.issparse(P_sample):
            P_sample = P_sample.tocsr()
        n_p_components = min(self.cfg.n_svd, min(P_sample.shape) - 1)
        if n_p_components < self.cfg.n_svd:
            logger.info(f"Adjusted prompt SVD components to n_components={n_p_components} (from configured {self.cfg.n_svd}) for matrix shape {P_sample.shape}.")
        if n_p_components >= 2:
            svd_p = TruncatedSVD(n_components=n_p_components, random_state=seed)
            emb_p = svd_p.fit_transform(P_sample)
        else:
            emb_p = P_sample.toarray() if sp.issparse(P_sample) else P_sample
        del P_sample
        gc.collect()

        emb_p_norms = np.linalg.norm(emb_p, axis=1, keepdims=True)
        emb_p_norms[emb_p_norms == 0] = 1e-12
        emb_p_normed = emb_p / emb_p_norms

        k_p = min(self.cfg.n_prompt_clusters, len(retained_p_indices))
        kmeans_p = MiniBatchKMeans(n_clusters=k_p, random_state=seed, n_init="auto")
        labels_p = kmeans_p.fit_predict(emb_p_normed)

        prompt_clusters: Dict[int, List[int]] = {}
        for idx, cluster_label in enumerate(labels_p):
            feat_g = int(retained_p_indices[idx])
            prompt_clusters.setdefault(cluster_label + 1, []).append(feat_g)
        del emb_p, emb_p_normed
        gc.collect()

        # 2. Response Feature Embeddings (SVD-128)
        logger.info(f"Computing SVD-{self.cfg.n_svd} embeddings for response features...")
        # D[sample_idx][:, retained_r] built from two small CSR row-slices (no full D).
        Cd_s = Cd[sample_idx][:, retained_r_indices]
        Rd_s = Rd[sample_idx][:, retained_r_indices]
        D_sample = (Cd_s - Rd_s).T
        del Cd_s, Rd_s
        gc.collect()
        if sp.issparse(D_sample):
            D_sample = D_sample.tocsr()
        n_r_components = min(self.cfg.n_svd, min(D_sample.shape) - 1)
        if n_r_components >= 2:
            svd_r = TruncatedSVD(n_components=n_r_components, random_state=seed)
            emb_r = svd_r.fit_transform(D_sample)
        else:
            emb_r = D_sample.toarray() if sp.issparse(D_sample) else D_sample
        del D_sample
        gc.collect()

        emb_r_norms = np.linalg.norm(emb_r, axis=1, keepdims=True)
        emb_r_norms[emb_r_norms == 0] = 1e-12
        emb_r_normed = emb_r / emb_r_norms

        k_r = min(self.cfg.n_resp_clusters, len(retained_r_indices))
        kmeans_r = MiniBatchKMeans(n_clusters=k_r, random_state=seed, n_init="auto")
        labels_r = kmeans_r.fit_predict(emb_r_normed)

        resp_clusters: Dict[int, List[int]] = {}
        for idx, cluster_label in enumerate(labels_r):
            feat_g = int(retained_r_indices[idx])
            resp_clusters.setdefault(cluster_label + 1, []).append(feat_g)
        del emb_r, emb_r_normed
        gc.collect()

        # 3. Example Scores & Welch Tests
        K_p_final = len(prompt_clusters)
        K_r_final = len(resp_clusters)

        # c_matrix[:, k] = mean over A_k features of P per example = (P @ S_p) / |A_k|.
        # u_matrix[:, m] = mean over R_m features of D per example = (Cd - Rd) @ S_r / |R_m|.
        # Sparse matmuls against tiny indicator matrices avoid P.tocsc() / D.tocsc().
        p_keys = sorted(prompt_clusters.keys())
        S_p = sp.lil_matrix((d_sae, len(p_keys)), dtype=np.float32)
        for col_idx, pk in enumerate(p_keys):
            S_p[prompt_clusters[pk], col_idx] = 1.0
        S_p = S_p.tocsr()

        P_cluster_sums = P @ S_p
        if sp.issparse(P_cluster_sums):
            P_cluster_sums = P_cluster_sums.toarray()
        p_sizes = np.asarray([len(prompt_clusters[pk]) for pk in p_keys], dtype=np.float32)
        c_matrix = (np.asarray(P_cluster_sums, dtype=np.float64) / p_sizes[None, :]).astype(np.float32)
        del P_cluster_sums, S_p
        gc.collect()

        r_keys = sorted(resp_clusters.keys())
        S_r = sp.lil_matrix((d_sae, len(r_keys)), dtype=np.float32)
        for col_idx, rk in enumerate(r_keys):
            S_r[resp_clusters[rk], col_idx] = 1.0
        S_r = S_r.tocsr()

        C_cluster_sums = Cd @ S_r
        R_cluster_sums = Rd @ S_r
        if sp.issparse(C_cluster_sums):
            C_cluster_sums = C_cluster_sums.toarray()
        if sp.issparse(R_cluster_sums):
            R_cluster_sums = R_cluster_sums.toarray()
        r_sizes = np.asarray([len(resp_clusters[rk]) for rk in r_keys], dtype=np.float32)
        u_matrix = ((np.asarray(C_cluster_sums, dtype=np.float64) - np.asarray(R_cluster_sums, dtype=np.float64)) / r_sizes[None, :]).astype(np.float32)
        del C_cluster_sums, R_cluster_sums, S_r
        gc.collect()

        logger.info(f"Computing Welch inside-vs-outside tests across {K_p_final} prompt x {K_r_final} response clusters...")
        hypotheses: List[PromptConditionedHypothesis] = []
        n_top = min(200, N // 2)

        if n_top >= 5:
            for p_col_idx, pk in enumerate(tqdm(range(len(p_keys)), desc="Testing prompt-response pairs")):
                pk_val = p_keys[pk]
                c_scores = c_matrix[:, p_col_idx]
                top_in_indices = np.argsort(c_scores)[-n_top:]
                in_mask = np.zeros(N, dtype=bool)
                in_mask[top_in_indices] = True
                out_mask = ~in_mask

                n_in = n_top
                n_out = N - n_in

                u_in_all = u_matrix[in_mask].mean(axis=0)
                u_out_all = u_matrix[out_mask].mean(axis=0)
                delta_all = u_in_all - u_out_all

                var_in_all = u_matrix[in_mask].var(axis=0, ddof=1)
                var_out_all = u_matrix[out_mask].var(axis=0, ddof=1)

                se_all = np.sqrt((var_in_all / n_in) + (var_out_all / n_out) + 1e-12)
                z_score_all = delta_all / se_all

                s_pooled_sq_all = (((n_in - 1) * var_in_all) + ((n_out - 1) * var_out_all)) / max(1, (N - 2))
                s_pooled_all = np.sqrt(s_pooled_sq_all + 1e-12)
                cohens_d_all = delta_all / s_pooled_all

                for r_col_idx, rk_val in enumerate(r_keys):
                    hypotheses.append(
                        PromptConditionedHypothesis(
                            k=int(pk_val),
                            m=int(rk_val),
                            n_prompt_feats=len(prompt_clusters[pk_val]),
                            n_resp_feats=len(resp_clusters[rk_val]),
                            u_in=float(u_in_all[r_col_idx]),
                            u_out=float(u_out_all[r_col_idx]),
                            delta=float(delta_all[r_col_idx]),
                            z_score=float(z_score_all[r_col_idx]),
                            cohens_d=float(cohens_d_all[r_col_idx]),
                        )
                    )

        hypotheses.sort(key=lambda h: abs(h.z_score), reverse=True)
        logger.info(f"Extracted {len(hypotheses)} prompt-conditioned hypotheses.")

        return PromptConditionedResult(
            prompt_clusters=prompt_clusters,
            resp_clusters=resp_clusters,
            c_matrix=c_matrix,
            u_matrix=u_matrix,
            hypotheses=hypotheses,
        )
