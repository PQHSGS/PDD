"""Prompt-Conditioned Pipeline (Appendix B.2) OOP implementation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from typing import Dict, List, Tuple

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
        tmp_filepath = filepath + ".tmp"
        with open(tmp_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_filepath, filepath)


class PromptConditionedPipeline:
    """Runner for Appendix B.2 Prompt-Conditioned Hypothesis Generation Pipeline."""

    def __init__(self, cfg: PromptConditionedConfig):
        self.cfg = cfg

    def run(
        self,
        matrices: FeatureMatrices,
        use_max_statistic: bool = False,
        seed: int = 0,
    ) -> PromptConditionedResult:
        """Execute Prompt-Conditioned Pipeline."""
        P = matrices.P_max if use_max_statistic else matrices.P_freq
        D = matrices.D_max if use_max_statistic else matrices.D_freq
        N, d_sae = P.shape

        effective_min_p = min(self.cfg.min_prompt_count, max(2, int(0.01 * N)))
        effective_min_r = min(self.cfg.min_resp_count, max(2, int(0.01 * N)))
        logger.info(f"Filtering features for prompt-conditioned pipeline (effective min_prompt_count={effective_min_p}, min_resp_count={effective_min_r})...")

        import scipy.sparse as sp

        if sp.issparse(P):
            p_counts = np.asarray((P > 0).sum(axis=0)).flatten()
        else:
            p_counts = np.sum(P > 0, axis=0)
        retained_p_indices = np.where(p_counts >= effective_min_p)[0]

        if sp.issparse(matrices.C_freq):
            r_counts = np.asarray(((matrices.C_freq > 0) + (matrices.R_freq > 0) > 0).sum(axis=0)).flatten()
        else:
            r_counts = np.sum((matrices.C_freq > 0) | (matrices.R_freq > 0), axis=0)

        if sp.issparse(D):
            d_means = np.asarray(D.mean(axis=0)).flatten()
            D_sq = D.copy()
            D_sq.data = D_sq.data ** 2
            d_stds = np.sqrt(np.maximum(0, np.asarray(D_sq.mean(axis=0)).flatten() - d_means ** 2))
        else:
            d_means = np.mean(D, axis=0)
            d_stds = np.std(D, axis=0)

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

        # 2. Response Feature Embeddings (SVD-128)
        logger.info(f"Computing SVD-{self.cfg.n_svd} embeddings for response features...")
        D_sample = D[sample_idx][:, retained_r_indices].T
        if sp.issparse(D_sample):
            D_sample = D_sample.tocsr()
        n_r_components = min(self.cfg.n_svd, min(D_sample.shape) - 1)
        if n_r_components >= 2:
            svd_r = TruncatedSVD(n_components=n_r_components, random_state=seed)
            emb_r = svd_r.fit_transform(D_sample)
        else:
            emb_r = D_sample.toarray() if sp.issparse(D_sample) else D_sample

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

        # 3. Example Scores & Welch Tests
        K_p_final = len(prompt_clusters)
        K_r_final = len(resp_clusters)

        c_matrix = np.zeros((N, K_p_final), dtype=np.float32)
        u_matrix = np.zeros((N, K_r_final), dtype=np.float32)

        p_keys = sorted(prompt_clusters.keys())
        for col_idx, pk in enumerate(p_keys):
            feats = prompt_clusters[pk]
            p_sub = P[:, feats]
            if sp.issparse(p_sub):
                c_matrix[:, col_idx] = np.asarray(p_sub.mean(axis=1)).flatten()
            else:
                c_matrix[:, col_idx] = np.mean(p_sub, axis=1)

        r_keys = sorted(resp_clusters.keys())
        for col_idx, rk in enumerate(r_keys):
            feats = resp_clusters[rk]
            d_sub = D[:, feats]
            if sp.issparse(d_sub):
                u_matrix[:, col_idx] = np.asarray(d_sub.mean(axis=1)).flatten()
            else:
                u_matrix[:, col_idx] = np.mean(d_sub, axis=1)

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
