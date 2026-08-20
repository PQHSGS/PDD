"""Validation metrics module: Empirical vs. Predicted feature correlation (R^2).

Paper Sec. 4 & Sec. 5:
- Predicted feature signal: \Delta_{predicted} = \bar{u}^{in}_{k,m} - \bar{u}^{out}_{k,m}
- Empirical feature shift: \Delta_{empirical} = mean(a(pi_DPO)) - mean(a(pi_0))
- Computes Linear Regression R^2 and Pearson correlation r between predicted and empirical shifts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from typing import Any, Dict, List
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from .logger import get_logger

logger = get_logger("PDD.Validation")


@dataclass
class ValidationMetrics:
    r2_score: float
    pearson_r: float
    p_value: float
    num_pairs: int
    slope: float
    intercept: float

    def save_json(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def compute_prediction_validation_metrics(
    delta_predicted: np.ndarray,      # (M,) predicted chosen-minus-rejected feature signal
    delta_empirical: np.ndarray,      # (M,) empirical post-DPO minus pre-DPO feature activation shift
) -> ValidationMetrics:
    """Compute R^2 regression score and Pearson correlation between predicted and empirical shifts."""
    valid_mask = ~np.isnan(delta_predicted) & ~np.isnan(delta_empirical)
    x = delta_predicted[valid_mask].reshape(-1, 1)
    y = delta_empirical[valid_mask]

    if len(y) < 2:
        logger.warning("Insufficient data points for validation regression.")
        return ValidationMetrics(r2_score=0.0, pearson_r=0.0, p_value=1.0, num_pairs=len(y), slope=0.0, intercept=0.0)

    reg = LinearRegression()
    reg.fit(x, y)
    r2 = float(reg.score(x, y))

    r, p_val = pearsonr(x.flatten(), y)

    logger.info(f"Empirical Validation Results: R^2 = {r2:.4f} | Pearson r = {r:.4f} (p={p_val:.2e}, N={len(y)})")

    return ValidationMetrics(
        r2_score=r2,
        pearson_r=float(r),
        p_value=float(p_val),
        num_pairs=len(y),
        slope=float(reg.coef_[0]),
        intercept=float(reg.intercept_),
    )


def cluster_validation_metrics(
    feature_clusters: Dict[int, List[int]],
    p4_dir: os.PathLike,
) -> Dict[str, Any]:
    """Aggregate per-feature predicted vs observed post-DPO deltas into per-cluster metrics.

    Reads the clustering-independent per-feature arrays from ``p4_dir``:
    - ``u_feature.npy``: (d_sae,) per-feature predicted chosen-minus-rejected signal.
    - ``delta_all_epoch<N>.npy``: (d_sae,) per-feature empirical post-DPO minus pre-DPO shift.

    The epoch file is auto-detected (highest ``N`` present) so re-runs with a different
    epoch count work without configuration changes. Returns ``{"r2", "pearson_r", "clusters"}``
    where ``clusters[m]`` holds the mean predicted/observed delta over the members of T_m.
    """
    from pathlib import Path

    p4_dir = Path(p4_dir)
    u_file = p4_dir / "u_feature.npy"
    if not u_file.exists():
        return {"r2": 0.0, "pearson_r": 0.0, "clusters": {}}

    # Auto-detect the highest available epoch file (delta_all_epoch{N}.npy).
    epoch_files = sorted(
        p4_dir.glob("delta_all_epoch*.npy"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )
    if not epoch_files:
        logger.warning(f"No delta_all_epoch*.npy found under {p4_dir}; skipping cluster validation.")
        return {"r2": 0.0, "pearson_r": 0.0, "clusters": {}}
    d_file = epoch_files[-1]

    try:
        u_feat = np.load(u_file, mmap_mode="r")
        d_emp = np.load(d_file, mmap_mode="r")
        if len(u_feat) != len(d_emp):
            logger.warning(f"u_feature.npy ({len(u_feat)}) and {d_file.name} ({len(d_emp)}) length mismatch.")
            return {"r2": 0.0, "pearson_r": 0.0, "clusters": {}}

        out_clusters = {}
        u_list, d_list = [], []
        for m, feats in feature_clusters.items():
            valid_f = [int(f) for f in feats if int(f) < len(u_feat)]
            if len(valid_f) >= 4:
                u_m = float(np.mean(u_feat[valid_f]))
                d_m = float(np.mean(d_emp[valid_f]))
                out_clusters[int(m)] = {
                    "predicted_delta": u_m,
                    "observed_delta": d_m,
                    "n_features": len(valid_f),
                }
                u_list.append(u_m)
                d_list.append(d_m)

        r2 = 0.0
        r_val = 0.0
        if len(u_list) > 2:
            r_val = float(np.corrcoef(u_list, d_list)[0, 1])
            r2 = float(r_val ** 2) if not np.isnan(r_val) else 0.0

        logger.info(
            f"Cluster validation from {d_file.name}: R^2 = {r2:.4f} over {len(out_clusters)} clusters."
        )
        return {"r2": r2, "pearson_r": r_val, "clusters": out_clusters}
    except Exception as e:
        logger.warning(f"Error computing cluster validation metrics: {e}")
        return {"r2": 0.0, "pearson_r": 0.0, "clusters": {}}
