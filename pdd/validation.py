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
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple

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
