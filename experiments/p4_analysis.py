"""Phase P4 offline rigor analysis (CPU-only, seconds). Reads runs/<cfg>/p4_validation/ artifacts.

Reports, per epoch:
- Positive control: reward margin pre-vs-post + mean|Δ| empirical shift (was the model moved at all).
- Measurement noise floor: split-half SE of the pre-DPO per-prompt firing means (is delta measurable
  above sampling noise?).
- Headline R^2 / Pearson / Spearman over top-k clusters and all retained clusters.
- Feature-level Spearman over all SAE features (large-N robustness).
- Permutation p-value and bootstrap CI on the top-k R^2 (is the correlation significant?).
- Negative control: R^2 on random cluster subsets (must be ~0; top-k must beat it).
- Direction check: sign agreement between predicted and empirical on top clusters.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from pdd.config import PipelineConfig
from pdd.feature_clusters import FeatureClusterMap
from pdd.logger import get_logger

logger = get_logger("PDD.Exp.P4.Analysis")


def lin_r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.all(x == x[0]):
        return 0.0
    return float(LinearRegression().fit(x.reshape(-1, 1), y).score(x.reshape(-1, 1), y))


def stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return {"n": int(len(x)), "r2": 0.0, "pearson": 0.0, "p_pearson": 1.0, "spearman": 0.0, "p_spearman": 1.0}
    r, p = pearsonr(x, y)
    rho, rho_p = spearmanr(x, y)
    return {"n": int(len(x)), "r2": lin_r2(x, y), "pearson": float(r), "p_pearson": float(p),
            "spearman": float(rho), "p_spearman": float(rho_p)}


def cluster_mean_of(arr: np.ndarray, feats: list[int]) -> float:
    if not feats:
        return float("nan")
    return float(arr[feats].mean())


def split_half_se(per_prompt: np.ndarray, cluster_ids: list[int], cluster_map) -> np.ndarray:
    """Per-cluster SE estimate from two prompt halves of the pre-DPO measurement."""
    h1 = per_prompt[0::2]
    h2 = per_prompt[1::2]
    se = np.zeros(len(cluster_ids), dtype=np.float64)
    for k, cid in enumerate(cluster_ids):
        feats = cluster_map.clusters[cid]
        if feats:
            se[k] = abs(float(h1[:, feats].mean()) - float(h2[:, feats].mean())) / 2.0
    return se


def analyze_epoch(epoch: int, dirpath: str, u_bar: np.ndarray, delta_full: np.ndarray,
                  delta_all: np.ndarray, u_feature: np.ndarray, cluster_ids: list[int],
                  cluster_map: FeatureClusterMap, ks: list[int],
                  n_perm: int, n_boot: int, n_neg: int, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    valid = ~np.isnan(delta_full)
    rank = np.argsort(-np.abs(u_bar))
    res: dict = {"epoch": epoch}

    # Headline top-k / all-cluster metrics
    res["top_k"] = {}
    for k in ks:
        idx = [i for i in rank[:k] if valid[i]]
        if len(idx) >= 2:
            res["top_k"][str(k)] = stats(u_bar[idx], delta_full[idx])
    idx = [i for i in range(len(u_bar)) if valid[i]]
    res["all_clusters"] = stats(u_bar[idx], delta_full[idx])

    # Feature-level Spearman (large-N)
    m = ~np.isnan(delta_all)
    if m.sum() >= 2:
        rho, rho_p = spearmanr(u_feature[m], delta_all[m])
        res["feature_level"] = {"n": int(m.sum()), "spearman": float(rho), "p": float(rho_p)}

    # Permutation + bootstrap on top-k
    k = min(50, int(valid.sum()))
    idx = [i for i in rank[:k] if valid[i]]
    x, y = u_bar[idx], delta_full[idx]
    if len(x) >= 5:
        r_obs, _ = pearsonr(x, y)
        n_ge = sum(abs(pearsonr(x, y[rng.permutation(len(y))])[0]) >= abs(r_obs) for _ in range(n_perm))
        res["permutation"] = {"k": k, "p": (n_ge + 1) / (n_perm + 1), "pearson_obs": float(r_obs)}
        boot = np.array([lin_r2(x[rng.randint(0, len(x), size=len(x))], y[rng.randint(0, len(y), size=len(y))]) for _ in range(n_boot)])
        res["bootstrap_ci"] = {"k": k, "r2_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "r2_obs": lin_r2(x, y)}

        # Negative control: random cluster subsets
        pool = np.where(valid)[0]
        neg = np.array([lin_r2(u_bar[rng.choice(pool, size=k, replace=False)], delta_full[rng.choice(pool, size=k, replace=False)]) for _ in range(n_neg)])
        res["negative_control"] = {"k": k, "random_r2_mean": float(neg.mean()), "random_r2_p95": float(np.percentile(neg, 95)),
                                   "topk_r2": lin_r2(x, y)}

        # Direction check: sign agreement on top clusters
        agree = np.mean(np.sign(x) == np.sign(y))
        res["direction_agreement"] = {"k": k, "frac_sign_agree": float(agree)}

    # Measurement noise floor for this epoch's delta (uses pre per-prompt + cluster structure)
    pre_pp = np.load(os.path.join(dirpath, "per_prompt_pre.npy"))
    se = split_half_se(pre_pp, cluster_ids, cluster_map)
    med_se = float(np.nanmedian(se))
    mean_abs_delta = float(np.nanmean(np.abs(delta_full)))
    res["noise_floor"] = {"median_cluster_se": med_se, "mean_abs_delta": mean_abs_delta, "snr": mean_abs_delta / med_se if med_se > 0 else float("inf")}
    return res


def main():
    parser = argparse.ArgumentParser(description="Phase P4 offline rigor analysis")
    parser.add_argument("--config", type=str, default="configs/qwen3_1.7b_base.json")
    parser.add_argument("--ks", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--n_perm", type=int, default=1000)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--n_neg", type=int, default=200)
    args = parser.parse_args()

    cfg = PipelineConfig.load_json(args.config)
    run_dir = cfg.output_dir
    out_dir = os.path.join(run_dir, "p4_validation")

    with open(os.path.join(run_dir, "pdd_summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    subfolder = summary.get("checkpoint_subfolder")
    with open(os.path.join(subfolder, "clusters.json"), "r", encoding="utf-8") as f:
        clusters_data = json.load(f)
    feat_to_cluster = {int(k): int(v) for k, v in clusters_data.get("feature_to_cluster", {}).items()}
    retained = {int(k): [int(x) for x in v] for k, v in clusters_data.get("clusters", {}).items()}
    cluster_map = FeatureClusterMap(clusters=retained, feature_to_cluster=feat_to_cluster)

    with open(os.path.join(out_dir, "cluster_ids.json"), "r", encoding="utf-8") as f:
        cluster_ids = [int(c) for c in json.load(f)]
    u_bar = np.load(os.path.join(out_dir, "u_bar_global.npy"))
    u_feature = np.load(os.path.join(out_dir, "u_feature.npy"))

    epoch_files = sorted(int(fn.split("epoch")[1].split(".")[0]) for fn in os.listdir(out_dir) if fn.startswith("delta_emp_full_epoch"))
    report = {"ks": args.ks}
    for epoch in epoch_files:
        delta_full = np.load(os.path.join(out_dir, f"delta_emp_full_epoch{epoch}.npy"))
        delta_all = np.load(os.path.join(out_dir, f"delta_all_epoch{epoch}.npy"))
        report[f"epoch_{epoch}"] = analyze_epoch(
            epoch, out_dir, u_bar, delta_full, delta_all, u_feature,
            cluster_ids, cluster_map,
            args.ks, args.n_perm, args.n_boot, args.n_neg, seed=cfg.seed,
        )

    with open(os.path.join(out_dir, "p4_analysis_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for ep, data in report.items():
        if not ep.startswith("epoch_"):
            continue
        tk = data["top_k"].get("50", {})
        fl = data.get("feature_level", {})
        perm = data.get("permutation", {})
        nf = data.get("noise_floor", {})
        print(f"[epoch {data['epoch']}] top50 R2={tk.get('r2', float('nan')):.3f} r={tk.get('pearson', float('nan')):.3f} "
              f"| feature rho={fl.get('spearman', float('nan')):.3f} | perm p={perm.get('p', float('nan')):.3f} "
              f"| SNR={nf.get('snr', float('nan')):.2f} | rand R2={data.get('negative_control', {}).get('random_r2_mean', float('nan')):.3f}")
    logger.info(f"Analysis report saved to '{os.path.join(out_dir, 'p4_analysis_report.json')}'")


if __name__ == "__main__":
    main()