"""Audit and diagnostics tool for SAE Feature Clusters (T_m) and Feature Retention.

Computes:
- Feature assignment coverage: Assigned (T_1..T_K) vs Dropped to Cluster 0.
- Cluster size distribution: min, max, mean, median, percentiles.
- Hypothesis eligibility: Clusters with >= 10 features (B.1 filter) vs < 10.
- Hypothesis coverage: Actual validated clusters with sign-consistent hypotheses.
- Top clusters breakdown with LLM labels (if available).

Usage:
    python -m experiments.audit_feature_clusters --config configs/qwen3_1.7b_batchtopk_65k.json
    python -m experiments.audit_feature_clusters --run_dir runs/qwen3_1.7b_batchtopk_65k
    python -m experiments.audit_feature_clusters --clusters_json checkpoints/.../clusters.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load {path}: {e}")
        return None


def resolve_paths(
    config_path: Optional[str] = None,
    run_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    clusters_json: Optional[str] = None,
) -> Dict[str, Optional[Path]]:
    """Resolve clusters.json, summary, hypotheses, and labels files."""
    paths: Dict[str, Optional[Path]] = {
        "clusters_json": None,
        "summary_json": None,
        "hypotheses_json": None,
        "labels_json": None,
        "matrices_mmap": None,
    }

    if clusters_json:
        paths["clusters_json"] = Path(clusters_json)

    if config_path:
        cfg = load_json(Path(config_path))
        if cfg:
            if not run_dir and cfg.get("output_dir"):
                run_dir = cfg["output_dir"]
            if not checkpoint_dir and cfg.get("checkpoint_dir"):
                checkpoint_dir = cfg["checkpoint_dir"]

    if run_dir:
        r_path = Path(run_dir)
        paths["summary_json"] = r_path / "pdd_summary.json"
        paths["hypotheses_json"] = r_path / "feature_conditioned_hypotheses.json"
        paths["labels_json"] = r_path / "feature_cluster_labels.json"

        # Check summary for checkpoint folder link
        summary = load_json(paths["summary_json"]) if paths["summary_json"] else None
        if summary and "checkpoint_subfolder" in summary:
            ckpt_sub = Path(summary["checkpoint_subfolder"])
            if not paths["clusters_json"] and (ckpt_sub / "clusters.json").exists():
                paths["clusters_json"] = ckpt_sub / "clusters.json"
            if (ckpt_sub / "matrices_mmap").is_dir():
                paths["matrices_mmap"] = ckpt_sub / "matrices_mmap"

    # Fallback search in checkpoints/ directory
    if not paths["clusters_json"]:
        search_dirs = [Path(checkpoint_dir)] if checkpoint_dir else [Path("checkpoints")]
        for sdir in search_dirs:
            if sdir.is_dir():
                candidates = sorted(sdir.glob("*/clusters.json"), key=os.path.getmtime, reverse=True)
                if candidates:
                    paths["clusters_json"] = candidates[0]
                    break

    return paths


def audit_feature_clusters(paths: Dict[str, Optional[Path]], d_sae_default: int = 16384) -> None:
    clusters_file = paths.get("clusters_json")
    if not clusters_file or not clusters_file.exists():
        print(f"[Error] Could not locate clusters.json. Checked: {clusters_file}")
        return

    print("=" * 70)
    print(" 🔍 PDD SAE FEATURE CLUSTERS AUDIT REPORT")
    print("=" * 70)
    print(f"File: {clusters_file}")

    data = load_json(clusters_file)
    if not data or "clusters" not in data:
        print("[Error] Malformed clusters.json (missing 'clusters' key).")
        return

    clusters: Dict[str, List[int]] = data["clusters"]
    feat_map: Dict[str, int] = data.get("feature_to_cluster", {})

    # Determine d_sae
    d_sae = d_sae_default
    if feat_map:
        d_sae = max(d_sae, max(int(k) for k in feat_map.keys()) + 1)

    summary = load_json(paths["summary_json"]) if paths.get("summary_json") else None
    if summary:
        cfg_sae = summary.get("config", {}).get("sae", {})
        if "d_sae" in cfg_sae:
            d_sae = int(cfg_sae["d_sae"])

    active_clusters = {int(k): v for k, v in clusters.items() if int(k) > 0}
    num_clusters = len(active_clusters)
    assigned_features = sum(len(v) for v in active_clusters.values())
    unassigned_features = d_sae - assigned_features
    coverage_pct = (assigned_features / d_sae) * 100.0
    dropped_pct = (unassigned_features / d_sae) * 100.0

    cluster_sizes = sorted([len(v) for v in active_clusters.values()])
    ge_10 = [s for s in cluster_sizes if s >= 10]
    ge_20 = [s for s in cluster_sizes if s >= 20]
    ge_50 = [s for s in cluster_sizes if s >= 50]
    lt_10 = [s for s in cluster_sizes if s < 10]

    print("\n📊 1. FEATURE ASSIGNMENT & COVERAGE")
    print("-" * 50)
    print(f"  • Total SAE Features (d_sae):     {d_sae:,}")
    print(f"  • Assigned Features (T_1..T_K):    {assigned_features:,}  ({coverage_pct:.2f}%)")
    print(f"  • Dropped into Cluster 0:          {unassigned_features:,}  ({dropped_pct:.2f}%)")

    print("\n📦 2. CLUSTER SIZE DISTRIBUTION (T_m)")
    print("-" * 50)
    print(f"  • Total Active Clusters (K_r):     {num_clusters}")
    if cluster_sizes:
        mean_size = sum(cluster_sizes) / len(cluster_sizes)
        median_size = cluster_sizes[len(cluster_sizes) // 2]
        p25 = cluster_sizes[len(cluster_sizes) // 4]
        p75 = cluster_sizes[(3 * len(cluster_sizes)) // 4]
        print(f"  • Cluster Sizes (Features):        Min={min(cluster_sizes)}, Max={max(cluster_sizes)}, Mean={mean_size:.1f}, Median={median_size}")
        print(f"  • Interquartile Range (25%-75%):   [{p25} ... {p75}] features")
        print(f"  • Clusters with >= 10 features:    {len(ge_10)} ({len(ge_10)/num_clusters*100:.1f}%)  [PASS B.1 Filter]")
        print(f"  • Clusters with >= 20 features:    {len(ge_20)} ({len(ge_20)/num_clusters*100:.1f}%)")
        print(f"  • Clusters with >= 50 features:    {len(ge_50)} ({len(ge_50)/num_clusters*100:.1f}%)")
        print(f"  • Clusters with < 10 features:     {len(lt_10)} ({len(lt_10)/num_clusters*100:.1f}%)  [DISCARDED at B.1]")

    # Check hypothesis coverage
    hypos_data = load_json(paths["hypotheses_json"]) if paths.get("hypotheses_json") else None
    if hypos_data:
        hypos_list = hypos_data if isinstance(hypos_data, list) else hypos_data.get("hypotheses", [])
        m_in_hypos = sorted(list({int(h.get("m")) for h in hypos_list if "m" in h}))
        print("\n🎯 3. HYPOTHESIS COVERAGE (B.1 Validated Universe)")
        print("-" * 50)
        print(f"  • Validated Clusters with Hypotheses: {len(m_in_hypos)} of {num_clusters} ({len(m_in_hypos)/num_clusters*100:.1f}%)")
        print(f"  • Total Generated Hypotheses:         {len(hypos_list):,}")
        print(f"  • Cluster ID List:                    {m_in_hypos[:15]}{' ...' if len(m_in_hypos) > 15 else ''}")

    # Top largest clusters + labels
    labels_data = load_json(paths["labels_json"]) if paths.get("labels_json") else None
    labels_dict = labels_data.get("feature_clusters", {}) if isinstance(labels_data, dict) else {}

    print("\n🏆 4. TOP 10 LARGEST FEATURE CLUSTERS")
    print("-" * 50)
    top_clusters = sorted(active_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for cid, feats in top_clusters:
        lbl_info = labels_dict.get(str(cid), {})
        title = lbl_info.get("title", "N/A") if isinstance(lbl_info, dict) else "N/A"
        has_hypo = "★" if (hypos_data and cid in m_in_hypos) else " "
        print(f"  {has_hypo} T_{cid:<4} | {len(feats):>4} features | {title}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit PDD SAE Feature Clusters and Feature Retention")
    parser.add_argument("--config", type=str, help="Path to pipeline config JSON (e.g. configs/qwen3_1.7b_batchtopk_65k.json)")
    parser.add_argument("--run_dir", type=str, help="Path to run directory (e.g. runs/qwen3_1.7b_batchtopk_65k)")
    parser.add_argument("--checkpoint_dir", type=str, help="Path to checkpoints directory")
    parser.add_argument("--clusters_json", type=str, help="Direct path to clusters.json file")
    parser.add_argument("--d_sae", type=int, default=16384, help="Default SAE feature dimension (default: 16384)")
    args = parser.parse_args()

    paths = resolve_paths(
        config_path=args.config,
        run_dir=args.run_dir,
        checkpoint_dir=args.checkpoint_dir,
        clusters_json=args.clusters_json,
    )
    audit_feature_clusters(paths, d_sae_default=args.d_sae)


if __name__ == "__main__":
    main()
