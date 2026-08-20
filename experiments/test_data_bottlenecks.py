#!/usr/bin/env python3
"""Exhaustive Data Bottleneck & Behavioral Interference Discovery Tool.

Performs vectorized matrix search across all K x K feature cluster combinations
(17,000+ pairs) over the 260k preference dataset in seconds to uncover severe
data deficits across all behavioral interference regimes:

Interference Modes:
1. 'all': Run exhaustive discovery across all regimes sequentially.
2. 'amp_sup' (Default): Decoupling Deficit (+A, -B) -> Amplify A while Suppressing B.
3. 'amp_amp': Co-Amplification Deficit (+A, +B) -> Multi-Skill Synthesis (Amplify A AND B).
4. 'sup_sup': Co-Suppression Deficit (-A, -B) -> Joint De-biasing (Suppress A AND B).
5. 'amp_neutral': Pure Isolation (+A, 0B) -> Amplify A while B remains strictly neutral.

Usage:
    # 1. Run exhaustive scan across ALL interference regimes:
    python experiments/test_data_bottlenecks.py --mode all --top_k 10

    # 2. Decoupling Deficit Scan (+A, -B):
    python experiments/test_data_bottlenecks.py --mode amp_sup --top_k 20

    # 3. Export all deficits as JSON for synthetic data generation:
    python experiments/test_data_bottlenecks.py --mode all --export_json deficits_all.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_bottlenecks(
    run_dir: str = "runs/qwen3_1.7b_batchtopk_65k",
    mode: str = "amp_sup",
    tau: float = 0.08,
    min_demand: int = 200,
    max_joint: int = 5,
    top_k: int = 20,
    target_cluster: Optional[int] = None,
    export_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run vectorized exhaustive search over (N, K) disparity score matrix across interference modes."""
    cache_dir = os.path.join(run_dir, "viewer_cache")
    labels_file = os.path.join(run_dir, "feature_cluster_labels.json")
    meta_file = os.path.join(cache_dir, "example_scores_meta.json")
    u_file = os.path.join(cache_dir, "example_u.npy")
    s_file = os.path.join(cache_dir, "example_s.npy")
    u_mat = None
    s_mat = None
    cluster_ids = None

    if os.path.exists(u_file) and os.path.exists(s_file) and os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cluster_ids = meta.get("cluster_ids", [])
        print(f"Loading score matrices from viewer cache '{cache_dir}' for mode '{mode}'...")
        u_mat = np.load(u_file, mmap_mode="r")
        s_mat = np.load(s_file, mmap_mode="r")
    else:
        summary_file = os.path.join(run_dir, "pdd_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
            ckpt_dir = summary.get("checkpoint_subfolder") or summary.get("config", {}).get("checkpoint_dir")
            fc_file = os.path.join(ckpt_dir, "feature_conditioned.npz") if ckpt_dir else None
            if fc_file and os.path.exists(fc_file):
                from pdd.feature_conditioned import FeatureConditionedResult
                fc = FeatureConditionedResult.load_checkpoint(fc_file)
                u_mat = fc.u_matrix
                s_mat = fc.s_matrix
                labels_raw_data = {}
                if os.path.exists(labels_file):
                    with open(labels_file, "r", encoding="utf-8") as lf:
                        labels_raw_data = json.load(lf).get("feature_clusters", {})
                cluster_ids = sorted(int(k) for k in labels_raw_data.keys()) if labels_raw_data else list(range(u_mat.shape[1]))
                print(f"Loaded score matrices directly from '{fc_file}' ({u_mat.shape})...")

    if u_mat is None or s_mat is None or cluster_ids is None:
        print(f"Error: Could not load score matrices from {run_dir}.", file=sys.stderr)
        sys.exit(1)

    labels_raw = {}
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            labels_raw = raw_data.get("feature_clusters", raw_data)

    K = len(cluster_ids)
    N = u_mat.shape[0]

    amp = ((u_mat > tau) & (s_mat > 0)).astype(np.float32)
    sup = ((u_mat < -tau) & (s_mat > 0)).astype(np.float32)
    neutral = ((np.abs(u_mat) <= tau) & (s_mat > 0)).astype(np.float32)

    if mode == "amp_sup":
        mat_a = amp
        mat_b = sup
        mode_desc = "Decoupling (+A, -B)"
    elif mode == "amp_amp":
        mat_a = amp
        mat_b = amp
        mode_desc = "Co-Amplification Multi-Skill (+A, +B)"
    elif mode == "sup_sup":
        mat_a = sup
        mat_b = sup
        mode_desc = "Co-Suppression De-biasing (-A, -B)"
    elif mode == "amp_neutral":
        mat_a = amp
        mat_b = neutral
        mode_desc = "Pure Isolation (+A, 0B)"
    else:
        print(f"Error: Unsupported mode '{mode}'. Choose from 'all', 'amp_sup', 'amp_amp', 'sup_sup', 'amp_neutral'.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    joint_matrix = (mat_a.T @ mat_b).astype(np.int32)
    totals_a = mat_a.sum(axis=0).astype(np.int32)
    totals_b = mat_b.sum(axis=0).astype(np.int32)
    elapsed = time.time() - t0

    np.fill_diagonal(joint_matrix, -1)

    total_pairs = K * (K - 1)
    zero_count = int((joint_matrix == 0).sum())
    under_5_count = int(((joint_matrix >= 0) & (joint_matrix <= 5)).sum())
    under_10_count = int(((joint_matrix >= 0) & (joint_matrix <= 10)).sum())

    print(f"Exhaustive search over {total_pairs:,} pairs across {N:,} examples completed in {elapsed:.2f}s!")
    print("=" * 85)
    print(f"INTERFERENCE REGIME: {mode_desc.upper()}")
    print(f"  * Total Off-Diagonal Pairs    : {total_pairs:,}")
    print(f"  * Pairs with EXACTLY 0 samples: {zero_count:,} ({(zero_count / total_pairs) * 100:.1f}%)")
    print(f"  * Pairs with <= 5 samples     : {under_5_count:,} ({(under_5_count / total_pairs) * 100:.1f}%)")
    print(f"  * Pairs with <= 10 samples    : {under_10_count:,} ({(under_10_count / total_pairs) * 100:.1f}%)")
    print("=" * 85 + "\n")

    candidates: List[Dict[str, Any]] = []
    for i in range(K):
        ma = cluster_ids[i]
        if target_cluster is not None and ma != target_cluster:
            continue
        if totals_a[i] < min_demand:
            continue

        for j in range(K):
            if i == j:
                continue
            c = int(joint_matrix[i, j])
            if c <= max_joint and totals_b[j] >= 50:
                mb = cluster_ids[j]
                ta = labels_raw.get(str(ma), {}).get("title", f"T_{ma}")
                tb = labels_raw.get(str(mb), {}).get("title", f"T_{mb}")
                candidates.append({
                    "mode": mode,
                    "target_cluster_a": ma,
                    "target_title_a": ta,
                    "demand_count_a": int(totals_a[i]),
                    "interference_cluster_b": mb,
                    "interference_title_b": tb,
                    "demand_count_b": int(totals_b[j]),
                    "joint_matching_samples": c,
                    "joint_ratio": float(c / max(1, totals_a[i])),
                    "deficit_severity": "Critical Zero" if c == 0 else ("Severe Deficit" if c <= 3 else "Moderate Deficit"),
                })

    candidates.sort(key=lambda x: (x["joint_matching_samples"], -x["demand_count_a"]))

    print(f"Identified {len(candidates)} Severe Interference Deficits (Joint <= {max_joint}, Demand >= {min_demand}):\n")
    for idx, c in enumerate(candidates[:top_k], 1):
        ma, ta, na = c["target_cluster_a"], c["target_title_a"], c["demand_count_a"]
        mb, tb, nb = c["interference_cluster_b"], c["interference_title_b"], c["demand_count_b"]
        j = c["joint_matching_samples"]
        sev = c["deficit_severity"]
        pct = c["joint_ratio"] * 100
        print(f"[{idx}] {sev.upper()}:")
        print(f"  ▲ Concept A: T_{ma} ({ta}) [Individual Demand = {na:,} pairs]")
        print(f"  ◆ Concept B: T_{mb} ({tb}) [Individual Demand = {nb:,} pairs]")
        print(f"  -> JOINT SAMPLES IN DATASET: {j} pair(s) ({pct:.3f}% of Concept A)")
        print(f"  -> Fix: Synthesize {min(500, na)} pairs satisfying ({mode})\n")

    if export_path:
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_dir": run_dir,
                "mode": mode,
                "tau": tau,
                "total_deficits_found": len(candidates),
                "deficits": candidates,
            }, f, indent=2)
        print(f"Saved deficit export to '{export_path}'.")

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Find behavioral interference & multi-objective training data deficits.")
    parser.add_argument("--run_dir", type=str, default="runs/qwen3_1.7b_batchtopk_65k", help="Target run directory with precomputed viewer cache")
    parser.add_argument("--mode", type=str, default="amp_sup", choices=["all", "amp_sup", "amp_amp", "sup_sup", "amp_neutral"], help="Interference mode: 'all' (runs all 3 main regimes), 'amp_sup' (+A, -B), 'amp_amp' (+A, +B), 'sup_sup' (-A, -B), 'amp_neutral' (+A, 0B)")
    parser.add_argument("--tau", type=float, default=0.08, help="Disparity threshold tau (default: 0.08)")
    parser.add_argument("--min_demand", type=int, default=200, help="Minimum individual sample count for Target A (default: 200)")
    parser.add_argument("--max_joint", type=int, default=5, help="Maximum joint matching samples to qualify as bottleneck (default: 5)")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top bottleneck pairs to display (default: 20)")
    parser.add_argument("--cluster", type=int, default=None, help="Inspect deficits for a specific Target Cluster ID (e.g. 10)")
    parser.add_argument("--export_json", type=str, default=None, help="Path to export bottleneck list as JSON")
    args = parser.parse_args()

    find_bottlenecks(
        run_dir=args.run_dir,
        mode=args.mode,
        tau=args.tau,
        min_demand=args.min_demand,
        max_joint=args.max_joint,
        top_k=args.top_k,
        target_cluster=args.cluster,
        export_path=args.export_json,
    )


if __name__ == "__main__":
    main()
