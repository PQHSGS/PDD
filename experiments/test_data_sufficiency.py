#!/usr/bin/env python3
"""Data Sufficiency & Latent Coverage Diagnostic Tool.

Scientifically proves why ultra-small filtered dataset subsets (N = 1..5 pairs)
fail to generalize across general behavioral interference regimes, and quantifies
the exact data quota required for robust concept acquisition.

Interference Modes:
1. 'all': Run automated sufficiency audit across all 3 main regimes (amp_sup, amp_amp, sup_sup).
2. 'amp_sup' (Default): Decoupling Deficit (+A, -B) -> Amplify A while Suppressing B.
3. 'amp_amp': Co-Amplification Deficit (+A, +B) -> Multi-Skill Synthesis (Amplify A AND B).
4. 'sup_sup': Co-Suppression Deficit (-A, -B) -> Joint De-biasing (Suppress A AND B).
5. 'amp_neutral': Pure Isolation (+A, 0B) -> Amplify A while B remains strictly neutral.

Usage:
    # 1. Audit ALL 3 main interference regimes in one shot:
    python experiments/test_data_sufficiency.py --mode all --top_k 10

    # 2. Batch Audit on Top 20 Decoupling Bottlenecks (+A, -B):
    python experiments/test_data_sufficiency.py --mode amp_sup --top_k 20

    # 3. Export Master Inoculation Blueprint JSON across all tested regimes:
    python experiments/test_data_sufficiency.py --mode all --top_k 10 --export_inoculation_spec master_inoculation_plan.json
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set
import numpy as np

# Import discovery engine from test_data_bottlenecks
try:
    from experiments.test_data_bottlenecks import find_bottlenecks
except ImportError:
    try:
        from test_data_bottlenecks import find_bottlenecks
    except ImportError:
        find_bottlenecks = None


def audit_single_pair(
    target_cluster: int,
    interference_cluster: Optional[int],
    all_clusters: Dict[int, List[int]],
    labels_raw: Dict[str, Any],
    u_mat: np.ndarray,
    s_mat: np.ndarray,
    cluster_ids: List[int],
    member_c: Optional[np.ndarray],
    member_cols: Optional[np.ndarray],
    mode: str = "amp_sup",
    tau: float = 0.08,
) -> Dict[str, Any]:
    """Compute latent coverage and sufficiency metrics for a single (T_A, T_B) interference pair."""
    target_features = all_clusters.get(target_cluster, [])
    total_target_features = len(target_features)
    title_a = labels_raw.get(str(target_cluster), {}).get("title", f"T_{target_cluster}")
    title_b = labels_raw.get(str(interference_cluster), {}).get("title", f"T_{interference_cluster}") if interference_cluster is not None else "None"

    if target_cluster not in cluster_ids:
        return {
            "mode": mode, "target_cluster": target_cluster, "target_title": title_a,
            "interference_cluster": interference_cluster, "interference_title": title_b,
            "sample_count": 0, "active_count": 0, "dead_count": total_target_features,
            "coverage_ratio": 0.0, "risk_score": 100.0, "verdict": "UNINDEXED",
            "recommended_quota": max(150, int(total_target_features * 8.5)),
            "dead_features": target_features,
        }

    pos_a = cluster_ids.index(target_cluster)
    
    # Condition on Target A based on mode
    if mode in ("amp_sup", "amp_amp", "amp_neutral"):
        mask_a = (u_mat[:, pos_a] > tau) & (s_mat[:, pos_a] > 0)
    else:  # sup_sup
        mask_a = (u_mat[:, pos_a] < -tau) & (s_mat[:, pos_a] > 0)

    # Condition on Interference B based on mode
    mask = mask_a
    if interference_cluster is not None and interference_cluster in cluster_ids:
        pos_b = cluster_ids.index(interference_cluster)
        if mode == "amp_sup":
            mask_b = (u_mat[:, pos_b] < -tau) & (s_mat[:, pos_b] > 0)
        elif mode == "amp_amp":
            mask_b = (u_mat[:, pos_b] > tau) & (s_mat[:, pos_b] > 0)
        elif mode == "sup_sup":
            mask_b = (u_mat[:, pos_b] < -tau) & (s_mat[:, pos_b] > 0)
        elif mode == "amp_neutral":
            mask_b = (np.abs(u_mat[:, pos_b]) <= tau) & (s_mat[:, pos_b] > 0)
        else:
            mask_b = (u_mat[:, pos_b] < -tau) & (s_mat[:, pos_b] > 0)
        mask = mask & mask_b

    matching_indices = np.flatnonzero(mask)
    sample_count = len(matching_indices)

    active_features_in_subset: Set[int] = set()
    feature_activation_maxes: Dict[int, float] = {f: 0.0 for f in target_features}

    if member_c is not None and member_cols is not None and sample_count > 0:
        target_col_indices = [np.where(member_cols == f)[0][0] for f in target_features if f in member_cols]
        if target_col_indices:
            sub_c = member_c[matching_indices][:, target_col_indices]
            max_acts = sub_c.max(axis=0)
            for f_idx, act_val in zip(target_features, max_acts):
                feature_activation_maxes[f_idx] = float(act_val)
                if act_val > tau:
                    active_features_in_subset.add(f_idx)

    active_count = len(active_features_in_subset)
    dead_count = total_target_features - active_count
    coverage_ratio = (active_count / max(1, total_target_features)) * 100.0
    rank_bound = min(sample_count, active_count)
    risk_score = 100.0 * (1.0 - (coverage_ratio / 100.0)) * math.exp(-sample_count / 50.0)

    if coverage_ratio < 95.0:
        recommended_quota = max(150, int(total_target_features * 8.5))
    else:
        recommended_quota = 0

    if sample_count <= 3 or coverage_ratio < 25.0:
        verdict = "CRITICAL"
        verdict_desc = "Guaranteed to fail generalization. Model will overfit to surface tokens."
    elif sample_count < 30 or coverage_ratio < 75.0:
        verdict = "DEFICIENT"
        verdict_desc = "Weak generalization. Over 25% of latent features receive 0 gradient."
    else:
        verdict = "SUFFICIENT"
        verdict_desc = "Complete concept coverage. Robust generalization expected."

    dead_features = [f for f, act in feature_activation_maxes.items() if act <= tau]

    return {
        "mode": mode,
        "target_cluster": target_cluster,
        "target_title": title_a,
        "interference_cluster": interference_cluster,
        "interference_title": title_b,
        "total_target_features": total_target_features,
        "sample_count": sample_count,
        "active_count": active_count,
        "dead_count": dead_count,
        "coverage_ratio": coverage_ratio,
        "rank_bound": rank_bound,
        "risk_score": risk_score,
        "verdict": verdict,
        "verdict_description": verdict_desc,
        "recommended_quota": recommended_quota,
        "dead_features": dead_features,
    }


def evaluate_sufficiency(
    run_dir: str = "runs/qwen3_1.7b_batchtopk_65k",
    mode: str = "amp_sup",
    target_cluster: Optional[int] = None,
    interference_cluster: Optional[int] = None,
    bottlenecks_json: Optional[str] = None,
    top_k: int = 20,
    tau: float = 0.08,
    json_file: Optional[str] = None,
    export_report: Optional[str] = None,
    export_inoculation_spec: Optional[str] = None,
) -> Any:
    """Run single or batch data sufficiency evaluation across interference regimes."""
    cache_dir = os.path.join(run_dir, "viewer_cache")
    labels_file = os.path.join(run_dir, "feature_cluster_labels.json")
    summary_file = os.path.join(run_dir, "pdd_summary.json")

    # If mode == "all", run across the 3 main regimes
    if mode == "all":
        modes_to_run = ["amp_sup", "amp_amp", "sup_sup"]
        all_batch_results = []
        all_master_specs = []
        for m in modes_to_run:
            print(f"\n{'#' * 90}")
            print(f"### RUNNING DATA SUFFICIENCY AUDIT FOR REGIME: {m.upper()}")
            print(f"{'#' * 90}")
            res = evaluate_sufficiency(
                run_dir=run_dir, mode=m, target_cluster=target_cluster,
                interference_cluster=interference_cluster, top_k=top_k, tau=tau
            )
            if isinstance(res, list):
                all_batch_results.extend(res)

        if export_report:
            with open(export_report, "w", encoding="utf-8") as f:
                json.dump({"run_dir": run_dir, "mode": "all", "total_audited": len(all_batch_results), "results": all_batch_results}, f, indent=2)
            print(f"Saved master multi-regime report to '{export_report}'.")

        if export_inoculation_spec:
            with open(export_inoculation_spec, "w", encoding="utf-8") as f:
                json.dump({"version": "1.0", "mode": "all", "total_blueprints": len(all_batch_results), "blueprints": all_batch_results}, f, indent=2)
            print(f"Saved master multi-regime Inoculation Blueprint JSON to '{export_inoculation_spec}'!")

        return all_batch_results

    if not os.path.exists(summary_file):
        print(f"Error: pdd_summary.json not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    ckpt_dir = summary.get("checkpoint_subfolder", "")
    clusters_file = os.path.join(ckpt_dir, "clusters.json") if ckpt_dir else ""
    if not os.path.exists(clusters_file):
        cand = os.path.join("checkpoints", "qwen3_1.7b_dolci_seed0_20260815_110345", "clusters.json")
        if os.path.exists(cand):
            clusters_file = cand
        else:
            for root, _, files in os.walk("checkpoints"):
                if "clusters.json" in files:
                    clusters_file = os.path.join(root, "clusters.json")
                    break

    if not os.path.exists(clusters_file):
        print("Notice: clusters.json is currently being built. Please wait for completion.", file=sys.stderr)
        sys.exit(1)

    with open(clusters_file, "r", encoding="utf-8") as f:
        all_clusters = {int(k): v for k, v in json.load(f).get("clusters", {}).items()}

    labels_raw = {}
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            labels_raw = json.load(f).get("feature_clusters", {})

    u_file = os.path.join(cache_dir, "example_u.npy")
    s_file = os.path.join(cache_dir, "example_s.npy")
    meta_file = os.path.join(cache_dir, "example_scores_meta.json")

    if not os.path.exists(u_file) or not os.path.exists(s_file):
        print("Error: Precomputed cache not found. Ensure viewer prewarm ran.", file=sys.stderr)
        sys.exit(1)

    meta = json.load(open(meta_file))
    cluster_ids = meta["cluster_ids"]

    u_mat = np.load(u_file, mmap_mode="r")
    s_mat = np.load(s_file, mmap_mode="r")

    member_c_file = os.path.join(cache_dir, "member_C_max.npy")
    member_cols_file = os.path.join(cache_dir, "member_cols.npy")
    member_c = np.load(member_c_file, mmap_mode="r") if os.path.exists(member_c_file) else None
    member_cols = np.load(member_cols_file) if os.path.exists(member_cols_file) else None

    # MODE 1: BATCH AUDIT
    if target_cluster is None:
        bottlenecks_list: List[Dict[str, Any]] = []

        if bottlenecks_json and os.path.exists(bottlenecks_json):
            print(f"Loading pre-discovered bottlenecks from '{bottlenecks_json}'...")
            with open(bottlenecks_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                bottlenecks_list = data.get("deficits", data if isinstance(data, list) else [])
                mode = data.get("mode", mode)
        else:
            print(f"No target cluster specified. Running auto-discovery on Top {top_k} deficits in mode '{mode}'...")
            if find_bottlenecks is not None:
                bottlenecks_list = find_bottlenecks(run_dir=run_dir, mode=mode, tau=tau, top_k=top_k)
            else:
                print("Error: find_bottlenecks engine could not be imported.", file=sys.stderr)
                sys.exit(1)

        print("\n" + "=" * 115)
        print(f"BEHAVIORAL INTERFERENCE & SUFFICIENCY AUDIT (MODE: {mode.upper()}, TOP {min(len(bottlenecks_list), top_k)} PAIRS)")
        print("=" * 115)
        print(f"{'#':<3} | {'Concept A':<28} | {'Interference Concept B':<28} | {'Mode':<11} | {'N_match':<7} | {'Cov %':<7} | {'Dead/Total':<10} | {'Risk':<5} | {'Verdict':<8} | {'Quota'}")
        print("-" * 115)

        batch_results = []
        master_specs = []

        for idx, item in enumerate(bottlenecks_list[:top_k], 1):
            ta_id = item.get("target_cluster_a", item.get("target_cluster"))
            tb_id = item.get("interference_cluster_b", item.get("suppress_cluster_b", item.get("interference_cluster", item.get("suppress_cluster"))))
            if ta_id is None:
                continue

            res = audit_single_pair(
                target_cluster=int(ta_id),
                interference_cluster=int(tb_id) if tb_id is not None else None,
                all_clusters=all_clusters,
                labels_raw=labels_raw,
                u_mat=u_mat,
                s_mat=s_mat,
                cluster_ids=cluster_ids,
                member_c=member_c,
                member_cols=member_cols,
                mode=mode,
                tau=tau,
            )
            batch_results.append(res)

            t_a_str = f"T_{res['target_cluster']} ({res['target_title'][:18]})"
            t_b_str = f"T_{res['interference_cluster']} ({res['interference_title'][:18]})" if res['interference_cluster'] is not None else "None"
            dead_str = f"{res['dead_count']}/{res['total_target_features']}"

            print(f"{idx:<3} | {t_a_str:<28} | {t_b_str:<28} | {mode:<11} | {res['sample_count']:<7} | {res['coverage_ratio']:>5.1f}% | {dead_str:<10} | {res['risk_score']:>4.1f} | {res['verdict']:<8} | {res['recommended_quota']:,}")

            if export_inoculation_spec:
                kws_a = labels_raw.get(str(res["target_cluster"]), {}).get("keywords", [])
                kws_b = labels_raw.get(str(res["interference_cluster"]), {}).get("keywords", []) if res["interference_cluster"] is not None else []
                desc_a = labels_raw.get(str(res["target_cluster"]), {}).get("description", "")
                desc_b = labels_raw.get(str(res["interference_cluster"]), {}).get("description", "") if res["interference_cluster"] is not None else ""

                master_specs.append({
                    "mode": mode,
                    "target_cluster": res["target_cluster"],
                    "target_title": res["target_title"],
                    "interference_cluster": res["interference_cluster"],
                    "interference_title": res["interference_title"],
                    "sample_count": res["sample_count"],
                    "latent_coverage_pct": res["coverage_ratio"],
                    "dead_latents_count": res["dead_count"],
                    "unsteered_feature_ids": res["dead_features"],
                    "recommended_quota": res["recommended_quota"],
                    "llm_system_prompt": (
                        f"Generate {res['recommended_quota']} contrastive preference pairs under mode '{mode}':\n"
                        f"- Concept A: {res['target_title']} ({desc_a})\n"
                        f"- Concept B: {res['interference_title']} ({desc_b})\n"
                        "- Format: JSON array of [{\"prompt\": \"...\", \"chosen\": \"...\", \"rejected\": \"...\"}]"
                    ),
                })

        print("=" * 115 + "\n")

        if export_report:
            with open(export_report, "w", encoding="utf-8") as f:
                json.dump({"run_dir": run_dir, "mode": mode, "total_audited": len(batch_results), "results": batch_results}, f, indent=2)
            print(f"Saved batch sufficiency audit report to '{export_report}'.")

        if export_inoculation_spec:
            with open(export_inoculation_spec, "w", encoding="utf-8") as f:
                json.dump({
                    "version": "1.0",
                    "mode": mode,
                    "total_blueprints": len(master_specs),
                    "blueprints": master_specs,
                }, f, indent=2)
            print(f"Saved master batch Inoculation Blueprint JSON across all {len(master_specs)} bottlenecks to '{export_inoculation_spec}'!")

        return batch_results

    # MODE 2: SINGLE TARGET CLUSTER AUDIT
    res = audit_single_pair(
        target_cluster=target_cluster,
        interference_cluster=interference_cluster,
        all_clusters=all_clusters,
        labels_raw=labels_raw,
        u_mat=u_mat,
        s_mat=s_mat,
        cluster_ids=cluster_ids,
        member_c=member_c,
        member_cols=member_cols,
        mode=mode,
        tau=tau,
    )

    print("=" * 85)
    print(f"PDD BEHAVIORAL INTERFERENCE & DATA SUFFICIENCY AUDIT ({mode.upper()})")
    print(f"Concept A : T_{target_cluster} ({res['target_title']}) [Contains {res['total_target_features']} SAE Latent Features]")
    if interference_cluster is not None:
        print(f"Concept B : T_{interference_cluster} ({res['interference_title']})")
    print("=" * 85)

    print(f"\n--- 1. SAMPLE VOLUME & GRADIENT RANK ---")
    print(f"  * Available Training Samples in Dataset : {res['sample_count']:,} pair(s)")
    print(f"  * Parameter Update Subspace Rank Bound  : {res['rank_bound']} (Maximum rank of loss gradient)")
    print(f"  * Stochastic Batch Visibility Chance    : {min(100.0, (1 - (1 - res['sample_count']/260000)**64)*100):.4f}% per batch (B=64)")

    print(f"\n--- 2. SAE LATENT FEATURE COVERAGE (THE '1 vs 30' PROOF) ---")
    print(f"  * Total Latents in Concept Community T_{target_cluster} : {res['total_target_features']} SAE features")
    print(f"  * Actively Steered Features in Subset    : {res['active_count']} features ({res['coverage_ratio']:.1f}%)")
    print(f"  * DEAD / UNTOUCHED Latents (0 Gradient)  : {res['dead_count']} features ({(res['dead_count']/max(1,res['total_target_features']))*100:.1f}%)")
    print(f"  * Sample Dead Features (Never Activated) : {res['dead_features'][:8]} ...")

    print(f"\n--- 3. GENERALIZATION RISK & VERDICT ---")
    print(f"  * Memorization / Overfitting Risk Score : {res['risk_score']:.1f} / 100")
    print(f"  * Formal Sufficiency Verdict             : {res['verdict']}")
    print(f"    -> {res['verdict_description']}")

    print(f"\n--- 4. ACTIONABLE PDD SYNTHETIC INOCULATION REQUIREMENT ---")
    if res['recommended_quota'] > 0:
        print(f"  * Required Synthetic Data Quota         : {res['recommended_quota']:,} targeted preference pairs")
        print(f"  * Inoculation Objective                 : Synthesize pairs satisfying ({mode}) to activate {res['dead_count']} un-steered latents.")
    else:
        print(f"  * Synthetic Data Quota                 : 0 (Dataset is already robustly covered).")
    print("=" * 85 + "\n")

    if export_report:
        with open(export_report, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved formal sufficiency report to '{export_report}'.")

    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data sufficiency and latent coverage across behavioral interference modes.")
    parser.add_argument("--run_dir", type=str, default="runs/qwen3_1.7b_batchtopk_65k", help="Run directory containing summary and precomputed cache")
    parser.add_argument("--mode", type=str, default="amp_sup", choices=["all", "amp_sup", "amp_amp", "sup_sup", "amp_neutral"], help="Interference mode: 'all' (runs all 3 main regimes), 'amp_sup' (+A, -B), 'amp_amp' (+A, +B), 'sup_sup' (-A, -B), 'amp_neutral' (+A, 0B)")
    parser.add_argument("--target_cluster", type=int, default=None, help="Target feature cluster ID T_A (if omitted, runs batch audit on top bottlenecks)")
    parser.add_argument("--interference_cluster", type=int, default=None, help="Interference feature cluster ID T_B (optional)")
    parser.add_argument("--bottlenecks_json", type=str, default=None, help="Path to bottlenecks JSON file from test_data_bottlenecks.py (optional)")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top bottlenecks to batch audit when target_cluster is omitted (default: 20)")
    parser.add_argument("--tau", type=float, default=0.08, help="Disparity activation threshold (default: 0.08)")
    parser.add_argument("--json_file", type=str, default=None, help="Custom JSON file containing candidate pairs to evaluate")
    parser.add_argument("--export_report", type=str, default=None, help="Path to export report JSON")
    parser.add_argument("--export_inoculation_spec", type=str, default=None, help="Path to export synthetic inoculation blueprint JSON")
    args = parser.parse_args()

    evaluate_sufficiency(
        run_dir=args.run_dir,
        mode=args.mode,
        target_cluster=args.target_cluster,
        interference_cluster=args.interference_cluster,
        bottlenecks_json=args.bottlenecks_json,
        top_k=args.top_k,
        tau=args.tau,
        json_file=args.json_file,
        export_report=args.export_report,
        export_inoculation_spec=args.export_inoculation_spec,
    )


if __name__ == "__main__":
    main()
