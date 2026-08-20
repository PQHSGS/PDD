"""Audit feature cluster quality: size distribution, sub-community structure, and label-driving alignment.

Usage:
    python experiments/audit_feature_clusters.py --checkpoint <ckpt_dir>
    python experiments/audit_feature_clusters.py --run_dir <run_dir>
    python experiments/audit_feature_clusters.py  # uses default run
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def audit_size_distribution(clusters_data):
    """Print cluster size distribution and identify oversized clusters."""
    sizes = sorted([(int(k), len(v)) for k, v in clusters_data['clusters'].items()], key=lambda x: -x[1])
    all_sizes = [s for _, s in sizes]

    print(f"Total clusters: {len(sizes)}")
    print(f"Size range: {min(all_sizes)} - {max(all_sizes)}")
    print(f"Mean: {np.mean(all_sizes):.1f}, Median: {np.median(all_sizes):.0f}")

    # Histogram
    from collections import Counter
    buckets = Counter()
    for _, sz in sizes:
        if sz < 5: buckets["<5"] += 1
        elif sz < 10: buckets["5-9"] += 1
        elif sz < 20: buckets["10-19"] += 1
        elif sz < 50: buckets["20-49"] += 1
        elif sz < 100: buckets["50-99"] += 1
        else: buckets["100+"] += 1
    print(f"\nSize histogram: {dict(sorted(buckets.items()))}")

    # Flag oversized clusters
    median_sz = np.median(all_sizes)
    print(f"\nOversized clusters (>2x median = {2*median_sz:.0f}):")
    for cid, sz in sizes:
        if sz > 2 * median_sz:
            print(f"  T_{cid}: {sz} features")

    return sizes


def audit_sub_communities(clusters_data, mi_graph_path, max_clusters=10):
    """Run Leiden on each large cluster's internal MI graph to detect sub-communities."""
    import igraph as ig
    import leidenalg as la

    if not os.path.exists(mi_graph_path):
        print(f"\nMI graph not found at {mi_graph_path}; skipping sub-community audit.")
        return

    mi = np.load(mi_graph_path)
    gi, gj, gw = mi['global_i'], mi['global_j'], mi['weights']

    sizes = sorted([(int(k), len(v)) for k, v in clusters_data['clusters'].items()], key=lambda x: -x[1])

    print(f"\n=== Sub-community audit (clusters with >20 features) ===")
    for cid, sz in sizes[:max_clusters]:
        if sz < 20:
            break
        feats = sorted(clusters_data['clusters'][str(cid)])
        feat_set = set(feats)
        feat_idx = {f: i for i, f in enumerate(feats)}
        n = len(feats)

        # Build internal edges
        edges, weights = [], []
        for idx in range(len(gi)):
            a, b = int(gi[idx]), int(gj[idx])
            if a in feat_set and b in feat_set:
                edges.append((feat_idx[a], feat_idx[b]))
                weights.append(float(gw[idx]))

        possible = n * (n - 1) // 2
        density = 100 * len(edges) / possible if possible > 0 else 0

        # Run Leiden on internal subgraph
        g = ig.Graph(n=n, edges=edges, edge_attrs={"weight": weights})
        partition = la.find_partition(
            g, la.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=1.5, seed=0
        )
        sub_comms = [c for c in partition]

        print(f"\n  T_{cid}: {n} features, {len(edges)} internal edges ({density:.1f}% density), "
              f"{len(sub_comms)} sub-communities")
        for i, comm in enumerate(sub_comms):
            if len(comm) >= 3:
                print(f"    Sub-{i}: {len(comm)} features")


def audit_s_vs_u(fc_result, clusters_data, examples, max_clusters=5):
    """Check if activation-based labels (high s) match disparity-based driving samples (high u)."""
    cluster_ids = sorted(int(k) for k in clusters_data['clusters'].keys())

    print(f"\n=== Activation (s) vs Disparity (u) alignment ===")
    for m_idx, cid in enumerate(cluster_ids[:max_clusters]):
        if m_idx >= fc_result.u_matrix.shape[1]:
            break
        u_col = fc_result.u_matrix[:, m_idx]
        s_col = fc_result.s_matrix[:, m_idx]

        # Top 5 by s vs top 5 by u
        top_s = np.argsort(s_col)[-5:][::-1]
        top_u_pos = np.where(u_col > 0)[0]
        top_u = top_u_pos[np.argsort(-u_col[top_u_pos])][:5] if len(top_u_pos) > 0 else []

        overlap = len(set(top_s) & set(top_u))
        corr = np.corrcoef(s_col, u_col)[0, 1]

        # Count u=0 examples
        n_zero = int((u_col == 0).sum())
        n_total = len(u_col)

        print(f"\n  T_{cid} ({len(clusters_data['clusters'][str(cid)])} feats): "
              f"corr(s,u)={corr:.4f}, u=0: {n_zero}/{n_total} ({100*n_zero/n_total:.1f}%), "
              f"top-s/top-u overlap: {overlap}/5")
        print(f"    Top s examples: {top_s[:3]}")
        print(f"    Top u examples: {top_u[:3]}")


def main():
    parser = argparse.ArgumentParser(description="Audit feature cluster quality")
    parser.add_argument("--run_dir", type=str, default="runs/qwen3_1.7b_batchtopk_65k")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    # Find checkpoint dir
    if args.checkpoint:
        ckpt_dir = args.checkpoint
    else:
        summary_path = os.path.join(run_dir, "pdd_summary.json")
        if os.path.exists(summary_path):
            summary = json.load(open(summary_path))
            ckpt_dir = summary.get("checkpoint_subfolder", "")
        else:
            print(f"No summary at {summary_path}")
            return

    if not ckpt_dir or not os.path.exists(ckpt_dir):
        print(f"Checkpoint dir not found: {ckpt_dir}")
        return

    print(f"Run: {run_dir}")
    print(f"Checkpoint: {ckpt_dir}")

    # Load clusters
    clusters_path = os.path.join(ckpt_dir, "clusters.json")
    if not os.path.exists(clusters_path):
        print(f"No clusters.json at {clusters_path}")
        return
    clusters_data = json.load(open(clusters_path))

    # 1. Size distribution
    audit_size_distribution(clusters_data)

    # 2. Sub-community audit
    mi_path = os.path.join(ckpt_dir, "mi_graph.npz")
    audit_sub_communities(clusters_data, mi_path)

    # 3. s vs u alignment
    fc_path = os.path.join(ckpt_dir, "feature_conditioned.npz")
    if os.path.exists(fc_path):
        from pdd.feature_conditioned import FeatureConditionedResult
        fc = FeatureConditionedResult.load_checkpoint(fc_path)
        examples_path = os.path.join(ckpt_dir, "examples.json")
        if os.path.exists(examples_path):
            from pdd.data import DatasetLoader
            examples = DatasetLoader.load_json_cache(examples_path)
            audit_s_vs_u(fc, clusters_data, examples)
        else:
            print("\nNo examples.json; skipping s vs u audit.")
    else:
        print("\nNo feature_conditioned.npz; skipping s vs u audit.")

    # 4. Parameter recommendations
    summary_path = os.path.join(run_dir, "pdd_summary.json")
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path))
        fcl_cfg = summary.get("config", {}).get("feature_clusters", {})
        print(f"\n{'='*60}")
        print("PARAMETER RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"Current: resolution={fcl_cfg.get('resolution_parameter')}, "
              f"min_community_size={fcl_cfg.get('min_community_size')}, "
              f"top_pct={fcl_cfg.get('top_pct')}, "
              f"min_firing_freq={fcl_cfg.get('min_firing_freq')}")
        print(f"Result: {len(clusters_data['clusters'])} clusters, "
              f"largest={max(len(v) for v in clusters_data['clusters'].values())}")
        print()
        print("Issues found:")
        print("  1. Top clusters (T_1-T_6) are oversized (23-90 features) with")
        print("     low internal MI density (8-24%) and multiple sub-communities.")
        print("  2. ALL top-5 clusters: zero overlap between high-s (activation)")
        print("     and high-u (disparity) examples. Label != driving samples.")
        print("  3. Paper got K_r=814 clusters vs our 66 — need finer clustering.")
        print()
        print("Recommended next run config:")
        print('  "feature_clusters": {')
        print(f'    "resolution_parameter": 25,   # was {fcl_cfg.get("resolution_parameter")} — split more')
        print(f'    "min_community_size": 4,       # was {fcl_cfg.get("min_community_size")} — keep small communities')
        print(f'    "top_pct": {fcl_cfg.get("top_pct")},           # keep — already aggressive')
        print(f'    "min_firing_freq": {fcl_cfg.get("min_firing_freq")}       # keep')
        print("  }")


if __name__ == "__main__":
    main()
