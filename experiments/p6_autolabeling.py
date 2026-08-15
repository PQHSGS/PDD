"""Phase P6 Experiment: Data Cluster Auto-Labeling (Appendix B.1.7).

Replicates Goodfire's paper (arXiv:2606.12360, Appendix B.1.7):
Samples centroid-nearest and random prompts per Spherical K-Means data cluster B_k
from actual extracted feature primitives (s_matrix) and dataset examples.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from pdd.autolabel import ClusterAutoLabeler
from pdd.config import PipelineConfig
from pdd.data import DatasetLoader, PreferenceExample
from pdd.feature_clusters import FeatureClusterMap
from pdd.feature_conditioned import FeatureConditionedPipeline
from pdd.feature_matrices import FeatureMatrices
from pdd.logger import get_logger

logger = get_logger("PDD.Exp.P6")


def main():
    parser = argparse.ArgumentParser(description="Phase P6: Data Cluster Auto-Labeling")
    parser.add_argument("--config", type=str, default="configs/gemma2_2b_base.json", help="Path to JSON config")
    parser.add_argument("--num_clusters", type=int, default=5, help="Number of data clusters to label")
    args = parser.parse_args()

    cfg = PipelineConfig.load_json(args.config)
    logger.info(f"=== Phase P6 Experiment: Real Data Cluster Auto-Labeling for '{cfg.name}' ===")

    run_dir = cfg.output_dir
    summary_path = os.path.join(run_dir, "pdd_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"PDD summary file not found at '{summary_path}'. Please run PDD pipeline first.")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    subfolder = summary_data.get("checkpoint_subfolder")
    if not subfolder or not os.path.exists(subfolder):
        raise FileNotFoundError(f"Checkpoint subfolder '{subfolder}' not found.")

    # 1. Load actual cached examples
    ex_path = os.path.join(subfolder, "examples.json")
    if not os.path.exists(ex_path):
        loader = DatasetLoader(cfg.data)
        examples = loader.load()
    else:
        with open(ex_path, "r", encoding="utf-8") as f:
            ex_dicts = json.load(f)
        examples = [PreferenceExample.from_dict(d) for d in ex_dicts]

    # 2. Load actual Feature Clusters mapping
    clusters_path = os.path.join(subfolder, "clusters.json")
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)

    feat_to_cluster = {int(k): int(v) for k, v in clusters_data.get("feature_to_cluster", {}).items()}
    retained_clusters = {int(k): [int(x) for x in v] for k, v in clusters_data.get("clusters", {}).items()}
    cluster_map = FeatureClusterMap(clusters=retained_clusters, feature_to_cluster=feat_to_cluster)

    # 3. Load actual Feature Matrices
    mmap_dir = os.path.join(subfolder, "matrices_mmap")
    npz_path = os.path.join(subfolder, "matrices.npz")
    if os.path.isdir(mmap_dir):
        mats = FeatureMatrices.load_mmap_dir(mmap_dir)
    elif os.path.exists(npz_path):
        mats = FeatureMatrices.load_npz(npz_path)
    else:
        raise FileNotFoundError(f"No feature matrices found in '{subfolder}'.")

    # 4. Compute REAL primitives (s_matrix) and data clusters B_k over actual feature clusters
    fc_pipeline = FeatureConditionedPipeline(cfg.feature_conditioned)
    fc_res = fc_pipeline.run(mats, cluster_map, seed=cfg.seed)
    s_matrix = fc_res.s_matrix
    cluster_assignments = fc_res.cluster_assignments

    # 5. Sample REAL prompts and generate auto-labels
    labeler = ClusterAutoLabeler(max_prompt_chars=600)
    labels = []

    unique_clusters = sorted(list(set(cluster_assignments.tolist())))[: args.num_clusters + 1]

    for k in unique_clusters:
        centroid_p, sample_p = labeler.sample_cluster_prompts(
            examples=examples,
            cluster_assignments=cluster_assignments,
            s_matrix=s_matrix,
            cluster_id=k,
            seed=cfg.seed,
        )
        label = labeler.generate_label(k, centroid_p, sample_p)
        labels.append(asdict(label))

    output_dir = os.path.join(cfg.output_dir, "p6_autolabeling")
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "cluster_labels.json")

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"total_clusters": len(labels), "labels": labels}, f, indent=2)

    logger.info(f"[Phase P6 Done] Real auto-labeling generated labels for {len(labels)} data clusters. Saved to '{summary_file}'")


if __name__ == "__main__":
    main()
