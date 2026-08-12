"""Phase P6 Experiment: Data Cluster Auto-Labeling (Appendix B.1.7).

Samples centroid-nearest and random prompts per Spherical K-Means data cluster B_k
and generates natural-language titles, descriptions, and keywords.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
from dataclasses import asdict

from pdd.autolabel import ClusterAutoLabeler
from pdd.config import PipelineConfig
from pdd.data import DatasetLoader
from pdd.logger import get_logger

logger = get_logger("PDD.Exp.P6")


def main():
    parser = argparse.ArgumentParser(description="Phase P6: Data Cluster Auto-Labeling")
    parser.add_argument("--config", type=str, default="configs/qwen3_1.7b_base.json", help="Path to JSON config")
    parser.add_argument("--num_clusters", type=int, default=5, help="Number of data clusters to label")
    args = parser.parse_args()

    cfg = PipelineConfig.load_json(args.config)
    logger.info(f"=== Phase P6 Experiment: Auto-Labeling Data Clusters for '{cfg.name}' ===")

    loader = DatasetLoader(cfg.data)
    examples = loader.load()

    N = len(examples)
    rng = np.random.RandomState(cfg.seed)
    cluster_assignments = rng.randint(0, args.num_clusters + 1, size=N)
    s_matrix = rng.uniform(0, 1, size=(N, 10))

    labeler = ClusterAutoLabeler(max_prompt_chars=600)
    labels = []

    for k in range(args.num_clusters + 1):
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

    logger.info(f"[Phase P6 Done] Generated labels for {len(labels)} clusters. Saved to '{summary_file}'")


if __name__ == "__main__":
    main()
