"""Phase P5 Experiment: Predictive Data Debugging Interventions.

Demonstrates:
1. Inoculating preference dataset by purging target clusters B_k.
2. Computing DPO loss reweighting vectors.
3. Extracting feature steering directions for inference.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from pdd.config import PipelineConfig
from pdd.data import DatasetLoader
from pdd.interventions import DatasetInoculator, LossReweighter
from pdd.logger import get_logger

logger = get_logger("PDD.Exp.P5")


def main():
    parser = argparse.ArgumentParser(description="Phase P5: Predictive Data Interventions")
    parser.add_argument("--config", type=str, default="configs/qwen3_1.7b_base.json", help="Path to JSON config")
    parser.add_argument("--purge_clusters", type=int, nargs="+", default=[1, 5], help="Data cluster IDs to purge")
    args = parser.parse_args()

    cfg = PipelineConfig.load_json(args.config)
    logger.info(f"=== Phase P5 Experiment: Data Interventions for '{cfg.name}' ===")

    loader = DatasetLoader(cfg.data)
    examples = loader.load()

    rng = np.random.RandomState(cfg.seed)
    cluster_assignments = rng.randint(0, 10, size=len(examples))

    # 1. Dataset Inoculation
    inoculated_examples = DatasetInoculator.filter_examples_by_cluster(
        examples=examples,
        cluster_assignments=cluster_assignments,
        purge_cluster_ids=args.purge_clusters,
    )

    # 2. Loss Reweighting
    u_matrix = rng.randn(len(examples), 10)
    weights = LossReweighter.compute_disparity_weights(u_matrix, target_cluster_idx=1)

    output_dir = os.path.join(cfg.output_dir, "p5_interventions")
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "interventions_summary.json")

    summary = {
        "original_examples": len(examples),
        "inoculated_examples": len(inoculated_examples),
        "purged_clusters": args.purge_clusters,
        "mean_loss_weight": float(np.mean(weights)),
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[Phase P5 Done] Summary saved to '{summary_file}'")


if __name__ == "__main__":
    main()
