"""Phase P5 Experiment: Predictive Data Debugging Interventions.

Replicates Goodfire's paper (arXiv:2606.12360, Appendix B.1 & B.2):
Performs real dataset inoculation (purging problematic data clusters B_k)
and computes real DPO loss reweighting vectors based on actual feature primitives (u_matrix).
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from pdd.config import PipelineConfig
from pdd.data import DatasetLoader, PreferenceExample
from pdd.feature_clusters import FeatureClusterMap
from pdd.feature_conditioned import FeatureConditionedPipeline
from pdd.feature_matrices import FeatureMatrices
from pdd.interventions import DatasetInoculator, LossReweighter
from pdd.logger import get_logger
from pdd.sae import ModelBackend
from experiments.p4_dpo_validation import DPODataset, train_dpo_model

logger = get_logger("PDD.Exp.P5")


def main():
    parser = argparse.ArgumentParser(description="Phase P5: Predictive Data Interventions")
    parser.add_argument("--config", type=str, default="configs/gemma2_2b_base.json", help="Path to JSON config")
    parser.add_argument("--purge_clusters", type=int, nargs="+", default=[1, 5], help="Data cluster IDs to purge")
    parser.add_argument("--train_gpu", action="store_true", default=True, help="Execute real GPU DPO fine-tuning on inoculated dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU step")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for DPO training")
    args = parser.parse_args()

    cfg = PipelineConfig.load_json(args.config)
    logger.info(f"=== Phase P5 Experiment: Real Data Interventions & Inoculated DPO Training for '{cfg.name}' ===")

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

    N = len(examples)

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

    # 4. Compute REAL primitives (s, u, v) and data clusters B_k over actual feature clusters
    fc_pipeline = FeatureConditionedPipeline(cfg.feature_conditioned)
    fc_res = fc_pipeline.run(mats, cluster_map, seed=cfg.seed)
    u_matrix = fc_res.u_matrix
    cluster_assignments = fc_res.cluster_assignments

    # 5. Perform REAL Dataset Inoculation
    inoculated_examples = DatasetInoculator.filter_examples_by_cluster(
        examples=examples,
        cluster_assignments=cluster_assignments,
        purge_cluster_ids=args.purge_clusters,
    )

    # 6. Compute REAL DPO Loss Reweighting
    target_cluster = args.purge_clusters[0] if args.purge_clusters else 1
    weights = LossReweighter.compute_disparity_weights(u_matrix, target_cluster_idx=target_cluster)

    # 7. Execute REAL GPU DPO Fine-Tuning on Inoculated Dataset
    inoculated_dpo_trained = False
    if args.train_gpu:
        logger.info(f"Loading model on GPU for real Inoculated DPO fine-tuning ({len(inoculated_examples):,} examples)...")
        model_backend = ModelBackend(cfg.model)
        model, tokenizer = model_backend.load()

        dpo_dataset = DPODataset(inoculated_examples, tokenizer)
        train_dpo_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dpo_dataset,
            device=cfg.model.device,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            beta=0.1,
            lr=args.lr,
        )
        inoculated_dpo_trained = True

    output_dir = os.path.join(cfg.output_dir, "p5_interventions")
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "interventions_summary.json")

    summary = {
        "original_examples": N,
        "inoculated_examples": len(inoculated_examples),
        "purged_clusters": args.purge_clusters,
        "mean_loss_weight": float(np.mean(weights)),
        "min_loss_weight": float(np.min(weights)),
        "max_loss_weight": float(np.max(weights)),
        "inoculated_dpo_trained_on_gpu": inoculated_dpo_trained,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[Phase P5 Done] Inoculated {len(inoculated_examples):,}/{N:,} examples and completed GPU DPO training. Summary saved to '{summary_file}'")


if __name__ == "__main__":
    main()
