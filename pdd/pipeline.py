"""Master Orchestrator Pipeline for Predictive Data Debugging with timestamped checkpoint subfolders."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
import time
from typing import Dict, Any, Optional

from .config import PipelineConfig
from .data import DatasetLoader
from .feature_clusters import LeidenFeatureClusterer
from .feature_conditioned import FeatureConditionedPipeline
from .feature_matrices import FeatureMatrixExtractor
from .logger import get_logger
from .prompt_conditioned import PromptConditionedPipeline
from .sae import ModelBackend, SAEBackend

logger = get_logger("PDD.Pipeline")


class PDDPipeline:
    """Master Orchestrator class managing the end-to-end PDD workflow."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.cfg.validate()

    def run(self) -> Dict[str, Any]:
        """Execute full PDD pipeline with timestamped subfolder checkpointing & logging."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        logger.info(f"=== Starting Predictive Data Debugging Pipeline: '{self.cfg.name}' ===")

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        # Resolve or create timestamped subfolder under checkpoints/
        run_ckpt_dir = self._resolve_checkpoint_subfolder(timestamp_str)
        logger.info(f"Seed: {self.cfg.seed} | Checkpoint Subfolder: {run_ckpt_dir} | Output: {self.cfg.output_dir}")

        examples_ckpt = os.path.join(run_ckpt_dir, "examples.json")
        matrices_ckpt = os.path.join(run_ckpt_dir, "matrices.npz")
        clusters_ckpt = os.path.join(run_ckpt_dir, "clusters.json")
        manifest_ckpt = os.path.join(run_ckpt_dir, "manifest.json")

        # 1. Dataset Loading (cached from JSON if available)
        data_loader = DatasetLoader(self.cfg.data)
        examples = data_loader.load(
            checkpoint_path=examples_ckpt,
            use_checkpoint=self.cfg.use_checkpoint,
        )


        # 2. Checkpoint check for feature matrices
        needs_model_load = not (self.cfg.use_checkpoint and os.path.exists(matrices_ckpt))

        if needs_model_load:
            # Load Model & SAE
            model_backend = ModelBackend(self.cfg.model)
            model, tokenizer = model_backend.load()

            sae_backend = SAEBackend(self.cfg.sae)
            sae = sae_backend.load()

            extractor = FeatureMatrixExtractor(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                hook_layer=self.cfg.sae.layer,
                device=self.cfg.model.device,
                batch_size=self.cfg.data.batch_size,
            )
            matrices = extractor.extract(
                examples=examples,
                checkpoint_path=matrices_ckpt,
                use_checkpoint=self.cfg.use_checkpoint,
            )
        else:
            logger.info(f"Found cached feature matrices at '{matrices_ckpt}'. Skipping model/SAE loading!")
            extractor = FeatureMatrixExtractor(None, None, None, self.cfg.sae.layer, self.cfg.model.device)
            matrices = extractor.extract(
                examples=examples,
                checkpoint_path=matrices_ckpt,
                use_checkpoint=True,
            )

        # 3. Leiden Feature Clustering
        binary_act = (matrices.C_freq > 0) | (matrices.R_freq > 0)
        clusterer = LeidenFeatureClusterer(
            top_pct=1.0,
            min_community_size=self.cfg.feature_conditioned.min_feat_cluster_size,
        )
        cluster_map = clusterer.cluster(
            binary_activations=binary_act,
            seed=self.cfg.seed,
            checkpoint_path=clusters_ckpt,
            use_checkpoint=self.cfg.use_checkpoint,
        )

        # Write manifest file into checkpoint subfolder
        manifest_data = {
            "name": self.cfg.name,
            "seed": self.cfg.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.cfg.to_dict(),
        }
        manifest_tmp = manifest_ckpt + ".tmp"
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        os.replace(manifest_tmp, manifest_ckpt)

        # 4. Feature-Conditioned Pipeline (Appendix B.1)
        fc_runner = FeatureConditionedPipeline(self.cfg.feature_conditioned)
        fc_res = fc_runner.run(matrices=matrices, cluster_map=cluster_map, seed=self.cfg.seed)

        fc_summary_file = os.path.join(self.cfg.output_dir, "feature_conditioned_hypotheses.json")
        fc_res.save_summary(fc_summary_file)

        # 5. Prompt-Conditioned Pipeline (Appendix B.2)
        pc_runner = PromptConditionedPipeline(self.cfg.prompt_conditioned)
        pc_res = pc_runner.run(matrices=matrices, seed=self.cfg.seed)

        pc_summary_file = os.path.join(self.cfg.output_dir, "prompt_conditioned_hypotheses.json")
        pc_res.save_summary(pc_summary_file)

        # 6. Overall Run Summary Output
        summary_file = os.path.join(self.cfg.output_dir, "pdd_summary.json")
        summary_tmp = summary_file + ".tmp"
        summary_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.cfg.to_dict(),
            "checkpoint_subfolder": run_ckpt_dir,
            "metrics": {
                "num_examples": len(examples),
                "num_sae_feature_clusters": cluster_map.num_clusters,
                "feature_conditioned_hypotheses": len(fc_res.hypotheses),
                "prompt_conditioned_hypotheses": len(pc_res.hypotheses),
            },
            "top_feature_conditioned_hypotheses": [asdict(h) for h in fc_res.hypotheses[:10]],
            "top_prompt_conditioned_hypotheses": [asdict(h) for h in pc_res.hypotheses[:10]],
        }
        with open(summary_tmp, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        os.replace(summary_tmp, summary_file)

        logger.info(f"=== PDD Pipeline Run Complete! Summary saved to '{summary_file}' ===")
        return summary_data

    def _resolve_checkpoint_subfolder(self, timestamp_str: str) -> str:
        """Find existing matching checkpoint subfolder with maximum progress or create new timestamped subfolder."""
        prefix = f"{self.cfg.name}_seed{self.cfg.seed}_"

        if self.cfg.use_checkpoint and os.path.exists(self.cfg.checkpoint_dir):
            matching_subdirs = [
                d for d in os.listdir(self.cfg.checkpoint_dir)
                if (d.startswith(prefix) or d.startswith(f"{self.cfg.name}_"))
                and os.path.isdir(os.path.join(self.cfg.checkpoint_dir, d))
            ]
            if matching_subdirs:
                # Rank matching subfolders by actual progress:
                # 1. Completed matrices.npz (score: 1,000,000,000)
                # 2. Highest last_batch_idx in matrices_partial.npz (score: last_batch_idx + 10)
                # 3. Subfolder with examples.json (score: 1)
                # 4. Fallback to newest timestamp
                matching_subdirs.sort(reverse=True)  # Newest timestamp as tie-breaker
                best_dir = None
                best_score = -1

                import numpy as np

                for d in matching_subdirs:
                    full_path = os.path.join(self.cfg.checkpoint_dir, d)
                    mat_ckpt = os.path.join(full_path, "matrices.npz")
                    part_ckpt = os.path.join(full_path, "matrices_partial.npz")
                    ex_ckpt = os.path.join(full_path, "examples.json")

                    score = 0
                    if os.path.exists(mat_ckpt):
                        try:
                            with np.load(mat_ckpt) as data:
                                score = len(data["example_ids"])
                        except Exception:
                            score = 100
                    elif os.path.exists(part_ckpt):
                        try:
                            with np.load(part_ckpt) as data:
                                if "example_ids" in data:
                                    score = len(data["example_ids"])
                                elif "P_max_shape" in data:
                                    score = int(data["P_max_shape"][0])
                                elif "P_max" in data:
                                    score = len(data["P_max"])
                                else:
                                    score = 10
                        except Exception:
                            score = 5
                    elif os.path.exists(ex_ckpt):
                        score = 1

                    if score > best_score:
                        best_score = score
                        best_dir = full_path

                if best_dir:
                    logger.info(f"Resolved existing checkpoint subfolder with highest progress: '{best_dir}' (progress score: {best_score})")
                    return best_dir

        # Create new timestamped subfolder: checkpoints/{name}_seed{seed}_{timestamp}/
        new_dir = os.path.join(self.cfg.checkpoint_dir, f"{prefix}{timestamp_str}")
        os.makedirs(new_dir, exist_ok=True)
        logger.info(f"Created new timestamped checkpoint subfolder: '{new_dir}'")
        return new_dir
