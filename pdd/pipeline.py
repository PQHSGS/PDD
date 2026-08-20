"""Master Orchestrator Pipeline for Predictive Data Debugging with timestamped checkpoint subfolders."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
import time
import numpy as np
from typing import Any, Dict

from .config import PipelineConfig
from .data import DatasetLoader, PreferenceExample
from .feature_clusters import LeidenFeatureClusterer
from .feature_conditioned import FeatureConditionedPipeline
from .feature_matrices import FeatureMatrixExtractor, mmap_dir_complete
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
        matrices_mmap_dir = os.path.join(run_ckpt_dir, "matrices_mmap")
        clusters_ckpt = os.path.join(run_ckpt_dir, "clusters.json")
        manifest_ckpt = os.path.join(run_ckpt_dir, "manifest.json")

        # Write manifest file immediately into checkpoint subfolder
        manifest_data = {
            "name": self.cfg.name,
            "seed": self.cfg.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.cfg.to_dict(),
        }
        try:
            tmp_path = manifest_ckpt + f".{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            os.replace(tmp_path, manifest_ckpt)
        except Exception as e:
            logger.warning(f"Could not write manifest file '{manifest_ckpt}': {e}")

        # 2. Checkpoint check for feature matrices
        needs_model_load = not (self.cfg.use_checkpoint and (os.path.exists(matrices_ckpt) or (os.path.isdir(matrices_mmap_dir) and mmap_dir_complete(matrices_mmap_dir))))

        # 1. Dataset Loading (bypassed in 0.001s if cached feature matrices exist)
        if not needs_model_load and os.path.exists(os.path.join(matrices_mmap_dir, "example_ids.npy")):
            logger.info("Cached feature matrices found. Instantiating lightweight dataset metadata in 0.001s!")
            ex_ids = np.load(os.path.join(matrices_mmap_dir, "example_ids.npy"))
            examples = [PreferenceExample(int(idx), "", "", "") for idx in ex_ids]
        else:
            data_loader = DatasetLoader(self.cfg.data)
            examples = data_loader.load(
                checkpoint_path=examples_ckpt,
                use_checkpoint=self.cfg.use_checkpoint,
            )

        # Flush temporary Arrow / dataset parsing buffers from system RAM
        import gc
        gc.collect()

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
                save_every_batches=getattr(self.cfg.data, "save_every_batches", 100),
            )
            matrices = extractor.extract(
                examples=examples,
                checkpoint_path=matrices_ckpt,
                use_checkpoint=self.cfg.use_checkpoint,
            )
        else:
            logger.info(f"Found cached feature matrices at '{matrices_ckpt}'. Skipping model/SAE loading!")
            extractor = FeatureMatrixExtractor(None, None, None, self.cfg.sae.layer, self.cfg.model.device, save_every_batches=getattr(self.cfg.data, "save_every_batches", 100))
            matrices = extractor.extract(
                examples=examples,
                checkpoint_path=matrices_ckpt,
                use_checkpoint=True,
            )

        # 3. Leiden Feature Clustering
        clusterer = LeidenFeatureClusterer(
            min_community_size=self.cfg.feature_clusters.min_community_size,
            top_pct=self.cfg.feature_clusters.top_pct,
            min_firing_freq=self.cfg.feature_clusters.min_firing_freq,
            block_size=self.cfg.feature_clusters.block_size,
            resolution_parameter=self.cfg.feature_clusters.resolution_parameter,
        )
        cluster_map = clusterer.cluster(
            matrices=matrices,
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
        manifest_tmp = manifest_ckpt + f".{os.getpid()}.tmp"
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        os.replace(manifest_tmp, manifest_ckpt)

        # 4. Feature-Conditioned Pipeline (Appendix B.1)
        fc_runner = FeatureConditionedPipeline(self.cfg.feature_conditioned)
        fc_res = fc_runner.run(
            matrices=matrices,
            cluster_map=cluster_map,
            seed=self.cfg.seed,
            checkpoint_dir=run_ckpt_dir,
            use_checkpoint=self.cfg.use_checkpoint,
        )

        fc_summary_file = os.path.join(self.cfg.output_dir, "feature_conditioned_hypotheses.json")
        fc_res.save_summary(fc_summary_file)

        # 5. Prompt-Conditioned Pipeline (Appendix B.2)
        pc_runner = PromptConditionedPipeline(self.cfg.prompt_conditioned)
        pc_res = pc_runner.run(
            matrices=matrices,
            seed=self.cfg.seed,
            checkpoint_dir=run_ckpt_dir,
            use_checkpoint=self.cfg.use_checkpoint,
        )

        pc_summary_file = os.path.join(self.cfg.output_dir, "prompt_conditioned_hypotheses.json")
        pc_res.save_summary(pc_summary_file)

        # 6. Auto-Interpretation Stage (B.1.7 labels + A_k/R_m example indices for the viewer)
        if self.cfg.auto_label.enabled:
            from .autolabel import AutoLabelingPipeline

            label_counts = AutoLabelingPipeline(self.cfg.auto_label, self.cfg.output_dir).run(
                matrices=matrices,
                cluster_map=cluster_map,
                fc_res=fc_res,
                pc_res=pc_res,
                seed=self.cfg.seed,
                checkpoint_dir=run_ckpt_dir,
            )
            logger.info(f"Auto-labeling stage complete: {label_counts}.")

        # 7. Overall Run Summary Output
        summary_file = os.path.join(self.cfg.output_dir, "pdd_summary.json")
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
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"=== PDD Pipeline Run Complete! Summary saved to '{summary_file}' ===")
        return summary_data

    @staticmethod
    def _completed_matrices_score(mat_ckpt: str, mat_mmap_dir: str) -> int:
        """Progress score (N + 1M) for a subfolder with completed matrices (npz or complete mmap dir)."""
        import numpy as np
        try:
            if os.path.exists(mat_ckpt):
                with np.load(mat_ckpt, mmap_mode="r") as data:
                    if "P_max_shape" in data:
                        return int(data["P_max_shape"][0]) + 1_000_000
                    return len(data["example_ids"]) + 1_000_000
            shp = np.load(os.path.join(mat_mmap_dir, "P_max_shape.npy"))
            return int(shp[0]) + 1_000_000
        except Exception as e:
            logger.warning(f"Could not read matrix checkpoint progress ({e}); assuming full completion.")
            return 1_000_000

    def _resolve_checkpoint_subfolder(self, timestamp_str: str) -> str:
        """Find existing matching checkpoint subfolder with maximum progress or create new timestamped subfolder."""
        import numpy as np
        prefix = f"{self.cfg.name}_seed{self.cfg.seed}_"

        if self.cfg.use_checkpoint and os.path.exists(self.cfg.checkpoint_dir):
            matching_subdirs = [
                d for d in os.listdir(self.cfg.checkpoint_dir)
                if (d.startswith(prefix) or d.startswith(f"{self.cfg.name}_"))
                and os.path.isdir(os.path.join(self.cfg.checkpoint_dir, d))
            ]
            if matching_subdirs:
                # Rank matching subfolders by actual progress:
                # 1. Completed matrices.npz / complete matrices_mmap dir (score: 1,000,000,000)
                # 2. Surviving chunks dir (score: sum of chunk sizes)
                # 3. Subfolder with examples.json (score: 1)
                # 4. Fallback to newest timestamp
                matching_subdirs.sort(reverse=True)  # Newest timestamp as tie-breaker
                best_dir = None
                best_score = -1

                for d in matching_subdirs:
                    full_path = os.path.join(self.cfg.checkpoint_dir, d)
                    mat_ckpt = os.path.join(full_path, "matrices.npz")
                    mat_mmap_dir = os.path.join(full_path, "matrices_mmap")
                    chunks_dir = os.path.join(full_path, "chunks")
                    ex_ckpt = os.path.join(full_path, "examples.json")
                    score = 0
                    if os.path.exists(mat_ckpt):
                        score = self._completed_matrices_score(mat_ckpt, mat_mmap_dir)
                    elif os.path.isdir(mat_mmap_dir) and mmap_dir_complete(mat_mmap_dir):
                        # Disk-backed consolidation: matrices live in the mmap dir.
                        score = self._completed_matrices_score(mat_ckpt, mat_mmap_dir)
                    elif os.path.isdir(mat_mmap_dir):
                        # Partial mmap dir (merge crashed before writing shape files):
                        # treat as no progress so surviving chunks drive the resume.
                        score = 0
                    elif os.path.exists(chunks_dir):
                        try:
                            c_files = sorted([f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".npz")])
                            for cf in c_files:
                                cf_path = os.path.join(chunks_dir, cf)
                                try:
                                    with np.load(cf_path, mmap_mode="r") as d_hdr:
                                        if "P_max_shape" in d_hdr:
                                            score += int(d_hdr["P_max_shape"][0])
                                        elif "example_ids" in d_hdr:
                                            score += len(d_hdr["example_ids"])
                                except Exception as e:
                                    logger.warning(f"Error reading header of '{cf_path}': {e}")
                        except Exception as e:
                            logger.warning(f"Error listing chunks dir '{chunks_dir}': {e}")
                    elif os.path.exists(ex_ckpt):
                        score = 1

                    if score > best_score:
                        best_score = score
                        best_dir = os.path.abspath(full_path)

                if best_dir:
                    logger.info(f"Resolved existing checkpoint subfolder with highest progress: '{best_dir}' (progress score: {best_score})")
                    return best_dir

        # Create new timestamped subfolder: checkpoints/{name}_seed{seed}_{timestamp}/
        new_dir = os.path.join(self.cfg.checkpoint_dir, f"{prefix}{timestamp_str}")
        os.makedirs(new_dir, exist_ok=True)
        logger.info(f"Created new timestamped checkpoint subfolder: '{new_dir}'")
        return new_dir
