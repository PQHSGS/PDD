"""CLI Entrypoint for Predictive Data Debugging (PDD).

Usage:
  python -m pdd.cli --config configs/qwen3_1.7b_base.json
  python -m pdd.cli --config configs/gemma2_2b_base.json --force_rerun
"""
from __future__ import annotations

import argparse
import os
import sys

# Disable HuggingFace tokenizer background worker forks to prevent RAM memory leaks (pt_data_worker OOM)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Prevent CUDA memory fragmentation for tight GPU VRAM environments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Force line-buffered unbuffered stdout so tqdm and logs stream immediately to tmux / log files
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

from .config import PipelineConfig
from .logger import get_logger
from .pipeline import PDDPipeline

logger = get_logger("PDD.CLI")


def main():
    parser = argparse.ArgumentParser(description="Predictive Data Debugging (PDD) Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3_1.7b_base.json",
        help="Path to JSON configuration file (e.g., configs/qwen3_1.7b_base.json)",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Bypass cached checkpoints and force fresh computation")
    parser.add_argument("--device", type=str, default=None, help="Override execution device ('cuda' or 'cpu')")
    parser.add_argument("--batch_size", type=int, default=None, help="Override extraction batch size")
    parser.add_argument("--save_every_batches", type=int, default=None, help="Override checkpoint chunk save frequency in batches")
    parser.add_argument("--sae_cpu", action="store_true", help="Run SAE encoding on CPU instead of CUDA")

    args = parser.parse_args()

    logger.info(f"Loading PDD configuration from '{args.config}'...")
    try:
        config = PipelineConfig.load_json(args.config)
    except Exception as e:
        logger.error(f"Failed to load config '{args.config}': {e}")
        sys.exit(1)

    if args.force_rerun:
        config.use_checkpoint = False
    if args.device:
        config.model.device = args.device
        config.sae.device = args.device
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.save_every_batches:
        config.data.save_every_batches = args.save_every_batches
    if args.sae_cpu:
        config.sae.sae_cpu = True


    pipeline = PDDPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
