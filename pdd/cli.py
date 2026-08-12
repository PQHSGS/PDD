"""CLI Entrypoint for Predictive Data Debugging (PDD).

Usage:
  python -m pdd.cli --config configs/qwen3_1.7b_base.json
  python -m pdd.cli --config configs/gemma2_2b_base.json --force_rerun
"""
from __future__ import annotations

import argparse
import sys

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


    pipeline = PDDPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
