"""Dataset loader for preference datasets with validation."""
from __future__ import annotations

from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Optional

from .config import DataConfig
from .logger import get_logger

logger = get_logger("PDD.Data")


@dataclass
class PreferenceExample:
    example_id: int
    prompt: str
    chosen: str
    rejected: str


class DatasetLoader:
    """Loader for HF preference datasets (e.g. Dolci-Instruct-DPO)."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def load(self) -> List[PreferenceExample]:
        """Load and filter preference dataset."""
        logger.info(f"Loading dataset from '{self.cfg.path}' (split='{self.cfg.split}')...")
        ds = load_dataset(self.cfg.path, split=self.cfg.split)

        if self.cfg.max_samples > 0 and len(ds) > self.cfg.max_samples:
            logger.info(f"Subsampling first {self.cfg.max_samples} dataset examples...")
            ds = ds.select(range(self.cfg.max_samples))

        examples: List[PreferenceExample] = []
        for idx, item in enumerate(tqdm(ds, desc="Processing dataset examples")):
            prompt = item.get(self.cfg.prompt_col, "")
            chosen = item.get(self.cfg.chosen_col, "")
            rejected = item.get(self.cfg.rejected_col, "")

            if isinstance(prompt, list):
                prompt = str(prompt)
            if isinstance(chosen, list):
                chosen = str(chosen)
            if isinstance(rejected, list):
                rejected = str(rejected)

            if not prompt.strip() or not chosen.strip() or not rejected.strip():
                continue

            examples.append(
                PreferenceExample(
                    example_id=idx,
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                )
            )

        logger.info(f"Loaded {len(examples)} valid non-empty preference pairs.")
        return examples
