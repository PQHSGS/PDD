"""Dataset loader for preference datasets with local datasets/ caching."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PreferenceExample:
        return cls(
            example_id=int(data["example_id"]),
            prompt=str(data["prompt"]),
            chosen=str(data["chosen"]),
            rejected=str(data["rejected"]),
        )


class DatasetLoader:
    """Loader for HF preference datasets with local datasets/ directory caching."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def load(
        self,
        checkpoint_path: Optional[str] = None,
        use_checkpoint: bool = True,
    ) -> List[PreferenceExample]:
        """Load preference dataset locally from datasets/ or HF repo with local saving."""
        if use_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
            cached = self.load_json_cache(checkpoint_path)
            if len(cached) > 0:
                logger.info(f"Loaded {len(cached)} cached preference examples from checkpoint: {checkpoint_path}")
                return cached
            else:
                logger.warning(f"Cached checkpoint file '{checkpoint_path}' is empty. Reloading dataset...")


        # 1. Determine local datasets/ target directory
        datasets_root = getattr(self.cfg, "datasets_dir", "datasets")
        if os.path.exists(self.cfg.path) and os.path.isdir(self.cfg.path):
            local_dir = self.cfg.path
        else:
            sanitized_name = self.cfg.path.replace("/", "___")
            local_dir = os.path.join(datasets_root, sanitized_name)

        # 2. Check if dataset exists in local datasets/ folder
        if os.path.exists(local_dir) and os.path.isdir(local_dir) and len(os.listdir(local_dir)) > 0:
            logger.info(f"Found local dataset in '{local_dir}'. Loading directly from disk...")
            try:
                ds_obj = load_from_disk(local_dir)
                if isinstance(ds_obj, DatasetDict):
                    ds = ds_obj[self.cfg.split] if self.cfg.split in ds_obj else ds_obj[list(ds_obj.keys())[0]]
                else:
                    ds = ds_obj
            except Exception as e:
                logger.warning(f"load_from_disk failed for '{local_dir}' ({e}). Falling back to HF load_dataset...")
                ds = self._load_and_save_hf(local_dir)
        else:
            logger.info(f"Dataset not found in local '{local_dir}'. Downloading from HF Hub ('{self.cfg.path}')...")
            ds = self._load_and_save_hf(local_dir)

        if self.cfg.max_samples > 0 and len(ds) > self.cfg.max_samples:
            logger.info(f"Subsampling first {self.cfg.max_samples} dataset examples...")
            ds = ds.select(range(self.cfg.max_samples))

        examples: List[PreferenceExample] = []
        for idx, item in enumerate(tqdm(ds, desc="Processing dataset examples")):
            chosen_raw = item.get(self.cfg.chosen_col, "")
            rejected_raw = item.get(self.cfg.rejected_col, "")
            prompt_raw = item.get(self.cfg.prompt_col, "")

            # Handle list of dialogue turns (e.g. Dolci-Instruct-DPO / UltraFeedback)
            if isinstance(chosen_raw, list) and len(chosen_raw) > 0 and isinstance(chosen_raw[0], dict):
                prompt = chosen_raw[0].get("content", "")
                chosen = chosen_raw[-1].get("content", "")
            else:
                prompt = str(prompt_raw) if prompt_raw else ""
                chosen = str(chosen_raw) if chosen_raw else ""

            if isinstance(rejected_raw, list) and len(rejected_raw) > 0 and isinstance(rejected_raw[0], dict):
                rejected = rejected_raw[-1].get("content", "")
            else:
                rejected = str(rejected_raw) if rejected_raw else ""

            prompt_str = (prompt or "").strip() if isinstance(prompt, str) else str(prompt or "").strip()
            chosen_str = (chosen or "").strip() if isinstance(chosen, str) else str(chosen or "").strip()
            rejected_str = (rejected or "").strip() if isinstance(rejected, str) else str(rejected or "").strip()

            if not prompt_str or not chosen_str or not rejected_str:
                continue

            examples.append(
                PreferenceExample(
                    example_id=idx,
                    prompt=prompt_str,
                    chosen=chosen_str,
                    rejected=rejected_str,
                )
            )



        logger.info(f"Loaded {len(examples)} valid non-empty preference pairs.")

        if checkpoint_path:
            logger.info(f"Caching processed preference examples to {checkpoint_path}...")
            self.save_json_cache(examples, checkpoint_path)

        return examples

    def _load_and_save_hf(self, local_dir: str) -> Dataset:
        """Download dataset from Hugging Face Hub and save local copy into datasets/."""
        ds = load_dataset(self.cfg.path, split=self.cfg.split)
        try:
            logger.info(f"Saving downloaded dataset locally to '{local_dir}'...")
            os.makedirs(local_dir, exist_ok=True)
            ds.save_to_disk(local_dir)
        except Exception as e:
            logger.warning(f"Could not save dataset locally to '{local_dir}': {e}")
        return ds

    @staticmethod
    def save_json_cache(examples: List[PreferenceExample], filepath: str) -> None:
        """Save processed preference examples to disk as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = [ex.to_dict() for ex in examples]
        tmp_filepath = filepath + ".tmp"
        with open(tmp_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_filepath, filepath)

    @staticmethod
    def load_json_cache(filepath: str) -> List[PreferenceExample]:
        """Load cached preference examples from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [PreferenceExample.from_dict(item) for item in data]
