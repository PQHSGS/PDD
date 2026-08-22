"""Dataset loader for preference datasets with local datasets/ caching."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, asdict
import json
import os
import threading
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import numpy as np
from tqdm import tqdm
from typing import Any, List, Optional

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
        """Save processed preference examples to disk as JSON (streamed record-by-record).

        Writing incrementally avoids materializing a second full dict-list copy of the
        dataset in RAM (the old `[ex.to_dict() for ex in examples]` doubled peak usage).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_filepath = filepath + ".tmp"
        with open(tmp_filepath, "w", encoding="utf-8") as f:
            f.write("[")
            for i, ex in enumerate(examples):
                if i:
                    f.write(",")
                json.dump(ex.to_dict(), f, separators=(",", ":"))
            f.write("]")
        os.replace(tmp_filepath, filepath)

    @staticmethod
    def load_json_cache(filepath: str) -> List[PreferenceExample]:
        """Load cached preference examples from JSON using fast C parser if available.

        Converts record-by-record and releases each parsed dict immediately, so the
        parsed dict list and the PreferenceExample objects never fully coexist in RAM
        (matters for ~260k-example checkpoints on memory-pressured hosts).
        """
        try:
            import orjson
            with open(filepath, "rb") as f:
                data = orjson.loads(f.read())
        except ImportError:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        out: List[PreferenceExample] = []
        for i in range(len(data)):
            out.append(PreferenceExample.from_dict(data[i]))
            data[i] = None
        return out


class LazyExampleStore:
    """Random-access PreferenceExample reader backed by an NDJSON sidecar.

    Builds `<ckpt>/examples.ndjson` (one compact JSON object per line) plus a
    cumulative byte-offset index `<ckpt>/examples_offsets.npy` from the canonical
    `examples.json` exactly ONCE; afterwards every record is served via seek+parse
    with a small LRU cache, so a viewer holds ~2 MB of offsets instead of ~5 GB of
    parsed objects on a 260k-example run.

    Sequence-compatible: consumers use len() and integer indexing only.
    Thread-safe: one shared read handle guarded by a lock (viewer serves
    concurrent requests).
    """

    LRU_MAX = 256

    def __init__(self, ndjson_path: str, offsets_path: str):
        self._path = ndjson_path
        self._offsets = np.load(offsets_path)
        self._cache: "OrderedDict[int, PreferenceExample]" = OrderedDict()
        self._lock = threading.Lock()
        self._fh = open(ndjson_path, "rb")

    @classmethod
    def build_if_missing(cls, examples_json: str) -> Optional["LazyExampleStore"]:
        """Build sidecars from examples.json once; return store or None if no source.

        The one-time build parses the full array (same cost as a full load) and
        streams it out as NDJSON while recording byte offsets. Corrupt or truncated
        sidecars are rebuilt automatically on the next call.
        """
        if not os.path.exists(examples_json):
            return None
        base = os.path.splitext(examples_json)[0]
        ndjson_path = base + ".ndjson"
        offsets_path = base + "_offsets.npy"

        try:
            store = cls(ndjson_path, offsets_path)
            expected_size = int(store._offsets[-1])
            if len(store._offsets) >= 1 and os.path.getsize(ndjson_path) == expected_size:
                return store
            store.close()
        except Exception:
            pass  # fall through to rebuild below

        # Full parse once (same cost as a full load), then stream-write NDJSON.
        records = cls._load_raw(examples_json)

        def _dumps(item: Any) -> bytes:
            try:
                import orjson
                return orjson.dumps(item) + b"\n"
            except ImportError:
                return (json.dumps(item, separators=(",", ":")) + "\n").encode("utf-8")

        offsets = [0]
        tmp_nd = ndjson_path + f".{os.getpid()}.tmp"
        with open(tmp_nd, "wb") as f:
            for item in records:
                line = _dumps(item)
                f.write(line)
                offsets.append(offsets[-1] + len(line))
        os.replace(tmp_nd, ndjson_path)

        tmp_off = offsets_path + f".{os.getpid()}.tmp"
        with open(tmp_off, "wb") as f:
            np.save(f, np.asarray(offsets, dtype=np.int64))
        os.replace(tmp_off, offsets_path)
        return cls(ndjson_path, offsets_path)

    @staticmethod
    def _load_raw(examples_json: str) -> List[Any]:
        try:
            import orjson
            with open(examples_json, "rb") as f:
                return orjson.loads(f.read())
        except ImportError:
            with open(examples_json, "r", encoding="utf-8") as f:
                return json.load(f)

    # -- Sequence protocol -------------------------------------------------

    def __len__(self) -> int:
        return max(0, len(self._offsets) - 1)

    def __getitem__(self, i: int) -> PreferenceExample:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        n = len(self)
        idx = i + n if i < 0 else i
        if not 0 <= idx < n:
            raise IndexError(f"example index {i} out of range [0, {n})")

        with self._lock:
            hit = self._cache.get(idx)
            if hit is not None:
                self._cache.move_to_end(idx)
                return hit
            start = int(self._offsets[idx])
            end = int(self._offsets[idx + 1])
            self._fh.seek(start)
            raw = self._fh.read(end - start)
        try:
            import orjson
            record = orjson.loads(raw)
        except ImportError:
            record = json.loads(raw.decode("utf-8"))
        ex = PreferenceExample.from_dict(record)
        with self._lock:
            self._cache[idx] = ex
            if len(self._cache) > self.LRU_MAX:
                self._cache.popitem(last=False)
        return ex

    def close(self) -> None:
        """Release the underlying file handle."""
        try:
            self._fh.close()
        except Exception:
            pass
