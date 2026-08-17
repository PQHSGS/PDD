"""Auto-labeling module for Spherical K-Means data clusters (Appendix B.1.7)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .data import PreferenceExample
from .logger import get_logger

logger = get_logger("PDD.AutoLabel")


@dataclass
class ClusterLabel:
    cluster_id: int
    title: str
    description: str
    keywords: List[str]
    centroid_prompts: List[str]
    sample_prompts: List[str]


class ClusterAutoLabeler:
    """Label Spherical K-Means data clusters B_k using centroid prompt sampling."""

    def __init__(self, max_prompt_chars: int = 600):
        self.max_prompt_chars = max_prompt_chars

    def sample_cluster_prompts(
        self,
        examples: List[PreferenceExample],
        cluster_assignments: np.ndarray,      # (N,) in 0..K_data
        s_matrix: np.ndarray,                  # (N, K_r)
        cluster_id: int,
        n_centroid: int = 30,
        n_random: int = 20,
        seed: int = 0,
    ) -> Tuple[List[str], List[str]]:
        """Sample 30 centroid-nearest prompts and 20 random prompts for data cluster B_k."""
        member_indices = np.where(cluster_assignments == cluster_id)[0]
        if len(member_indices) == 0:
            return [], []

        s_members = s_matrix[member_indices]
        norms = np.linalg.norm(s_members, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        s_normed = s_members / norms

        # Compute centroid of cluster
        centroid = np.mean(s_normed, axis=0, keepdims=True)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Cosine similarity to centroid
        sims = (s_normed @ centroid.T).squeeze(-1)
        top_centroid_local_idx = np.argsort(sims)[-n_centroid:][::-1]
        centroid_indices = member_indices[top_centroid_local_idx]

        # Random sample
        rng = np.random.RandomState(seed)
        random_indices = rng.choice(member_indices, size=min(n_random, len(member_indices)), replace=False)

        centroid_prompts = [examples[i].prompt[-self.max_prompt_chars:] for i in centroid_indices]
        sample_prompts = [examples[i].prompt[-self.max_prompt_chars:] for i in random_indices]

        return centroid_prompts, sample_prompts

    def generate_label(
        self,
        cluster_id: int,
        centroid_prompts: List[str],
        sample_prompts: List[str],
    ) -> ClusterLabel:
        """Generate heuristic/keyword-based label for cluster B_k."""
        if cluster_id == 0:
            return ClusterLabel(
                cluster_id=0,
                title="Silent Bucket (B_0)",
                description="Examples with low response-side SAE feature activity.",
                keywords=["silent", "low_activation"],
                centroid_prompts=[],
                sample_prompts=[],
            )

        # Extract common words as keywords
        all_text = " ".join(centroid_prompts + sample_prompts).lower()
        words = [w.strip(".,!?;:\"'") for w in all_text.split() if len(w) > 4]

        from collections import Counter
        word_counts = Counter(words)
        top_keywords = [w for w, c in word_counts.most_common(5)]

        title = f"Topic Cluster {cluster_id}: {'/'.join(top_keywords[:2]).capitalize()}"
        description = f"Cluster containing responses associated with {', '.join(top_keywords[:4])}."

        return ClusterLabel(
            cluster_id=cluster_id,
            title=title,
            description=description,
            keywords=top_keywords,
            centroid_prompts=centroid_prompts[:5],
            sample_prompts=sample_prompts[:5],
        )


class LLMClusterLabeler(ClusterAutoLabeler):
    """Generate semantic cluster labels with a small LOCAL instruct model (paper B.1.7).

    The paper assigns each active data cluster a natural-language label via an LLM
    (title/description/keywords). This implementation runs a small model on the local
    GPU (or CPU when VRAM is congested), so no external API budget is consumed.
    Every label is generated from REAL sampled cluster prompts; on any failure it
    falls back to the keyword heuristic rather than fabricating output.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_prompt_chars: int = 600,
        max_examples: int = 10,
        min_vram_gb: float = 3.5,
    ):
        super().__init__(max_prompt_chars=max_prompt_chars)
        self.model_path = model_path
        self.max_examples = max_examples
        self.min_vram_gb = min_vram_gb
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self) -> None:
        """Load the small instruct model, falling back to CPU when VRAM is tight."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is not None:
            return

        device = "cpu"
        dtype = torch.float32
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
                if free_gb >= self.min_vram_gb:
                    device = "cuda"
                    dtype = torch.bfloat16
                else:
                    logger.warning(
                        f"Label model VRAM congested ({free_gb:.2f} GB free). Running labeler on CPU."
                    )
            except Exception:
                pass

        logger.info(f"Loading label model {self.model_path} on {device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map=device
        )
        self._model.eval()
        self._device = device
        logger.info(f"Label model ready on {device}.")

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        text = text.strip().replace("\n", " ")
        return text[:max_chars] if len(text) > max_chars else text

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract the first balanced JSON object, tolerating prose/markdown/truncation around it."""
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        start = text.find("{")
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
            start = text.find("{", start + 1)
        return None

    def _label_dict(self, texts: List[str], kind: str = "prompt") -> Optional[Dict[str, Any]]:
        """Ask the LLM to label a cluster of ``texts``, returning the raw parsed JSON dict.

        ``kind`` switches the instruction between "user prompts" (data clusters B_k)
        and "response texts" (SAE feature clusters T_m). Returns None on any failure
        so the caller can fall back to the keyword heuristic.
        """
        import torch

        if not texts:
            return None
        examples = "\n".join(f"{i}. {self._truncate(t, self.max_prompt_chars)}" for i, t in enumerate(texts, 1))
        if kind == "response":
            subject = "response texts"
            what = "responses"
        else:
            subject = "user prompts"
            what = "prompts"
        instruction = (
            f"You are labeling clusters of {subject} sampled from an RLHF preference dataset. "
            f"Each cluster groups similar {what}. Here are representative {what} from one cluster:\n\n"
            f"{examples}\n\n"
            'Reply with a single JSON object: {"title": "...", "description": "...", "keywords": ["...", "..."]}.\n'
            'Do NOT reason. Do NOT explain. Output ONLY the JSON object, nothing else.\n\n'
            'Example of the exact required format:\n'
            '{"title": "Gardening Tips", "description": "Prompts asking about pruning, trimming and plant care.", "keywords": ["pruning", "gardening", "plants"]}'
        )
        messages = [{"role": "user", "content": instruction}]
        try:
            inputs = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False}, return_tensors="pt",
            ).to(self._device)
        except Exception:
            inputs = self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_ids = outputs[0][inputs.shape[1]:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self._extract_json(text)

    def generate_label(
        self,
        cluster_id: int,
        centroid_prompts: List[str],
        sample_prompts: List[str],
    ) -> ClusterLabel:
        """Generate a semantic label for data cluster B_k from REAL sampled prompts."""
        # B_0 is always the silent (low-activity) bucket; never send it to the LLM.
        if cluster_id == 0:
            return super().generate_label(cluster_id, centroid_prompts, sample_prompts)

        fallback = super().generate_label(cluster_id, centroid_prompts, sample_prompts)
        all_prompts = centroid_prompts[: self.max_examples] or sample_prompts[: self.max_examples]

        try:
            self.load()
            parsed = self._label_dict(all_prompts, kind="prompt")
        except Exception as e:
            logger.warning(f"LLM labeling failed for B_{cluster_id} ({e}); using keyword fallback.")
            return fallback

        if parsed is None:
            logger.warning(f"LLM label for B_{cluster_id} was not valid JSON; using keyword fallback.")
            return fallback

        title = str(parsed.get("title", "")).strip() or fallback.title
        desc = str(parsed.get("description", "")).strip() or fallback.description
        kws = parsed.get("keywords", [])
        if isinstance(kws, str):
            kws = [kws]
        kws = [str(k).strip() for k in kws if str(k).strip()][:5] or fallback.keywords

        return ClusterLabel(
            cluster_id=cluster_id,
            title=title[:120],
            description=desc[:600],
            keywords=kws,
            centroid_prompts=centroid_prompts[:5],
            sample_prompts=sample_prompts[:5],
        )
