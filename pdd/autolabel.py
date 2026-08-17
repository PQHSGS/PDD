"""Auto-labeling and cluster interpretation module (Appendix B.1.7 + viewer interpretation).

Contains:
- ClusterLabel dataclass
- ClusterAutoLabeler (keyword-heuristic labeler)
- LLMClusterLabeler (local instruct model labeler with dense prompts and clean title/desc filters)
- AutoLabelingPipeline (3-pass pipeline stage generating B_k, T_m, and A_k/R_m artifacts)
- Shared artifact path helpers for the viewer (cluster_labels_path, feature_cluster_labels_path, pc_cluster_examples_path)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

from .config import AutoLabelConfig, FeatureConditionedConfig, PromptConditionedConfig
from .data import DatasetLoader, PreferenceExample
from .logger import get_logger

logger = get_logger("PDD.AutoLabel")


# ---------------------------------------------------------------------------
# Artifact Path Helpers (Shared single source of truth with viewer)
# ---------------------------------------------------------------------------

def cluster_labels_path(run_dir: Union[str, "os.PathLike[str]"]) -> str:
    """LLM labels for data clusters B_k (title/description/keywords)."""
    return os.path.join(run_dir, "cluster_labels.json")


def feature_cluster_labels_path(run_dir: Union[str, "os.PathLike[str]"]) -> str:
    """Whole-cluster LLM labels for SAE feature clusters T_m."""
    return os.path.join(run_dir, "feature_cluster_labels.json")


def pc_cluster_examples_path(run_dir: Union[str, "os.PathLike[str]"]) -> str:
    """Real example indices expressing prompt clusters A_k / response-delta clusters R_m."""
    return os.path.join(run_dir, "prompt_conditioned_cluster_examples.json")


def _save_json(path: str, data: Dict[str, Any]) -> None:
    """Atomic JSON write (tmp + replace) so a crash never corrupts an artifact."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)
    logger.info(f"Saved '{path}'.")


def _load_examples(checkpoint_dir: str) -> List[Any]:
    """Load the REAL cached preference examples using fast compiled C parser."""
    ex_path = os.path.join(checkpoint_dir, "examples.json")
    if not os.path.exists(ex_path):
        raise FileNotFoundError(f"No cached examples.json in checkpoint '{checkpoint_dir}' for auto-labeling.")
    return DatasetLoader.load_json_cache(ex_path)


_STOPWORDS = frozenset({
    "about", "after", "again", "against", "almost", "along", "already", "also",
    "although", "always", "among", "another", "answer", "anyone", "anything",
    "around", "because", "before", "behind", "being", "below", "between", "beyond",
    "cannot", "could", "during", "either", "enough", "every", "except", "explain",
    "first", "following", "further", "given", "giving", "having", "instead",
    "itself", "little", "mainly", "making", "many", "might", "myself", "neither",
    "never", "nothing", "number", "often", "other", "others", "please", "provide",
    "rather", "really", "regarding", "several", "should", "since", "someone",
    "something", "sometimes", "still", "their", "theirs", "them", "themselves",
    "there", "these", "things", "those", "though", "through", "together", "toward",
    "under", "until", "using", "various", "welcome", "where", "whether", "which",
    "while", "whose", "within", "without", "would", "write", "writing", "yours",
    "assistant", "human", "prompt", "response", "chosen", "rejected",
})


def _top_tokens(texts: Sequence[str], weights: Sequence[float], n_tokens: int = 8) -> List[str]:
    """Score-weighted top content tokens across a cluster's strongest examples.

    Higher-weight examples (stronger c_matrix / |u_matrix|) contribute more to a
    token's total, so the returned tokens are the ones that most express the
    cluster in the real data. Pure offline text statistics — no model calls.
    """
    counts: Dict[str, float] = {}
    for text, w in zip(texts, weights):
        if not text:
            continue
        for tok in re.split(r"[^\w']+", text.lower()):
            tok = tok.strip("'")
            if len(tok) > 4 and tok.isalpha() and tok not in _STOPWORDS:
                counts[tok] = counts.get(tok, 0.0) + w
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:n_tokens]]


# ---------------------------------------------------------------------------
# Data Structures & Labelers
# ---------------------------------------------------------------------------

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
        words = [w.strip(".,!?;:\"'") for w in all_text.split() if len(w) > 4 and w not in _STOPWORDS]

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

    @staticmethod
    def _clean_title(title: str, fallback: str) -> str:
        """Strip meta-echoes and verbose prefixes from generated titles."""
        t = title.strip().strip('"\'')
        for prefix in ("cluster of response texts on", "cluster of response texts", "cluster of user prompts on",
                       "cluster of user prompts", "cluster of", "cluster on", "labeling clusters of",
                       "labeling cluster of", "user_prompts", "response_texts"):
            if t.lower().startswith(prefix):
                t = t[len(prefix):].strip(" :-,")
        return t.capitalize()[:120] if len(t) >= 3 else fallback

    @staticmethod
    def _clean_desc(desc: str, fallback: str) -> str:
        """Strip filler prefixes from generated descriptions."""
        d = desc.strip().strip('"\'')
        for prefix in ("cluster of response texts discussing", "cluster of response texts related to",
                       "cluster of response texts on", "cluster of responses related to",
                       "cluster of responses discussing", "cluster of user prompts asking about",
                       "cluster of user prompts related to", "cluster responses based on",
                       "cluster containing responses associated with", "cluster of"):
            if d.lower().startswith(prefix):
                d = d[len(prefix):].strip(" :-,")
        return d.capitalize()[:600] if len(d) >= 5 else fallback

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
        what = "responses" if kind == "response" else "prompts"
        instruction = (
            f"Summarize the common theme of these representative {what} from an RLHF preference dataset into a dense semantic concept label.\n\n"
            f"{examples}\n\n"
            "Rules:\n"
            "- 'title': 2-5 words naming the specific topic/task. NEVER include meta words like 'Cluster', 'User Prompts', 'Response', or 'Labeling'.\n"
            "- 'description': 1 concise sentence describing the core topic directly without filler phrases.\n"
            "- 'keywords': 3-5 specific domain keywords (lowercase).\n\n"
            'Output ONLY a valid JSON object: {"title": "...", "description": "...", "keywords": ["...", "..."]}'
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

        title = self._clean_title(str(parsed.get("title", "")), fallback.title)
        desc = self._clean_desc(str(parsed.get("description", "")), fallback.description)
        kws = parsed.get("keywords", [])
        if isinstance(kws, str):
            kws = [kws]
        kws = [str(k).strip().lower() for k in kws if str(k).strip()][:5] or fallback.keywords

        return ClusterLabel(
            cluster_id=cluster_id,
            title=title[:120],
            description=desc[:600],
            keywords=kws,
            centroid_prompts=centroid_prompts[:5],
            sample_prompts=sample_prompts[:5],
        )


# ---------------------------------------------------------------------------
# Auto-Labeling Pipeline Orchestrator (Pass 1, Pass 2, Pass 3)
# ---------------------------------------------------------------------------

class AutoLabelingPipeline:
    """Runs the three auto-interpretation passes and writes them under the run directory."""

    def __init__(self, cfg: AutoLabelConfig, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir

    def _labeler(self):
        """Build the LLM or keyword labeler."""
        if self.cfg.heuristic:
            logger.info("Using keyword-heuristic labels (auto_label.heuristic).")
            return ClusterAutoLabeler(max_prompt_chars=self.cfg.max_prompt_chars)
        logger.info(f"Using local LLM labels ({self.cfg.label_model}).")
        return LLMClusterLabeler(
            model_path=self.cfg.label_model,
            max_prompt_chars=self.cfg.max_prompt_chars,
            max_examples=self.cfg.max_examples,
        )

    def _label_data_clusters(
        self,
        labeler: Any,
        examples: List[Any],
        fc_res: Any,
        seed: int,
    ) -> int:
        """Pass 1: LLM (or keyword) labels for data clusters B_k."""
        unique_clusters = sorted(set(fc_res.cluster_assignments.tolist()))
        if self.cfg.num_clusters > 0:
            unique_clusters = unique_clusters[: self.cfg.num_clusters]

        out_path = cluster_labels_path(self.run_dir)
        labels: List[Dict[str, Any]] = []
        for idx, k in enumerate(tqdm(unique_clusters, desc="Pass 1: labeling data clusters", unit="cluster")):
            centroid_p, sample_p = labeler.sample_cluster_prompts(
                examples=examples,
                cluster_assignments=fc_res.cluster_assignments,
                s_matrix=fc_res.s_matrix,
                cluster_id=k,
                seed=seed,
            )
            labels.append(asdict(labeler.generate_label(k, centroid_p, sample_p)))
            if (idx + 1) % 10 == 0:
                _save_json(out_path, {"total_clusters": len(labels), "labels": labels})
                logger.info(f"  ...{idx + 1}/{len(unique_clusters)} labeled (checkpoint saved).")

        _save_json(out_path, {"total_clusters": len(labels), "labels": labels})
        return len(labels)

    def _label_feature_clusters(
        self,
        labeler: Any,
        examples: List[Any],
        matrices: Any,
        clusters: Dict[int, Sequence[int]],
    ) -> int:
        """Pass 2: whole-cluster labels for SAE feature clusters T_m."""
        d_sae = matrices.C_max.shape[1]
        out_path = feature_cluster_labels_path(self.run_dir)
        labels: Dict[str, Dict[str, Any]] = {}
        for m in tqdm(sorted(clusters.keys()), desc="Pass 2: labeling feature clusters", unit="cluster"):
            feats = [f for f in clusters[m] if 0 <= f < d_sae]
            if not feats:
                continue
            firing = matrices.C_max[:, feats] + matrices.R_max[:, feats]
            scores = np.asarray(firing.sum(axis=1)).ravel()

            # Dynamic sample scaling: 10 to 20 representative firing examples based on cluster size
            n_samples = min(20, max(10, len(feats) // 2))
            idxs = [int(i) for i in np.argsort(scores)[-n_samples:][::-1] if scores[int(i)] > 0 and int(i) < len(examples)]
            texts = [(examples[i].chosen or examples[i].rejected or examples[i].prompt or "").strip() for i in idxs]

            fallback = labeler.generate_label(m, texts, []) if hasattr(labeler, "generate_label") else None
            fallback_title = fallback.title if fallback else f"Feature cluster T_{m}"
            fallback_desc = fallback.description if fallback else "Cluster of SAE features"

            if isinstance(labeler, LLMClusterLabeler):
                parsed = labeler._label_dict(texts, kind="response")
                if parsed is None:
                    title, desc, kws = fallback_title, fallback_desc, []
                else:
                    raw_title = str(parsed.get("title", ""))
                    raw_desc = str(parsed.get("description", ""))
                    title = labeler._clean_title(raw_title, fallback_title)
                    desc = labeler._clean_desc(raw_desc, fallback_desc)
                    kws = parsed.get("keywords", [])
            else:
                title, desc, kws = fallback_title, fallback_desc, fallback.keywords if fallback else []

            if isinstance(kws, str):
                kws = [kws]
            labels[str(m)] = {
                "title": str(title)[:120],
                "description": str(desc)[:600],
                "keywords": [str(k).strip().lower() for k in kws if str(k).strip()][:5],
            }
            logger.info(f"T_{m}: {labels[str(m)]['title']}")

        payload: Dict[str, Any] = {"feature_clusters": labels}
        _save_json(out_path, payload)
        return len(labels)

    def _pc_cluster_examples(self, pc_res: Any, examples: List[Any], n_top: int) -> int:
        """Pass 3: real-example indices + top tokens for prompt clusters A_k and response-delta clusters R_m."""
        prompt_ex: Dict[str, List[int]] = {}
        prompt_tokens: Dict[str, List[str]] = {}
        n_ex = len(examples)
        for col, k in enumerate(sorted(pc_res.prompt_clusters.keys())):
            scores = np.asarray(pc_res.c_matrix[:, col]).ravel()
            idxs = [int(i) for i in np.argsort(scores)[-n_top:][::-1] if int(i) < n_ex]
            prompt_ex[str(k)] = idxs
            prompt_tokens[str(k)] = _top_tokens(
                [examples[i].prompt or "" for i in idxs], [float(scores[i]) for i in idxs]
            )

        resp_ex: Dict[str, List[int]] = {}
        resp_tokens: Dict[str, List[str]] = {}
        for col, m in enumerate(sorted(pc_res.resp_clusters.keys())):
            scores = np.abs(np.asarray(pc_res.u_matrix[:, col]).ravel())
            idxs = [int(i) for i in np.argsort(scores)[-n_top:][::-1] if int(i) < n_ex]
            resp_ex[str(m)] = idxs
            resp_tokens[str(m)] = _top_tokens(
                [examples[i].chosen or "" for i in idxs], [float(scores[i]) for i in idxs]
            )

        _save_json(pc_cluster_examples_path(self.run_dir), {
            "n_top": n_top,
            "prompt_cluster_examples": prompt_ex,
            "response_cluster_examples": resp_ex,
            "prompt_cluster_tokens": prompt_tokens,
            "response_cluster_tokens": resp_tokens,
        })
        return len(prompt_ex) + len(resp_ex)

    def run(
        self,
        matrices: Any,
        cluster_map: Any,
        fc_res: Optional[Any] = None,
        pc_res: Optional[Any] = None,
        seed: int = 0,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, int]:
        """Execute all three passes."""
        from .feature_conditioned import FeatureConditionedPipeline
        from .prompt_conditioned import PromptConditionedPipeline

        if checkpoint_dir is None:
            raise ValueError("AutoLabelingPipeline.run requires checkpoint_dir to load the real cached examples.")
        examples = _load_examples(checkpoint_dir)
        labeler = self._labeler()

        counts: Dict[str, int] = {}

        if fc_res is None:
            logger.info("Re-running B.1 feature-conditioned pipeline for auto-labeling (no precomputed result passed).")
            fc_res = FeatureConditionedPipeline(FeatureConditionedConfig()).run(matrices, cluster_map, seed=seed)
        counts["data_clusters"] = self._label_data_clusters(labeler, examples, fc_res, seed=seed)

        if not self.cfg.skip_feature_clusters:
            counts["feature_clusters"] = self._label_feature_clusters(
                labeler, examples, matrices, cluster_map.clusters
            )

        if not self.cfg.skip_pc_examples:
            if pc_res is None:
                logger.info("Re-running B.2 prompt-conditioned pipeline for auto-labeling (no precomputed result passed).")
                pc_cfg = PromptConditionedConfig()
                pc_res = PromptConditionedPipeline(pc_cfg).run(matrices, seed=seed, checkpoint_dir=checkpoint_dir)
            counts["pc_clusters"] = self._pc_cluster_examples(pc_res, examples, self.cfg.pc_n_top)

        return counts
