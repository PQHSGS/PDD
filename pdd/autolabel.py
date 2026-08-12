"""Auto-labeling module for Spherical K-Means data clusters (Appendix B.1.7)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

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
