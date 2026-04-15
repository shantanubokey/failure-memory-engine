"""
Failure Clustering & Rule Synthesis (Component 4)
UMAP dimensionality reduction → HDBSCAN clustering → LLM rule synthesis
Runs offline every N=100 new failures or daily.
"""

import numpy as np
from typing import List, Dict, Optional
from core.schema import FailureCase, RuleRecord
import uuid
from datetime import datetime


class ClusterEngine:
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        self.min_cluster_size = min_cluster_size
        self.min_samples      = min_samples

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """UMAP: 1536-dim → 2-dim for visualization and clustering."""
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                random_state=42,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("[Cluster] umap-learn not installed — using PCA fallback")
            from sklearn.decomposition import PCA
            return PCA(n_components=2).fit_transform(embeddings)

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """HDBSCAN clustering. Returns cluster labels (-1 = noise)."""
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
            )
            return clusterer.fit_predict(embeddings)
        except ImportError:
            print("[Cluster] hdbscan not installed — using KMeans fallback")
            from sklearn.cluster import KMeans
            k = max(2, len(embeddings) // 10)
            return KMeans(n_clusters=k, random_state=42).fit_predict(embeddings)

    def assign_clusters(self, cases: List[FailureCase]) -> List[FailureCase]:
        """Run full clustering pipeline and assign cluster_id to each case."""
        if len(cases) < self.min_cluster_size:
            return cases

        embeddings = np.array([c.embedding for c in cases if c.embedding])
        if len(embeddings) == 0:
            return cases

        reduced = self.reduce_dimensions(embeddings)
        labels  = self.cluster(reduced)

        j = 0
        for case in cases:
            if case.embedding:
                case.cluster_id = int(labels[j]) if labels[j] >= 0 else None
                j += 1
        return cases

    def get_cluster_groups(
        self, cases: List[FailureCase]
    ) -> Dict[int, List[FailureCase]]:
        groups: Dict[int, List[FailureCase]] = {}
        for case in cases:
            if case.cluster_id is not None and case.cluster_id >= 0:
                groups.setdefault(case.cluster_id, []).append(case)
        return groups


class RuleSynthesizer:
    """
    Synthesizes avoidance rules from high-frequency failure clusters.
    Uses LLM to generate: IF [context] THEN AVOID [action] BECAUSE [cause]
    """

    def __init__(self, llm_fn=None, recurrence_threshold: int = 3):
        self.llm_fn = llm_fn or self._mock_llm
        self.recurrence_threshold = recurrence_threshold

    def _mock_llm(self, prompt: str) -> str:
        """Mock LLM for testing without API key."""
        return (
            "IF the task involves API calls with unvalidated parameters "
            "THEN AVOID executing without schema validation "
            "BECAUSE unvalidated inputs cause TOOL_FAIL errors in 73% of cases."
        )

    def synthesize_rule(
        self,
        cluster_id: int,
        cases: List[FailureCase],
    ) -> Optional[RuleRecord]:
        """Generate a rule from a cluster of failures."""
        high_recurrence = [c for c in cases
                           if c.recurrence_count >= self.recurrence_threshold]
        if not high_recurrence:
            return None

        # Build synthesis prompt
        examples = "\n".join([
            f"- Task: {c.task_context[:80]} | Error: {c.error_type.value} | "
            f"Root: {c.root_cause[:80]} | Fix: {c.corrective_action[:80]}"
            for c in high_recurrence[:5]
        ])

        prompt = f"""Given these {len(high_recurrence)} failure cases from cluster {cluster_id}:
{examples}

Extract a generalizable avoidance rule in the format:
IF [context_pattern] THEN AVOID [action_pattern] BECAUSE [causal_summary]

Rule:"""

        rule_text = self.llm_fn(prompt)

        return RuleRecord(
            id         = str(uuid.uuid4()),
            cluster_id = cluster_id,
            rule_text  = rule_text.strip(),
            confidence = min(0.95, 0.5 + len(high_recurrence) * 0.05),
            created_at = datetime.utcnow().isoformat(),
            validated  = False,
        )

    def validate_rule(
        self,
        rule: RuleRecord,
        held_out: List[FailureCase],
    ) -> float:
        """
        Validate rule against held-out failure set.
        Returns precision: fraction of held-out cases the rule correctly covers.
        """
        if not held_out:
            return 0.0
        # Simplified: check keyword overlap between rule and held-out cases
        rule_words = set(rule.rule_text.lower().split())
        hits = 0
        for case in held_out:
            case_words = set((case.task_context + " " + case.root_cause).lower().split())
            overlap = len(rule_words & case_words) / (len(rule_words) + 1)
            if overlap > 0.15:
                hits += 1
        precision = hits / len(held_out)
        rule.confidence = precision
        rule.validated  = True
        return precision
