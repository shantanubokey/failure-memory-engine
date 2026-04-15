"""
Failure-Aware Inference — RAG over Failures (Component 3)
Retrieves similar past failures at inference time and injects
FAILURE MEMORY BLOCK into the agent's system prompt.
"""

from typing import List, Tuple, Optional, Dict
from core.schema import FailureCase
from core.vector_store import InMemoryVectorStore
from core.embedding import EmbeddingEngine


SIMILARITY_THRESHOLD = 0.82
HIGH_RISK_THRESHOLD  = 0.75
CONFIDENCE_GATE      = 0.90


FAILURE_MEMORY_BLOCK = """
┌─────────────────────────────────────────────────────────────┐
│ [FAILURE MEMORY — {risk_level}]                             │
│ In a past similar task you encountered:                     │
│   Error: {error_type} — {root_cause}                        │
│   Verified Fix: {corrective_action}                         │
│   Recurrence: {recurrence_count}x | Similarity: {sim:.2f}  │
│ ⚠ Avoid repeating this pattern. Apply the verified fix.    │
└─────────────────────────────────────────────────────────────┘
""".strip()


class FailureRAG:
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        embedding_engine: EmbeddingEngine,
        theta: float = SIMILARITY_THRESHOLD,
    ):
        self.store  = vector_store
        self.embed  = embedding_engine
        self.theta  = theta

    def retrieve(
        self,
        task_context: str,
        sub_task: str,
        top_k: int = 5,
    ) -> List[Tuple[FailureCase, float]]:
        """Embed current context and retrieve top-k similar failures."""
        q_embed = self.embed.embed_query(task_context, sub_task)
        return self.store.search(q_embed, top_k=top_k, min_similarity=self.theta)

    def compute_risk_score(
        self,
        retrieved: List[Tuple[FailureCase, float]],
    ) -> float:
        """
        risk_score = max(similarity_i × severity_i) for i in top-k
        """
        if not retrieved:
            return 0.0
        return max(sim * case.severity for case, sim in retrieved)

    def build_memory_prompt(
        self,
        retrieved: List[Tuple[FailureCase, float]],
        risk_score: float,
    ) -> Optional[str]:
        """
        Build FAILURE MEMORY BLOCK for injection into system prompt.
        Returns None if risk_score below threshold.
        """
        if risk_score < HIGH_RISK_THRESHOLD or not retrieved:
            return None

        risk_level = "CRITICAL" if risk_score >= CONFIDENCE_GATE else "HIGH RISK"
        best_case, best_sim = max(retrieved, key=lambda x: x[1] * x[0].severity)

        return FAILURE_MEMORY_BLOCK.format(
            risk_level        = risk_level,
            error_type        = best_case.error_type.value,
            root_cause        = best_case.root_cause[:120],
            corrective_action = best_case.corrective_action[:120],
            recurrence_count  = best_case.recurrence_count,
            sim               = best_sim,
        )

    def requires_confidence_gate(self, risk_score: float) -> bool:
        """If risk_score > 0.90 → activate 2-step self-check before execution."""
        return risk_score >= CONFIDENCE_GATE

    def run(
        self,
        task_context: str,
        sub_task: str,
    ) -> Dict:
        """
        Full inference-time failure check.
        Returns dict with: retrieved, risk_score, memory_prompt, gate_active
        """
        retrieved    = self.retrieve(task_context, sub_task)
        risk_score   = self.compute_risk_score(retrieved)
        memory_prompt = self.build_memory_prompt(retrieved, risk_score)
        gate_active  = self.requires_confidence_gate(risk_score)

        return {
            "retrieved":      retrieved,
            "risk_score":     risk_score,
            "memory_prompt":  memory_prompt,
            "gate_active":    gate_active,
            "n_retrieved":    len(retrieved),
        }
