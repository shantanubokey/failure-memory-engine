"""
FME-Agent — Main Engine
Orchestrates all components: embedding, RAG, clustering, decay, self-correction.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.schema import FailureCase, ErrorType
from core.embedding import EmbeddingEngine
from core.vector_store import InMemoryVectorStore
from retrieval.failure_rag import FailureRAG
from clustering.cluster_engine import ClusterEngine, RuleSynthesizer
from decay.decay_engine import DecayEngine
from self_correction.correction_loop import SelfCorrectionLoop
from typing import Optional, Dict, Any


class FMEAgent:
    """
    Failure Memory Engine — top-level orchestrator.
    Plug into any LangGraph agent as a middleware layer.
    """

    def __init__(
        self,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        agent_id: str = "fme-agent-01",
        cluster_every_n: int = 100,
    ):
        self.embed   = EmbeddingEngine(use_openai=use_openai,
                                       openai_api_key=openai_api_key)
        self.store   = InMemoryVectorStore()
        self.rag     = FailureRAG(self.store, self.embed)
        self.cluster = ClusterEngine()
        self.rules   = RuleSynthesizer()
        self.decay   = DecayEngine()
        self.loop    = SelfCorrectionLoop(self.rag, self.store, self.embed,
                                          agent_id=agent_id)
        self.agent_id        = agent_id
        self.cluster_every_n = cluster_every_n
        self._failure_count  = 0

    def pre_action_check(
        self,
        task_context: str,
        sub_task: str,
    ) -> Dict:
        """
        Call BEFORE executing any agent action.
        Returns risk assessment and optional memory prompt for injection.
        """
        result = self.rag.run(task_context, sub_task)
        if result["memory_prompt"]:
            print(f"[FME] ⚠ Risk={result['risk_score']:.2f} — injecting failure memory")
        if result["gate_active"]:
            print("[FME] 🔒 CONFIDENCE GATE ACTIVE — require 2-step self-check")
        return result

    def on_failure(
        self,
        task_context: str,
        sub_task: str,
        action: str,
        error: str,
        severity: float = 0.5,
    ) -> Dict:
        """
        Call WHEN an agent action fails.
        Runs self-correction loop and stores the failure case.
        """
        result = self.loop.run(task_context, sub_task, action, error, severity)
        self._failure_count += 1

        # Trigger offline clustering every N failures
        if self._failure_count % self.cluster_every_n == 0:
            self._run_offline_pipeline()

        return result

    def _run_offline_pipeline(self):
        """Offline: update decay, cluster, synthesize rules, prune."""
        print(f"[FME] Running offline pipeline (N={self._failure_count})...")
        all_cases = self.store.get_all()

        # Update decay scores
        self.decay.update_all(all_cases)

        # Cluster
        clustered = self.cluster.assign_clusters(all_cases)
        groups    = self.cluster.get_cluster_groups(clustered)

        # Synthesize rules per cluster
        for cluster_id, cases in groups.items():
            rule = self.rules.synthesize_rule(cluster_id, cases)
            if rule:
                print(f"[FME] Rule synthesized for cluster {cluster_id}: "
                      f"{rule.rule_text[:80]}...")

        # Prune stale memories
        self.decay.prune(self.store)
        print(f"[FME] Offline pipeline complete. Store size: {self.store.count()}")

    def stats(self) -> Dict:
        return {
            "total_failures_captured": self._failure_count,
            "active_memories":         self.store.count(),
            "agent_id":                self.agent_id,
        }
