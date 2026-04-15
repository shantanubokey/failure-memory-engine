"""
Temporal Decay & Memory Pruning (Component 5)
Exponential decay: decay_score(t) = exp(−λ × Δt_days)
Default λ=0.02, half-life ≈ 35 days.
"""

import math
from datetime import datetime, timezone
from typing import List, Tuple
from core.schema import FailureCase
from core.vector_store import InMemoryVectorStore


DEFAULT_LAMBDA = 0.02   # half-life ≈ 35 days
PRUNE_THRESHOLD = 0.05  # remove if decay < 0.05 AND recurrence < 3
HIGH_RECURRENCE = 10    # preserve regardless of age


def compute_decay(timestamp_iso: str, lam: float = DEFAULT_LAMBDA) -> float:
    """
    decay_score(t) = exp(−λ × Δt_days)
    """
    try:
        ts  = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta_days = (now - ts).total_seconds() / 86400
        return math.exp(-lam * delta_days)
    except Exception:
        return 1.0


def adjusted_score(
    cosine_sim: float,
    case: FailureCase,
) -> float:
    """
    Retrieval re-ranking with decay:
    adjusted = cosine_similarity × decay_score × severity
    """
    return cosine_sim * case.decay_score * case.severity


class DecayEngine:
    def __init__(self, lam: float = DEFAULT_LAMBDA):
        self.lam = lam

    def update_all(self, cases: List[FailureCase]) -> List[FailureCase]:
        """Recalculate decay_score for all cases."""
        for case in cases:
            case.decay_score = compute_decay(case.timestamp, self.lam)
        return cases

    def should_prune(self, case: FailureCase) -> bool:
        """
        Prune if: decay_score < 0.05 AND recurrence_count < 3
        Preserve if: recurrence_count >= 10 (high-recurrence pattern)
        """
        if case.recurrence_count >= HIGH_RECURRENCE:
            return False
        return case.decay_score < PRUNE_THRESHOLD and case.recurrence_count < 3

    def prune(
        self,
        store: InMemoryVectorStore,
    ) -> Tuple[List[str], List[FailureCase]]:
        """
        Run pruning pass.
        Returns (pruned_ids, archived_cases).
        """
        all_cases = store.get_all()
        self.update_all(all_cases)

        pruned   = []
        archived = []

        for case in all_cases:
            if self.should_prune(case):
                store.delete(case.id)
                archived.append(case)   # move to cold store
                pruned.append(case.id)

        print(f"[Decay] Pruned {len(pruned)} cases | Archived {len(archived)}")
        return pruned, archived

    def get_half_life_days(self) -> float:
        return math.log(2) / self.lam
