"""
Evaluation Framework (Part 4)
M1: Failure Recurrence Rate (FRR)
M2: Mean Recovery Steps (MRS)
M3: Retrieval Precision@5
M4: Rule Precision
M5: Confidence Calibration (ECE)
"""

import numpy as np
from typing import List, Dict, Tuple
from core.schema import FailureCase


def failure_recurrence_rate(
    task_runs: List[Dict],
    known_failure_ids: List[str],
) -> float:
    """
    M1: % of task runs that repeat a previously seen failure pattern.
    Lower is better.
    """
    if not task_runs:
        return 0.0
    repeated = sum(1 for run in task_runs
                   if run.get("failure_id") in known_failure_ids)
    return repeated / len(task_runs)


def mean_recovery_steps(results: List[Dict]) -> float:
    """
    M2: Avg number of replanning iterations before successful correction.
    Lower is better.
    """
    steps = [r["iterations"] for r in results if r.get("success")]
    return float(np.mean(steps)) if steps else float("inf")


def retrieval_precision_at_k(
    retrieved_lists: List[List[FailureCase]],
    relevant_ids_lists: List[List[str]],
    k: int = 5,
) -> float:
    """
    M3: % of retrieved failures genuinely relevant to current task.
    Target > 85%.
    """
    precisions = []
    for retrieved, relevant in zip(retrieved_lists, relevant_ids_lists):
        top_k   = retrieved[:k]
        hits    = sum(1 for c in top_k if c.id in relevant)
        precisions.append(hits / k)
    return float(np.mean(precisions)) if precisions else 0.0


def rule_precision(
    rules_applied: List[Dict],
) -> float:
    """
    M4: % of auto-generated rules that correctly prevent future failures.
    Target > 78%.
    """
    if not rules_applied:
        return 0.0
    correct = sum(1 for r in rules_applied if r.get("prevented_failure"))
    return correct / len(rules_applied)


def mean_reciprocal_rank(
    retrieved_lists: List[List[FailureCase]],
    relevant_ids_lists: List[List[str]],
) -> float:
    """MRR@5 for retrieval evaluation (Enhancement 2: synthetic augmentation)."""
    mrr = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_ids_lists):
        for rank, case in enumerate(retrieved[:5], 1):
            if case.id in relevant:
                mrr += 1.0 / rank
                break
    return mrr / len(retrieved_lists) if retrieved_lists else 0.0


def print_report(
    frr: float,
    mrs: float,
    prec5: float,
    rule_prec: float,
    ece: float,
):
    print("\n" + "="*55)
    print("  FME-Agent — Evaluation Report")
    print("="*55)
    print(f"  M1 Failure Recurrence Rate  : {frr:.2%}  (↓ lower better)")
    print(f"  M2 Mean Recovery Steps      : {mrs:.2f}   (↓ lower better)")
    print(f"  M3 Retrieval Precision@5    : {prec5:.2%} (↑ target >85%)")
    print(f"  M4 Rule Precision           : {rule_prec:.2%} (↑ target >78%)")
    print(f"  M5 ECE (Calibration Error)  : {ece:.4f}  (↓ lower better)")
    print("="*55)
