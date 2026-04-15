"""
Self-Correction Loop (Component 6)
On execution failure: capture → root cause → replan (max 3) → store result.
"""

from typing import Callable, Optional, Dict, Any
from core.schema import FailureCase, ErrorType
from core.embedding import EmbeddingEngine
from core.vector_store import InMemoryVectorStore
from retrieval.failure_rag import FailureRAG
from decay.decay_engine import compute_decay
from datetime import datetime
import uuid


MAX_ITERATIONS = 3


class SelfCorrectionLoop:
    def __init__(
        self,
        rag:        FailureRAG,
        store:      InMemoryVectorStore,
        embed:      EmbeddingEngine,
        llm_fn:     Callable[[str], str] = None,
        agent_id:   str = "default",
    ):
        self.rag      = rag
        self.store    = store
        self.embed    = embed
        self.llm_fn   = llm_fn or self._mock_llm
        self.agent_id = agent_id

    def _mock_llm(self, prompt: str) -> str:
        return "The root cause was an unvalidated API parameter causing a schema mismatch."

    def _generate_root_cause(self, action: str, error: str) -> str:
        prompt = (
            f"You attempted: {action}\n"
            f"The result was: {error}\n"
            f"In one sentence, what was the root cause?"
        )
        return self.llm_fn(prompt)

    def _replan(
        self,
        task_context: str,
        sub_task: str,
        root_cause: str,
        memory_prompt: Optional[str],
        iteration: int,
    ) -> str:
        memory_section = f"\n{memory_prompt}" if memory_prompt else ""
        prompt = (
            f"Task: {task_context}\n"
            f"Failed step: {sub_task}\n"
            f"Root cause: {root_cause}\n"
            f"{memory_section}\n"
            f"Iteration {iteration}/{MAX_ITERATIONS}. "
            f"Provide a corrected action plan:"
        )
        return self.llm_fn(prompt)

    def _classify_error(self, error: str) -> ErrorType:
        error_lower = error.lower()
        if "timeout" in error_lower:    return ErrorType.TIMEOUT
        if "refused" in error_lower:    return ErrorType.REFUSED
        if "loop" in error_lower:       return ErrorType.LOOP
        if "hallucin" in error_lower:   return ErrorType.HALLUCINATION
        if "tool" in error_lower or "api" in error_lower: return ErrorType.TOOL_FAIL
        return ErrorType.LOGIC

    def run(
        self,
        task_context: str,
        sub_task: str,
        action: str,
        error: str,
        base_severity: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Execute self-correction loop.
        Returns result dict with outcome, stored FailureCase, and iterations used.
        """
        root_cause  = self._generate_root_cause(action, error)
        error_type  = self._classify_error(error)
        severity    = base_severity
        corrective  = None
        success     = False

        for iteration in range(1, MAX_ITERATIONS + 1):
            # Get failure memory for this context
            rag_result    = self.rag.run(task_context, sub_task)
            memory_prompt = rag_result["memory_prompt"]

            # Attempt replanning
            corrective = self._replan(
                task_context, sub_task, root_cause, memory_prompt, iteration
            )

            # Simulate execution (in production: actually execute corrective)
            # For now: assume success on iteration 2+ if memory was injected
            if iteration >= 2 and memory_prompt:
                success = True
                break

            # Escalate severity on continued failure
            severity = min(1.0, severity + 0.1)

        # Store FailureCase
        case = FailureCase(
            id                = str(uuid.uuid4()),
            timestamp         = datetime.utcnow().isoformat(),
            task_context      = task_context,
            sub_task          = sub_task,
            error_type        = error_type,
            severity          = severity,
            root_cause        = root_cause,
            corrective_action = corrective or "",
            outcome_verified  = success,
            decay_score       = 1.0,
            recurrence_count  = 1,
            source_agent_id   = self.agent_id,
        )
        case.embedding = self.embed.embed_failure(case)
        self.store.upsert(case)

        outcome = "SUCCESS" if success else "ESCALATED"
        print(f"[SelfCorrection] {outcome} after {iteration} iterations | "
              f"severity={severity:.2f} | verified={success}")

        return {
            "outcome":     outcome,
            "iterations":  iteration,
            "case_id":     case.id,
            "severity":    severity,
            "success":     success,
            "corrective":  corrective,
        }
