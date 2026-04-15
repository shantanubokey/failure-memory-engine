"""
FailureCase — Structured Failure Schema (Component 1)
Core typed record for every captured agent failure event.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime
import uuid


class ErrorType(str, Enum):
    TOOL_FAIL    = "TOOL_FAIL"
    HALLUCINATION = "HALLUCINATION"
    LOOP         = "LOOP"
    TIMEOUT      = "TIMEOUT"
    REFUSED      = "REFUSED"
    LOGIC        = "LOGIC"


@dataclass
class FailureCase:
    # Identity
    id:                str   = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:         str   = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Context
    task_context:      str   = ""   # full natural-language task description
    sub_task:          str   = ""   # granular step where failure occurred

    # Classification
    error_type:        ErrorType = ErrorType.LOGIC
    severity:          float = 0.5  # normalized [0.0–1.0]

    # Causal analysis
    root_cause:        str   = ""   # LLM-generated causal analysis
    corrective_action: str   = ""   # verified fix that resolved the failure
    outcome_verified:  bool  = False

    # Vector & clustering
    embedding:         Optional[List[float]] = None   # 1536-dim
    cluster_id:        Optional[int]         = None
    rule_id:           Optional[str]         = None

    # Temporal & frequency
    decay_score:       float = 1.0
    recurrence_count:  int   = 1

    # Synthetic flag (Enhancement 2)
    synthetic:         bool  = False
    source_agent_id:   Optional[str] = None   # Enhancement 1: federation

    def to_dict(self) -> dict:
        return {
            "id":                self.id,
            "timestamp":         self.timestamp,
            "task_context":      self.task_context,
            "sub_task":          self.sub_task,
            "error_type":        self.error_type.value,
            "severity":          self.severity,
            "root_cause":        self.root_cause,
            "corrective_action": self.corrective_action,
            "outcome_verified":  self.outcome_verified,
            "cluster_id":        self.cluster_id,
            "rule_id":           self.rule_id,
            "decay_score":       self.decay_score,
            "recurrence_count":  self.recurrence_count,
            "synthetic":         self.synthetic,
            "source_agent_id":   self.source_agent_id,
        }


@dataclass
class RuleRecord:
    id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id:  int   = 0
    rule_text:   str   = ""   # IF [context] THEN AVOID [action] BECAUSE [cause]
    confidence:  float = 0.0
    created_at:  str   = field(default_factory=lambda: datetime.utcnow().isoformat())
    validated:   bool  = False
