"""
Enhancement 5 — Failure Forecasting (Proactive FME)
Predicts P(failure in next action step) before execution.
Uses XGBoost on [current_embedding_features, risk_score, task_complexity].
"""

import numpy as np
from typing import List, Optional, Dict


class FailureForecaster:
    """
    Binary classifier: P(failure) given current context + retrieved risk.
    Trained on historical (embedding, risk_score, complexity) → failure_occurred.
    """

    def __init__(self):
        self._model = None
        self._trained = False

    def _extract_features(
        self,
        embedding: List[float],
        risk_score: float,
        task_complexity: float,
        n_retrieved: int,
    ) -> np.ndarray:
        """
        Feature vector:
        - PCA-reduced embedding (top 32 components)
        - risk_score
        - task_complexity (e.g. number of steps / tools)
        - n_retrieved (how many similar failures found)
        """
        emb_arr = np.array(embedding[:32])   # first 32 dims as proxy
        extra   = np.array([risk_score, task_complexity, n_retrieved])
        return np.concatenate([emb_arr, extra])

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Train XGBoost classifier on historical failure data."""
        try:
            from xgboost import XGBClassifier
            self._model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
            )
            self._model.fit(X, y)
            self._trained = True
            print("[Forecaster] XGBoost trained")
        except ImportError:
            print("[Forecaster] xgboost not installed — using logistic fallback")
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(max_iter=500)
            self._model.fit(X, y)
            self._trained = True

    def predict_proba(
        self,
        embedding: List[float],
        risk_score: float,
        task_complexity: float = 1.0,
        n_retrieved: int = 0,
    ) -> float:
        """Returns P(failure) for the current action step."""
        if not self._trained or self._model is None:
            # Fallback: use risk_score as proxy
            return min(1.0, risk_score * 1.2)

        features = self._extract_features(
            embedding, risk_score, task_complexity, n_retrieved
        ).reshape(1, -1)

        proba = self._model.predict_proba(features)[0][1]
        return float(proba)

    def should_warn(self, p_failure: float, threshold: float = 0.65) -> bool:
        return p_failure >= threshold

    def compute_ece(
        self,
        probas: List[float],
        actuals: List[int],
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error (ECE)."""
        probas  = np.array(probas)
        actuals = np.array(actuals)
        bins    = np.linspace(0, 1, n_bins + 1)
        ece     = 0.0
        for i in range(n_bins):
            mask = (probas >= bins[i]) & (probas < bins[i + 1])
            if mask.sum() == 0:
                continue
            avg_conf = probas[mask].mean()
            avg_acc  = actuals[mask].mean()
            ece     += mask.sum() * abs(avg_conf - avg_acc)
        return float(ece / len(probas))
