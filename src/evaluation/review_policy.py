from __future__ import annotations

from typing import Any

import numpy as np


def top_k_review_selection(
    scores: np.ndarray,
    review_rate: float = 0.05,
) -> tuple[np.ndarray, int]:
    """Select exactly the requested number of highest-risk transactions.

    This is intentionally rank-based rather than threshold-quantile based.
    Isotonic calibration can produce many tied probabilities, so a quantile
    threshold does not necessarily select exactly the requested review rate.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional array")
    if not 0 < review_rate <= 1:
        raise ValueError("review_rate must be in (0, 1]")
    if len(scores) == 0:
        return np.zeros(0, dtype=bool), 0

    review_count = max(1, int(np.ceil(len(scores) * review_rate)))
    review_count = min(review_count, len(scores))

    # Stable deterministic ordering makes tied calibrated scores reproducible.
    order = np.argsort(-scores, kind="mergesort")
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:review_count]] = True
    return selected, review_count


def rank_review_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    review_rate: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a fixed analyst-review capacity using exact top-k selection."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")

    selected, review_count = top_k_review_selection(scores, review_rate)
    fraud_count = int(y_true.sum())
    reviewed_fraud_count = int(y_true[selected].sum())

    return {
        "review_rate": float(review_count / len(scores)) if len(scores) else 0.0,
        "target_review_rate": float(review_rate),
        "review_count": int(review_count),
        "recall": float(reviewed_fraud_count / fraud_count) if fraud_count else 0.0,
        "precision": float(y_true[selected].mean()) if review_count else 0.0,
        "selected_fraud_count": reviewed_fraud_count,
        "selected_indices": np.flatnonzero(selected).tolist(),
    }


def score_threshold_at_review_capacity(
    scores: np.ndarray,
    review_rate: float = 0.05,
) -> float:
    """Return the lowest score among the exact top-k reviewed transactions.

    This threshold is for reporting/routing only. Because scores can be tied,
    runtime systems that require exactly N reviews should use rank_review_metrics
    or top_k_review_selection rather than ``scores >= threshold``.
    """
    scores = np.asarray(scores, dtype=float)
    selected, _ = top_k_review_selection(scores, review_rate)
    selected_scores = scores[selected]
    return float(selected_scores.min()) if len(selected_scores) else 0.0
