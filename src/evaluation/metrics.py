from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "ROC_AUC": float(roc_auc_score(y_true, scores)),
        "PR_AUC": float(average_precision_score(y_true, scores)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1_Score": float(f1_score(y_true, predictions, zero_division=0)),
        "False_Positive_Rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "False_Negative_Rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "Confusion_Matrix": {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        },
        "threshold": float(threshold),
    }


def recall_at_review_rate(
    y_true: np.ndarray,
    scores: np.ndarray,
    review_rate: float,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_review = max(1, int(np.ceil(len(scores) * review_rate)))
    order = np.argsort(-scores)
    selected = order[:n_review]
    total_fraud = int(y_true.sum())
    if total_fraud == 0:
        return 0.0
    return float(y_true[selected].sum() / total_fraud)


def precision_at_review_rate(
    y_true: np.ndarray,
    scores: np.ndarray,
    review_rate: float,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_review = max(1, int(np.ceil(len(scores) * review_rate)))
    selected = np.argsort(-scores)[:n_review]
    return float(y_true[selected].mean())


def operating_point_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    review_rates: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[str, float]:
    result: dict[str, float] = {}
    for rate in review_rates:
        key = f"{int(rate * 100)}%"
        result[f"Recall_at_{key}_review"] = recall_at_review_rate(
            y_true, scores, rate
        )
        result[f"Precision_at_{key}_review"] = precision_at_review_rate(
            y_true, scores, rate
        )
    return result
