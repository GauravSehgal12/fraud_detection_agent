from __future__ import annotations

import numpy as np

from .metrics import classification_metrics, precision_at_review_rate, recall_at_review_rate


def threshold_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = classification_metrics(y_true, scores, threshold)
        cm = metrics["Confusion_Matrix"]
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["Precision"],
                "recall": metrics["Recall"],
                "f1": metrics["F1_Score"],
                "false_positive_rate": metrics["False_Positive_Rate"],
                "review_rate": float((np.asarray(scores) >= threshold).mean()),
                "TP": float(cm["TP"]),
                "FP": float(cm["FP"]),
                "FN": float(cm["FN"]),
                "TN": float(cm["TN"]),
            }
        )
    return rows


def review_capacity_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    review_rates: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> list[dict[str, float]]:
    rows = []
    for rate in review_rates:
        rows.append(
            {
                "review_rate": float(rate),
                "review_count": float(max(1, int(np.ceil(len(scores) * rate)))),
                "recall": recall_at_review_rate(y_true, scores, rate),
                "precision": precision_at_review_rate(y_true, scores, rate),
            }
        )
    return rows
