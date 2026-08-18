from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        scores,
        n_bins=n_bins,
        strategy="quantile",
    )

    return {
        "Brier_Score": float(brier_score_loss(y_true, scores)),
        "calibration_curve": {
            "mean_predicted_probability": mean_predicted.tolist(),
            "fraction_positive": fraction_positive.tolist(),
        },
    }
