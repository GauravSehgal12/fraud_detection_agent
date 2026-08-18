from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import classification_metrics, operating_point_metrics


SEGMENTS = {
    "existing_card_existing_device": lambda df: (df["new_card"] == 0) & (df["new_device_profile"] == 0),
    "existing_card_new_device": lambda df: (df["new_card"] == 0) & (df["new_device_profile"] == 1),
    "new_card_existing_device": lambda df: (df["new_card"] == 1) & (df["new_device_profile"] == 0),
    "new_card_new_device": lambda df: (df["new_card"] == 1) & (df["new_device_profile"] == 1),
}


def segment_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    for name, selector in SEGMENTS.items():
        mask = selector(df).to_numpy()
        count = int(mask.sum())
        if count == 0:
            result[name] = {"count": 0}
            continue
        metrics = classification_metrics(y_true[mask], scores[mask], threshold)
        metrics.update(operating_point_metrics(y_true[mask], scores[mask]))
        metrics["count"] = count
        result[name] = metrics

    return result
