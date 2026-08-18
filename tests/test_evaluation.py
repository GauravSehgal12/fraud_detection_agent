import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    classification_metrics,
    operating_point_metrics,
)
from src.evaluation.segment_analysis import segment_metrics
from src.evaluation.threshold_analysis import threshold_analysis


def test_classification_metrics():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = classification_metrics(y, scores, threshold=0.5)

    assert result["Precision"] == 1.0
    assert result["Recall"] == 1.0
    assert result["F1_Score"] == 1.0
    assert result["Confusion_Matrix"] == {
        "TN": 2, "FP": 0, "FN": 0, "TP": 2
    }


def test_review_capacity_metrics():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8])

    result = operating_point_metrics(y, scores, review_rates=(0.20, 0.40))

    assert result["Recall_at_20%_review"] == 0.5
    assert result["Recall_at_40%_review"] == 1.0


def test_threshold_analysis():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    rows = threshold_analysis(y, scores, thresholds=(0.5,))

    assert len(rows) == 1
    assert rows[0]["precision"] == 1.0
    assert rows[0]["recall"] == 1.0
    assert rows[0]["review_rate"] == 0.5


def test_cold_start_segments():
    df = pd.DataFrame(
        {
            "new_card": [0, 0, 1, 1],
            "new_device_profile": [0, 1, 0, 1],
        }
    )
    y = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.8, 0.7, 0.9])

    result = segment_metrics(df, y, scores, threshold=0.5)

    assert result["existing_card_existing_device"]["count"] == 1
    assert result["existing_card_new_device"]["count"] == 1
    assert result["new_card_existing_device"]["count"] == 1
    assert result["new_card_new_device"]["count"] == 1
