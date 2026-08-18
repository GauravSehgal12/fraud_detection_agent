from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.split import time_based_split
from src.services.model_loader import ModelLoader
from src.services.rule_engine import RuleEngine
from src.evaluation.evaluate import (
    FEATURE_STORE_PATH,
    TRANSACTION_PATH,
    TARGET_REVIEW_RATE,
    behavioral_scores,
    fit_calibrator,
    load_feature_store,
    model_scores,
    choose_fusion_policy,
    _find_temporal_windows,
)
from src.evaluation.review_policy import rank_review_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "eval_results" / "fusion_segment_diagnostics.json"
CALIBRATOR_PATH = PROJECT_ROOT / "models" / "isotonic_calibrator.joblib"


def _segment_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    amount = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0.0)
    amount_ratio = pd.to_numeric(df["amount_vs_card_avg"], errors="coerce").fillna(0.0)
    return {
        "all": pd.Series(True, index=df.index),
        "new_card": df["new_card"].fillna(0).astype(int).eq(1),
        "known_card": df["new_card"].fillna(0).astype(int).eq(0),
        "new_device": df["new_device_profile"].fillna(0).astype(int).eq(1),
        "known_device": df["new_device_profile"].fillna(0).astype(int).eq(0),
        "new_card_and_new_device": (
            df["new_card"].fillna(0).astype(int).eq(1)
            & df["new_device_profile"].fillna(0).astype(int).eq(1)
        ),
        "known_card_new_device": (
            df["new_card"].fillna(0).astype(int).eq(0)
            & df["new_device_profile"].fillna(0).astype(int).eq(1)
        ),
        "new_card_known_device": (
            df["new_card"].fillna(0).astype(int).eq(1)
            & df["new_device_profile"].fillna(0).astype(int).eq(0)
        ),
        "new_card_device_pair": df["card_device_seen_before"].fillna(0).astype(int).eq(0),
        "known_card_device_pair": df["card_device_seen_before"].fillna(0).astype(int).eq(1),
        "high_amount_gt_200": amount.gt(200),
        "high_amount_gt_500": amount.gt(500),
        "amount_gt_2x_card_avg": amount_ratio.gt(2),
        "velocity_1h": df["card_txn_count_1h"].fillna(0).gt(0),
        "velocity_24h": df["card_txn_count_24h"].fillna(0).gt(0),
        "shared_device_10plus_cards": df["device_profile_unique_cards"].fillna(0).ge(10),
        "shared_device_20plus_cards": df["device_profile_unique_cards"].fillna(0).ge(20),
    }


def _top_k_metrics(y: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y_segment = y[mask]
    score_segment = scores[mask]
    if len(y_segment) == 0:
        return {"rows": 0, "frauds": 0, "fraud_rate": None, "recall_at_5pct": None, "precision_at_5pct": None}
    metrics = rank_review_metrics(y_segment, score_segment, TARGET_REVIEW_RATE)
    return {
        "rows": int(len(y_segment)),
        "frauds": int(y_segment.sum()),
        "fraud_rate": float(y_segment.mean()),
        "recall_at_5pct": float(metrics["recall"]),
        "precision_at_5pct": float(metrics["precision"]),
        "review_count": int(metrics["review_count"]),
    }


def _compare_segment(name: str, df: pd.DataFrame, y: np.ndarray, baseline: np.ndarray, fusion: np.ndarray) -> dict[str, Any]:
    masks = _segment_masks(df)
    mask = masks[name].to_numpy(dtype=bool)
    base = _top_k_metrics(y, baseline, mask)
    fused = _top_k_metrics(y, fusion, mask)
    if base["recall_at_5pct"] is None:
        delta_recall = None
        delta_precision = None
    else:
        delta_recall = fused["recall_at_5pct"] - base["recall_at_5pct"]
        delta_precision = fused["precision_at_5pct"] - base["precision_at_5pct"]
    return {
        "segment": name,
        "baseline": base,
        "fusion": fused,
        "delta_recall": delta_recall,
        "delta_precision": delta_precision,
    }


def run() -> dict[str, Any]:
    print("\n========== FUSION SEGMENT DIAGNOSTICS ==========")
    df = load_feature_store()
    _, validation_df, _ = time_based_split(df)
    calibration_df, policy_df, stability_df = _find_temporal_windows(validation_df.reset_index(drop=True))

    loaded = ModelLoader().load_all()
    model, features = loaded["model"], loaded["features"]
    rule_engine = RuleEngine()

    model_cal = model_scores(model, calibration_df, features)
    model_policy = model_scores(model, policy_df, features)
    model_stability = model_scores(model, stability_df, features)
    behavioral_policy, _ = behavioral_scores(policy_df, rule_engine)
    behavioral_stability, _ = behavioral_scores(stability_df, rule_engine)

    y_cal = calibration_df["isFraud"].to_numpy(dtype=int)
    y_policy = policy_df["isFraud"].to_numpy(dtype=int)
    y_stability = stability_df["isFraud"].to_numpy(dtype=int)

    calibrator = fit_calibrator(model_cal, y_cal)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    calibrated_policy = np.clip(calibrator.predict(model_policy), 0.0, 1.0)
    calibrated_stability = np.clip(calibrator.predict(model_stability), 0.0, 1.0)

    policy = choose_fusion_policy(y_policy, calibrated_policy, behavioral_policy)
    fusion_policy = policy["model_weight"] * calibrated_stability + policy["behavioral_weight"] * behavioral_stability
    baseline_policy = calibrated_stability

    segment_names = list(_segment_masks(stability_df).keys())
    rows = [
        _compare_segment(name, stability_df, y_stability, baseline_policy, fusion_policy)
        for name in segment_names
    ]

    rows_sorted = sorted(
        rows[1:],
        key=lambda item: (
            item["delta_recall"] if item["delta_recall"] is not None else 999,
            item["delta_precision"] if item["delta_precision"] is not None else 999,
        ),
    )
    worst = rows_sorted[:5]
    best = sorted(
        rows[1:],
        key=lambda item: item["delta_recall"] if item["delta_recall"] is not None else -999,
        reverse=True,
    )[:5]

    result = {
        "policy": policy,
        "stability_window": {
            "rows": int(len(stability_df)),
            "frauds": int(y_stability.sum()),
            "fraud_rate": float(y_stability.mean()),
        },
        "overall": {
            "baseline": _top_k_metrics(y_stability, baseline_policy, np.ones(len(y_stability), dtype=bool)),
            "fusion": _top_k_metrics(y_stability, fusion_policy, np.ones(len(y_stability), dtype=bool)),
        },
        "segments": rows,
        "worst_fusion_segments_by_recall": worst,
        "best_fusion_segments_by_recall": best,
        "interpretation": {
            "purpose": "Identify temporal segments where behavioral fusion loses recall or precision versus the calibrated-model baseline at fixed 5% review capacity.",
            "do_not_use_test_for_tuning": True,
            "next_step": "Use the worst stability segments to design a conservative behavioral overlay, then re-test on the untouched test set.",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Fusion policy: model={policy['model_weight']:.2f}, behavioral={policy['behavioral_weight']:.2f}")
    print(f"Stability rows: {len(stability_df):,} | frauds: {int(y_stability.sum()):,}")
    print("\nWorst segments for fusion:")
    for item in worst:
        print(
            f"  {item['segment']}: "
            f"recall delta={item['delta_recall']:.4f}, "
            f"precision delta={item['delta_precision']:.4f}"
        )
    print("\nBest segments for fusion:")
    for item in best:
        print(
            f"  {item['segment']}: "
            f"recall delta={item['delta_recall']:.4f}, "
            f"precision delta={item['delta_precision']:.4f}"
        )
    print(f"\nSaved: {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    run()
