from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data.split import time_based_split
from src.services.model_loader import ModelLoader
from src.services.rule_engine import RuleEngine

from .calibration import calibration_metrics
from .metrics import classification_metrics, operating_point_metrics
from .segment_analysis import segment_metrics
from .threshold_analysis import review_capacity_analysis, threshold_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store.csv"
TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
RESULTS_DIR = PROJECT_ROOT / "eval_results"
CALIBRATOR_PATH = PROJECT_ROOT / "models" / "isotonic_calibrator.joblib"
FUSION_CONFIG_PATH = PROJECT_ROOT / "models" / "risk_fusion_config.json"

MAX_REVIEW_RATE = 0.10
TARGET_REVIEW_RATE = 0.05
MIN_FRAUDS_PER_VALIDATION_WINDOW = 100
WEIGHT_GRID = tuple(np.round(np.arange(0.50, 1.01, 0.05), 2))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_feature_store() -> pd.DataFrame:
    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(f"Feature store not found: {FEATURE_STORE_PATH}")
    if not TRANSACTION_PATH.exists():
        raise FileNotFoundError(f"Training transaction data not found: {TRANSACTION_PATH}")

    print(f"Loading feature store: {FEATURE_STORE_PATH}")
    features_df = pd.read_csv(FEATURE_STORE_PATH)
    required = {
        "TransactionID", "TransactionDT", "new_card", "new_device_profile",
        "card_device_seen_before", "card_txn_count_1h", "card_txn_count_24h",
        "device_profile_unique_cards", "amount_vs_card_avg",
    }
    missing = sorted(required - set(features_df.columns))
    if missing:
        raise ValueError("Feature store is missing columns required for evaluation: " + ", ".join(missing))

    print(f"Feature store shape: {features_df.shape}")
    print(f"Loading labels from: {TRANSACTION_PATH}")
    labels_df = pd.read_csv(TRANSACTION_PATH, usecols=["TransactionID", "isFraud"])
    if labels_df["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionID values found in label data.")

    df = features_df.merge(labels_df, on="TransactionID", how="left", validate="one_to_one")
    missing_labels = int(df["isFraud"].isna().sum())
    if missing_labels:
        raise ValueError(f"{missing_labels:,} feature-store rows have no matching isFraud label.")
    df["isFraud"] = pd.to_numeric(df["isFraud"], errors="raise").astype(int)
    return df.sort_values("TransactionDT").reset_index(drop=True)


def model_scores(model: Any, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError("Feature store is missing model features: " + ", ".join(missing))
    X = df[features].copy().apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return model.predict_proba(X)[:, 1]


def behavioral_scores(df: pd.DataFrame, rule_engine: RuleEngine) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    scores = np.zeros(len(df), dtype=float)
    all_rules: list[list[dict[str, Any]]] = []
    for position, (_, row) in enumerate(df.iterrows()):
        features = {key: row.get(key, 0) for key in (
            "TransactionAmt", "card_txn_count_1h", "card_txn_count_24h",
            "device_profile_unique_cards", "amount_vs_card_avg")}
        cold_start = {
            "is_new_card": bool(row.get("new_card", 0)),
            "is_new_device": bool(row.get("new_device_profile", 0)),
            "is_new_card_device_pair": not bool(row.get("card_device_seen_before", 0)),
            "card_history_available": not bool(row.get("new_card", 0)),
            "device_history_available": not bool(row.get("new_device_profile", 0)),
            "card_device_history_available": bool(row.get("card_device_seen_before", 0)),
        }
        transaction = {
            "TransactionID": row.get("TransactionID"),
            "TransactionDT": row.get("TransactionDT"),
            "TransactionAmt": row.get("TransactionAmt", 0.0),
            "card1": row.get("card1"),
            "DeviceInfo": row.get("DeviceInfo"),
        }
        result = rule_engine.evaluate(transaction=transaction, model_risk_score=None,
                                      cold_start_status=cold_start, features=features)
        scores[position] = float(result["behavioral_risk_score"])
        all_rules.append(result["rules_triggered"])
    return scores, all_rules


def fit_calibrator(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration window must contain both legitimate and fraud examples.")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(scores, y)
    return calibrator


def _find_temporal_windows(validation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = validation_df.sort_values("TransactionDT").reset_index(drop=True)
    fraud_positions = np.flatnonzero(df["isFraud"].to_numpy(dtype=int) == 1)
    required_frauds = MIN_FRAUDS_PER_VALIDATION_WINDOW * 3
    if len(fraud_positions) < required_frauds:
        raise ValueError(f"Validation period has {len(fraud_positions)} frauds; need at least {required_frauds}.")

    first_boundary = int(fraud_positions[MIN_FRAUDS_PER_VALIDATION_WINDOW - 1] + 1)
    second_boundary = int(fraud_positions[2 * MIN_FRAUDS_PER_VALIDATION_WINDOW - 1] + 1)
    min_window = max(1000, int(len(df) * 0.15))
    first_boundary = max(first_boundary, min_window)
    second_boundary = max(second_boundary, first_boundary + min_window)
    second_boundary = min(second_boundary, len(df) - min_window)

    calibration = df.iloc[:first_boundary].copy().reset_index(drop=True)
    policy = df.iloc[first_boundary:second_boundary].copy().reset_index(drop=True)
    stability = df.iloc[second_boundary:].copy().reset_index(drop=True)
    for name, window in (("calibration", calibration), ("policy", policy), ("stability", stability)):
        fraud_count = int(window["isFraud"].sum())
        if fraud_count < MIN_FRAUDS_PER_VALIDATION_WINDOW or window["isFraud"].nunique() < 2:
            raise ValueError(f"{name.capitalize()} window invalid: rows={len(window):,}, frauds={fraud_count:,}.")
    return calibration, policy, stability


def _review_metrics(y_true: np.ndarray, scores: np.ndarray, review_rate: float = TARGET_REVIEW_RATE) -> dict[str, float]:
    n_review = max(1, int(np.ceil(len(scores) * review_rate)))
    selected = np.argsort(-scores)[:n_review]
    total_fraud = int(y_true.sum())
    return {
        "review_rate": float(review_rate),
        "recall": float(y_true[selected].sum() / total_fraud) if total_fraud else 0.0,
        "precision": float(y_true[selected].mean()) if len(selected) else 0.0,
        "review_count": float(n_review),
    }


def choose_fusion_policy(y_policy: np.ndarray, calibrated_model_policy: np.ndarray, behavioral_policy: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for model_weight in WEIGHT_GRID:
        behavioral_weight = 1.0 - model_weight
        fused = model_weight * calibrated_model + behavioral_weight * behavioral
        review = _review_metrics(y, fused, TARGET_REVIEW_RATE)
        threshold = float(np.quantile(fused, 1.0 - TARGET_REVIEW_RATE))
        metrics = classification_metrics(y, fused, threshold=threshold)

        candidate = {
            "model_weight": float(model_weight),
            "behavioral_weight": float(behavioral_weight),
            "review_threshold": threshold,
            "high_threshold": float(min(0.95, max(threshold + 0.15, 0.70))),
            "validation_f1": float(metrics["F1_Score"]),
            "validation_precision": float(metrics["Precision"]),
            "validation_recall": float(metrics["Recall"]),
            "validation_review_rate": float((fused >= threshold).mean()),
            "validation_recall_at_5pct": review["recall"],
            "validation_precision_at_5pct": review["precision"],
        }
        key = (candidate["validation_recall_at_5pct"], candidate["validation_precision_at_5pct"], candidate["validation_f1"])
        if best is None or key > (best["validation_recall_at_5pct"], best["validation_precision_at_5pct"], best["validation_f1"]):
            best = candidate
    return best or {
        "model_weight": 1.0, "behavioral_weight": 0.0, "review_threshold": 0.5,
        "high_threshold": 0.8, "validation_f1": 0.0, "validation_precision": 0.0,
        "validation_recall": 0.0, "validation_review_rate": 0.0,
        "validation_recall_at_5pct": 0.0, "validation_precision_at_5pct": 0.0,
    }


def stability_gate(y: np.ndarray, calibrated_model: np.ndarray, behavioral: np.ndarray, policy: dict[str, float]) -> dict[str, Any]:
    fused = policy["model_weight"] * calibrated_model + policy["behavioral_weight"] * behavioral
    baseline = _review_metrics(y, calibrated_model, TARGET_REVIEW_RATE)
    fusion = _review_metrics(y, fused, TARGET_REVIEW_RATE)
    recall_change = fusion["recall"] - baseline["recall"]
    precision_change = fusion["precision"] - baseline["precision"]
    return {
        "enabled": bool(recall_change >= 0.0 and precision_change >= -0.01),
        "target_review_rate": TARGET_REVIEW_RATE, "baseline": baseline, "fusion": fusion,
        "recall_change": float(recall_change), "precision_change": float(precision_change),
    }


def evaluate() -> dict[str, Any]:
    print("\n========== FRAUD SYSTEM EVALUATION ==========")
    loaded = ModelLoader().load_all()
    model, features = loaded["model"], loaded["features"]
    df = load_feature_store()
    train_df, validation_df, test_df = time_based_split(df)
    validation_df = validation_df.reset_index(drop=True)
    test = test_df.reset_index(drop=True)
    calibration_df, policy_df, stability_df = _find_temporal_windows(validation_df)
    y_test = test["isFraud"].to_numpy(dtype=int)

    print(f"Total rows: {len(df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    for name, window in (("Calibration", calibration_df), ("Policy", policy_df), ("Stability", stability_df)):
        print(f"  {name} window: {len(window):,} rows | frauds: {int(window['isFraud'].sum()):,} | fraud rate: {window['isFraud'].mean():.4%}")
    print(f"Test rows: {len(test):,} | frauds: {int(y_test.sum()):,} | fraud rate: {y_test.mean():.4%}")

    rule_engine = RuleEngine()
    model_calibration = model_scores(model, calibration_df, features)
    model_policy = model_scores(model, policy_df, features)
    model_stability = model_scores(model, stability_df, features)
    model_test = model_scores(model, test, features)
    behavioral_calibration, _ = behavioral_scores(calibration_df, rule_engine)
    behavioral_policy, _ = behavioral_scores(policy_df, rule_engine)
    behavioral_stability, _ = behavioral_scores(stability_df, rule_engine)
    behavioral_test, test_rules = behavioral_scores(test, rule_engine)

    y_calibration = calibration_df["isFraud"].to_numpy(dtype=int)
    y_policy = policy_df["isFraud"].to_numpy(dtype=int)
    y_stability = stability_df["isFraud"].to_numpy(dtype=int)
    calibrator = fit_calibrator(model_calibration, y_calibration)
    calibrated_policy = np.clip(calibrator.predict(model_policy), 0.0, 1.0)
    calibrated_stability = np.clip(calibrator.predict(model_stability), 0.0, 1.0)
    calibrated_test = np.clip(calibrator.predict(model_test), 0.0, 1.0)
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, CALIBRATOR_PATH)

    policy = choose_fusion_policy(y_policy, calibrated_policy, behavioral_policy)
    stability = stability_gate(y_stability, calibrated_stability, behavioral_stability, policy)
    policy["enabled"] = bool(stability["enabled"])
    if not policy["enabled"]:
        policy["fallback_reason"] = "Fusion failed the later temporal stability gate; runtime uses calibrated model score."

    # Production fallback threshold is learned from the policy window, never from test.
    calibrated_model_review_threshold = float(np.quantile(calibrated_policy, 1.0 - TARGET_REVIEW_RATE))
    policy["calibrated_model_review_threshold"] = calibrated_model_review_threshold
    policy["production_review_rate"] = TARGET_REVIEW_RATE

    if policy["enabled"]:
        final_test_scores = policy["model_weight"] * calibrated_test + policy["behavioral_weight"] * behavioral_test
        final_test_threshold = float(policy["review_threshold"])
        production_policy = "risk_fusion"
    else:
        final_test_scores = calibrated_test.copy()
        final_test_threshold = calibrated_model_review_threshold
        production_policy = "calibrated_model_fallback"

    FUSION_CONFIG_PATH.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    model_metrics = classification_metrics(y_test, model_test, threshold=0.5)
    model_metrics.update(operating_point_metrics(y_test, model_test))
    calibrated_metrics = classification_metrics(y_test, calibrated_test, threshold=0.5)
    calibrated_metrics.update(operating_point_metrics(y_test, calibrated_test))
    behavioral_metrics = classification_metrics(y_test, behavioral_test, threshold=0.5)
    final_metrics = classification_metrics(y_test, final_test_scores, threshold=final_test_threshold)
    final_metrics.update(operating_point_metrics(y_test, final_test_scores))
    final_metrics.update({
        "production_policy": production_policy,
        "production_review_rate": TARGET_REVIEW_RATE,
        "production_threshold": final_test_threshold,
    })
    final_threshold_05_metrics = classification_metrics(y_test, final_test_scores, threshold=0.5)

    results: dict[str, Any] = {
        "evaluation_type": "temporal_holdout_with_class_safe_calibration_and_stability_gate",
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "validation_windows": {
            "calibration_rows": int(len(calibration_df)), "policy_rows": int(len(policy_df)),
            "stability_rows": int(len(stability_df)), "calibration_fraud_count": int(y_calibration.sum()),
            "policy_fraud_count": int(y_policy.sum()), "stability_fraud_count": int(y_stability.sum()),
        },
        "test_rows": int(len(test)), "test_fraud_count": int(y_test.sum()), "test_fraud_rate": float(y_test.mean()),
        "model": model_metrics, "calibrated_model": calibrated_metrics, "behavioral_rules_only": behavioral_metrics,
        "risk_fusion": {
            **final_metrics, "enabled": policy["enabled"], "model_weight": policy["model_weight"],
            "behavioral_weight": policy["behavioral_weight"], "review_threshold": policy["review_threshold"],
            "high_threshold": policy["high_threshold"], "calibrated_model_review_threshold": calibrated_model_review_threshold,
            "validation_policy": policy, "stability_gate": stability,
        },
        "final_production_policy": {
            "name": production_policy, "threshold": final_test_threshold, "review_rate": TARGET_REVIEW_RATE,
            "metrics": final_metrics, "threshold_0_5_comparison": final_threshold_05_metrics,
        },
        "model_calibration": calibration_metrics(y_test, model_test),
        "calibrated_model_calibration": calibration_metrics(y_test, calibrated_test),
        "model_thresholds": threshold_analysis(y_test, model_test),
        "model_review_capacity": review_capacity_analysis(y_test, model_test),
        "fusion_thresholds": threshold_analysis(y_test, final_test_scores),
        "fusion_review_capacity": review_capacity_analysis(y_test, final_test_scores),
        "cold_start_segments_model": segment_metrics(test, y_test, model_test, threshold=0.5),
        "cold_start_segments_fusion": segment_metrics(test, y_test, final_test_scores, threshold=final_test_threshold),
        "test_rule_trigger_rate": float(np.mean([bool(r) for r in test_rules])),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "system_evaluation.json").write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")
    pd.DataFrame(results["model_thresholds"]).to_csv(RESULTS_DIR / "threshold_analysis.csv", index=False)
    pd.DataFrame(results["fusion_thresholds"]).to_csv(RESULTS_DIR / "fusion_threshold_analysis.csv", index=False)
    (RESULTS_DIR / "cold_start_segments.json").write_text(json.dumps(_json_safe(results["cold_start_segments_fusion"]), indent=2), encoding="utf-8")
    (RESULTS_DIR / "validation_stability.json").write_text(json.dumps(_json_safe(stability), indent=2), encoding="utf-8")

    print("\n========== MODEL ==========")
    print(json.dumps(_json_safe(model_metrics), indent=2))
    print("\n========== CALIBRATED MODEL ==========")
    print(json.dumps(_json_safe(calibrated_metrics), indent=2))
    print("\n========== RULES ONLY ==========")
    print(json.dumps(_json_safe(behavioral_metrics), indent=2))
    print("\n========== SELECTED POLICY ==========")
    print(json.dumps(_json_safe(policy), indent=2))
    print("\n========== LATE VALIDATION STABILITY GATE ==========")
    print(json.dumps(_json_safe(stability), indent=2))
    print("\n========== FINAL PRODUCTION POLICY ==========")
    print(f"Policy: {production_policy}")
    print(f"Review rate: {TARGET_REVIEW_RATE:.0%}")
    print(f"Threshold: {final_test_threshold:.6f}")
    print(json.dumps(_json_safe(final_metrics), indent=2))
    print("\n========== TEST: THRESHOLD 0.5 COMPARISON ==========")
    print(json.dumps(_json_safe(final_threshold_05_metrics), indent=2))
    print("\nArtifacts:")
    print(CALIBRATOR_PATH)
    print(FUSION_CONFIG_PATH)
    print(RESULTS_DIR / "system_evaluation.json")
    print(RESULTS_DIR / "validation_stability.json")
    print("===============================================\n")
    return results


def main() -> None:
    argparse.ArgumentParser(description="Evaluate fraud model with class-safe temporal calibration and stable risk fusion.").parse_args()
    evaluate()


if __name__ == "__main__":
    main()
