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
WEIGHT_GRID = tuple(np.round(np.arange(0.50, 1.01, 0.05), 2))
THRESHOLD_GRID = tuple(np.round(np.arange(0.30, 0.91, 0.05), 2))


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
            "device_profile_unique_cards", "amount_vs_card_avg",
        )}
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
        result = rule_engine.evaluate(
            transaction=transaction,
            model_risk_score=None,
            cold_start_status=cold_start,
            features=features,
        )
        scores[position] = float(result["behavioral_risk_score"])
        all_rules.append(result["rules_triggered"])
    return scores, all_rules


def fit_calibrator(validation_scores: np.ndarray, y_validation: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(validation_scores, y_validation)
    return calibrator


def choose_fusion_policy(y_validation: np.ndarray, calibrated_model_validation: np.ndarray, behavioral_validation: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for model_weight in WEIGHT_GRID:
        behavioral_weight = 1.0 - model_weight
        fused = model_weight * calibrated_model_validation + behavioral_weight * behavioral_validation
        for threshold in THRESHOLD_GRID:
            review_rate = float((fused >= threshold).mean())
            if review_rate > MAX_REVIEW_RATE or review_rate <= 0:
                continue
            metrics = classification_metrics(y_validation, fused, threshold=threshold)
            candidate = {
                "model_weight": float(model_weight),
                "behavioral_weight": float(behavioral_weight),
                "review_threshold": float(threshold),
                "high_threshold": float(min(0.80, max(threshold + 0.15, 0.70))),
                "validation_f1": float(metrics["F1_Score"]),
                "validation_precision": float(metrics["Precision"]),
                "validation_recall": float(metrics["Recall"]),
                "validation_review_rate": review_rate,
            }
            if best is None or candidate["validation_f1"] > best["validation_f1"]:
                best = candidate
    return best or {
        "model_weight": 0.80, "behavioral_weight": 0.20, "review_threshold": 0.50,
        "high_threshold": 0.80, "validation_f1": 0.0,
        "validation_precision": 0.0, "validation_recall": 0.0, "validation_review_rate": 0.0,
    }


def evaluate() -> dict[str, Any]:
    print("\n========== FRAUD SYSTEM EVALUATION ==========")
    loaded = ModelLoader().load_all()
    model, features = loaded["model"], loaded["features"]
    df = load_feature_store()
    train_df, validation_df, test_df = time_based_split(df)
    validation = validation_df.reset_index(drop=True)
    test = test_df.reset_index(drop=True)
    y_validation = validation["isFraud"].to_numpy(dtype=int)
    y_test = test["isFraud"].to_numpy(dtype=int)

    print(f"Total rows: {len(df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")

    rule_engine = RuleEngine()
    model_validation = model_scores(model, validation, features)
    model_test = model_scores(model, test, features)
    behavioral_validation, _ = behavioral_scores(validation, rule_engine)
    behavioral_test, test_rules = behavioral_scores(test, rule_engine)

    calibrator = fit_calibrator(model_validation, y_validation)
    calibrated_validation = np.clip(calibrator.predict(model_validation), 0.0, 1.0)
    calibrated_test = np.clip(calibrator.predict(model_test), 0.0, 1.0)
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, CALIBRATOR_PATH)

    policy = choose_fusion_policy(y_validation, calibrated_validation, behavioral_validation)
    fusion_validation = policy["model_weight"] * calibrated_validation + policy["behavioral_weight"] * behavioral_validation
    fusion_test = policy["model_weight"] * calibrated_test + policy["behavioral_weight"] * behavioral_test
    FUSION_CONFIG_PATH.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    model_metrics = classification_metrics(y_test, model_test, threshold=0.5)
    model_metrics.update(operating_point_metrics(y_test, model_test))
    calibrated_metrics = classification_metrics(y_test, calibrated_test, threshold=0.5)
    calibrated_metrics.update(operating_point_metrics(y_test, calibrated_test))
    behavioral_metrics = classification_metrics(y_test, behavioral_test, threshold=0.5)
    behavioral_metrics.update(operating_point_metrics(y_test, behavioral_test))
    fusion_metrics = classification_metrics(y_test, fusion_test, threshold=policy["review_threshold"])
    fusion_metrics.update(operating_point_metrics(y_test, fusion_test))

    results: dict[str, Any] = {
        "evaluation_type": "temporal_holdout",
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "test_rows": int(len(test)),
        "test_fraud_count": int(y_test.sum()),
        "test_fraud_rate": float(y_test.mean()),
        "model": model_metrics,
        "calibrated_model": calibrated_metrics,
        "behavioral_rules_only": behavioral_metrics,
        "risk_fusion": {**fusion_metrics, **{k: policy[k] for k in ("model_weight", "behavioral_weight", "review_threshold", "high_threshold")}, "validation_policy": policy},
        "model_calibration": calibration_metrics(y_test, model_test),
        "calibrated_model_calibration": calibration_metrics(y_test, calibrated_test),
        "model_thresholds": threshold_analysis(y_test, model_test),
        "model_review_capacity": review_capacity_analysis(y_test, model_test),
        "fusion_thresholds": threshold_analysis(y_test, fusion_test),
        "fusion_review_capacity": review_capacity_analysis(y_test, fusion_test),
        "cold_start_segments_model": segment_metrics(test, y_test, model_test, threshold=0.5),
        "cold_start_segments_fusion": segment_metrics(test, y_test, fusion_test, threshold=policy["review_threshold"]),
        "test_rule_trigger_rate": float(np.mean([bool(r) for r in test_rules])),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "system_evaluation.json").write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")
    pd.DataFrame(results["model_thresholds"]).to_csv(RESULTS_DIR / "threshold_analysis.csv", index=False)
    pd.DataFrame(results["fusion_thresholds"]).to_csv(RESULTS_DIR / "fusion_threshold_analysis.csv", index=False)
    (RESULTS_DIR / "cold_start_segments.json").write_text(json.dumps(_json_safe(results["cold_start_segments_fusion"]), indent=2), encoding="utf-8")

    print("\n========== MODEL ==========")
    print(json.dumps(_json_safe(model_metrics), indent=2))
    print("\n========== CALIBRATED MODEL ==========")
    print(json.dumps(_json_safe(calibrated_metrics), indent=2))
    print("\n========== RULES ONLY ==========")
    print(json.dumps(_json_safe(behavioral_metrics), indent=2))
    print("\n========== CALIBRATED MODEL + BEHAVIOR ==========")
    print(json.dumps(_json_safe(fusion_metrics), indent=2))
    print("\n========== SELECTED VALIDATION POLICY ==========")
    print(json.dumps(_json_safe(policy), indent=2))
    print("\nArtifacts:")
    print(CALIBRATOR_PATH)
    print(FUSION_CONFIG_PATH)
    print(RESULTS_DIR / "system_evaluation.json")
    print("===============================================\n")
    return results


def main() -> None:
    argparse.ArgumentParser(description="Evaluate and tune fraud risk fusion.").parse_args()
    evaluate()


if __name__ == "__main__":
    main()
