from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.split import time_based_split
from src.services.decision_engine import DecisionEngine
from src.services.model_loader import ModelLoader
from src.services.rule_engine import RuleEngine

from .calibration import calibration_metrics
from .metrics import classification_metrics, operating_point_metrics
from .segment_analysis import segment_metrics
from .threshold_analysis import review_capacity_analysis, threshold_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store.csv"
TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw"/"train_transaction.csv"
RESULTS_DIR = PROJECT_ROOT / "eval_results"


def _model_risk_level(score: float, high: float = 0.90, medium: float = 0.70) -> str:
    if score >= high:
        return "HIGH"
    if score >= medium:
        return "MEDIUM"
    return "LOW"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_feature_store() -> pd.DataFrame:
    """
    Load the model feature store and attach the ground-truth fraud label.

    The feature store intentionally does not need to contain isFraud because
    labels are not model input features. For evaluation, the label is loaded
    separately from the original IEEE-CIS train_transaction.csv and joined
    by TransactionID.
    """
    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Feature store not found: {FEATURE_STORE_PATH}"
        )

    if not TRANSACTION_PATH.exists():
        raise FileNotFoundError(
            f"Training transaction data not found: {TRANSACTION_PATH}. "
            "The evaluator needs train_transaction.csv only for ground-truth labels."
        )

    print(f"Loading feature store: {FEATURE_STORE_PATH}")
    features_df = pd.read_csv(FEATURE_STORE_PATH)

    required_features = {
        "TransactionID",
        "TransactionDT",
        "new_card",
        "new_device_profile",
        "card_txn_count_1h",
        "card_txn_count_24h",
        "device_profile_unique_cards",
        "amount_vs_card_avg",
    }
    missing = sorted(required_features - set(features_df.columns))
    if missing:
        raise ValueError(
            "Feature store is missing columns required for evaluation: "
            + ", ".join(missing)
        )

    print(f"Feature store shape: {features_df.shape}")
    print(f"Loading labels from: {TRANSACTION_PATH}")

    # Only read the two columns needed for evaluation. This avoids loading
    # the complete 394-column transaction table into memory unnecessarily.
    labels_df = pd.read_csv(
        TRANSACTION_PATH,
        usecols=["TransactionID", "isFraud"],
    )

    if labels_df["TransactionID"].duplicated().any():
        raise ValueError("Duplicate TransactionID values found in label data.")

    df = features_df.merge(
        labels_df,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    missing_labels = int(df["isFraud"].isna().sum())
    if missing_labels:
        raise ValueError(
            f"{missing_labels:,} feature-store rows have no matching isFraud label."
        )

    df["isFraud"] = pd.to_numeric(df["isFraud"], errors="raise").astype(int)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    print(f"Evaluation dataframe shape: {df.shape}")
    print(f"Fraud rows: {int(df['isFraud'].sum()):,}")
    print("Label successfully attached to feature store.")

    return df


def model_scores(model: Any, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(
            "Feature store is missing model features: " + ", ".join(missing)
        )

    X = df[features].copy().apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return model.predict_proba(X)[:, 1]


def behavioral_decisions(
    df: pd.DataFrame,
    scores: np.ndarray,
    rule_engine: RuleEngine,
    decision_engine: DecisionEngine,
) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, Any]]]]:
    behavioral_scores = np.zeros(len(df), dtype=float)
    decisions = np.zeros(len(df), dtype=int)
    all_rules: list[list[dict[str, Any]]] = []

    for position, (_, row) in enumerate(df.iterrows()):
        features = {
            key: row.get(key, 0)
            for key in (
                "TransactionAmt",
                "card_txn_count_1h",
                "card_txn_count_24h",
                "device_profile_unique_cards",
                "amount_vs_card_avg",
            )
        }

        cold_start_status = {
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

        behavioral = rule_engine.evaluate(
            transaction=transaction,
            model_risk_score=None,
            cold_start_status=cold_start_status,
            features=features,
        )
        behavioral_scores[position] = behavioral["behavioral_risk_score"]

        model_score = float(scores[position])
        model_level = _model_risk_level(model_score)
        hybrid = rule_engine.evaluate(
            transaction=transaction,
            model_risk_score=model_score,
            cold_start_status=cold_start_status,
            features=features,
        )

        action = decision_engine.decide(
            model_risk_score=model_score,
            model_risk_level=model_level,
            behavioral_risk_level=hybrid["behavioral_risk_level"],
            cold_start_status=cold_start_status,
            rules_triggered=hybrid["rules_triggered"],
        )

        decisions[position] = int(action == "REVIEW")
        all_rules.append(hybrid["rules_triggered"])

    return behavioral_scores, decisions, all_rules


def evaluate() -> dict[str, Any]:
    print("\n========== FRAUD SYSTEM EVALUATION ==========")

    loaded = ModelLoader().load_all()
    model = loaded["model"]
    features = loaded["features"]

    df = load_feature_store()
    train_df, validation_df, test_df = time_based_split(df)
    test = test_df.reset_index(drop=True)
    y_test = test["isFraud"].to_numpy(dtype=int)

    print(f"Total rows: {len(df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Test fraud rate: {y_test.mean():.6f}")

    scores = model_scores(model, test, features)

    rule_engine = RuleEngine()
    decision_engine = DecisionEngine()
    behavioral_scores, decisions, all_rules = behavioral_decisions(
        test, scores, rule_engine, decision_engine
    )

    model_metrics = classification_metrics(y_test, scores, threshold=0.5)
    model_metrics.update(operating_point_metrics(y_test, scores))

    behavioral_metrics = classification_metrics(
        y_test, behavioral_scores, threshold=0.5
    )
    behavioral_metrics.update(
        operating_point_metrics(y_test, behavioral_scores)
    )

    # APPROVE/REVIEW is a binary production decision, not a probability.
    hybrid_metrics = classification_metrics(
        y_test, decisions.astype(float), threshold=0.5
    )

    results: dict[str, Any] = {
        "evaluation_type": "temporal_holdout",
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "test_rows": int(len(test)),
        "test_fraud_count": int(y_test.sum()),
        "test_fraud_rate": float(y_test.mean()),
        "model": model_metrics,
        "behavioral_rules_only": behavioral_metrics,
        "model_plus_rules_decision": hybrid_metrics,
        "model_calibration": calibration_metrics(y_test, scores),
        "model_thresholds": threshold_analysis(y_test, scores),
        "model_review_capacity": review_capacity_analysis(y_test, scores),
        "cold_start_segments": segment_metrics(test, y_test, scores),
        "decision_review_rate": float(decisions.mean()),
        "rule_trigger_rate": float(
            np.mean([len(rules) > 0 for rules in all_rules])
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "system_evaluation.json"
    output_path.write_text(
        json.dumps(_json_safe(results), indent=2),
        encoding="utf-8",
    )

    threshold_path = RESULTS_DIR / "threshold_analysis.csv"
    pd.DataFrame(results["model_thresholds"]).to_csv(
        threshold_path,
        index=False,
    )

    segment_path = RESULTS_DIR / "cold_start_segments.json"
    segment_path.write_text(
        json.dumps(
            _json_safe(results["cold_start_segments"]),
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== MODEL ==========")
    print(json.dumps(_json_safe(model_metrics), indent=2))

    print("\n========== RULES ONLY ==========")
    print(json.dumps(_json_safe(behavioral_metrics), indent=2))

    print("\n========== MODEL + RULES ==========")
    print(json.dumps(_json_safe(hybrid_metrics), indent=2))

    print("\n========== OUTPUTS ==========")
    print(output_path)
    print(threshold_path)
    print(segment_path)
    print("========================================\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the fraud detection system."
    )
    parser.parse_args()
    evaluate()


if __name__ == "__main__":
    main()
