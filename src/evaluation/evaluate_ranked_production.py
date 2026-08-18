from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.split import time_based_split
from src.services.model_loader import ModelLoader
from src.services.rule_engine import RuleEngine
from src.evaluation.metrics import classification_metrics, operating_point_metrics
from src.evaluation.review_policy import rank_review_metrics, score_threshold_at_review_capacity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store.csv"
TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
CALIBRATOR_PATH = PROJECT_ROOT / "models" / "isotonic_calibrator.joblib"
FUSION_CONFIG_PATH = PROJECT_ROOT / "models" / "risk_fusion_config.json"
RESULTS_PATH = PROJECT_ROOT / "eval_results" / "ranked_production_evaluation.json"
TARGET_REVIEW_RATE = 0.05


def model_scores(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = df[features].copy().apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return model.predict_proba(X)[:, 1]


def behavioral_scores(df: pd.DataFrame, rule_engine: RuleEngine) -> np.ndarray:
    scores = np.zeros(len(df), dtype=float)
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
    return scores


def main() -> None:
    print("\n========== EXACT RANK-BASED PRODUCTION EVALUATION ==========")

    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(f"Feature store not found: {FEATURE_STORE_PATH}")
    if not TRANSACTION_PATH.exists():
        raise FileNotFoundError(f"Training labels not found: {TRANSACTION_PATH}")
    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(f"Calibrator not found: {CALIBRATOR_PATH}. Run evaluation.evaluate first.")

    loaded = ModelLoader().load_all()
    model = loaded["model"]
    features = loaded["features"]

    feature_df = pd.read_csv(FEATURE_STORE_PATH)
    labels = pd.read_csv(TRANSACTION_PATH, usecols=["TransactionID", "isFraud"])
    df = feature_df.merge(labels, on="TransactionID", how="inner", validate="one_to_one")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    _, _, test_df = time_based_split(df)
    test_df = test_df.reset_index(drop=True)
    y_test = test_df["isFraud"].to_numpy(dtype=int)

    raw_model_scores = model_scores(model, test_df, features)
    calibrator = joblib.load(CALIBRATOR_PATH)
    calibrated_scores = np.clip(calibrator.predict(raw_model_scores), 0.0, 1.0)

    config = {}
    if FUSION_CONFIG_PATH.exists():
        config = json.loads(FUSION_CONFIG_PATH.read_text(encoding="utf-8"))

    production_policy = "calibrated_model_fallback"
    final_scores = calibrated_scores

    if bool(config.get("enabled", False)):
        rule_engine = RuleEngine()
        behavioral = behavioral_scores(test_df, rule_engine)
        model_weight = float(config.get("model_weight", 1.0))
        behavioral_weight = float(config.get("behavioral_weight", 0.0))
        final_scores = model_weight * calibrated_scores + behavioral_weight * behavioral
        production_policy = "risk_fusion"

    ranked = rank_review_metrics(y_test, final_scores, TARGET_REVIEW_RATE)
    cutoff = score_threshold_at_review_capacity(final_scores, TARGET_REVIEW_RATE)

    # Threshold metrics are included only for comparison. Production metrics use exact top-k ranking.
    threshold_metrics = classification_metrics(y_test, final_scores, threshold=cutoff)
    threshold_metrics.update(operating_point_metrics(y_test, final_scores))

    result = {
        "production_policy": production_policy,
        "target_review_rate": TARGET_REVIEW_RATE,
        "actual_review_rate": ranked["review_rate"],
        "review_count": ranked["review_count"],
        "precision_at_production_capacity": ranked["precision"],
        "recall_at_production_capacity": ranked["recall"],
        "reporting_cutoff": cutoff,
        "threshold_metrics_comparison": threshold_metrics,
        "test_rows": len(test_df),
        "test_frauds": int(y_test.sum()),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n========== FINAL PRODUCTION POLICY ==========")
    print(f"Policy: {production_policy}")
    print(f"Target review rate: {TARGET_REVIEW_RATE:.2%}")
    print(f"Actual review rate: {ranked['review_rate']:.2%}")
    print(f"Review count: {int(ranked['review_count']):,}")
    print(f"Precision @ production capacity: {ranked['precision']:.4%}")
    print(f"Recall @ production capacity: {ranked['recall']:.4%}")
    print(f"Reporting cutoff: {cutoff:.6f}")
    print("\nThreshold metrics comparison:")
    print(json.dumps(threshold_metrics, indent=2))
    print(f"\nSaved: {RESULTS_PATH}")
    print("================================================")


if __name__ == "__main__":
    main()
