import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "eval_results"

BASE_FEATURES = [
    "TransactionAmt", "TransactionAmt_log", "transaction_hour", "transaction_day",
    "has_identity", "missing_card_info", "missing_address", "missing_value_count",
    "card_transaction_count", "card_avg_amount", "amount_vs_card_avg", "new_card",
    "card_txn_count_1h", "card_txn_count_24h", "has_device_info", "device_profile_count",
    "device_profile_unique_cards", "new_device_profile", "card_device_transaction_count",
    "card_device_seen_before", "device_unique_cards_historical",
]
V2_FEATURES = BASE_FEATURES + [
    "amount_vs_card_avg_log", "card_amount_std", "amount_vs_card_zscore",
    "card_max_amount_ratio", "card_min_amount_ratio", "card_txn_rate_1h",
    "card_txn_rate_24h", "device_card_share_ratio", "amount_x_velocity_1h",
    "amount_x_velocity_24h", "amount_x_device_cards",
]


def add_v2_features(df):
    out = df.copy()
    eps = 1e-6
    out["amount_vs_card_avg_log"] = np.log1p(np.maximum(out["amount_vs_card_avg"], 0.0))
    out["card_amount_std"] = (out["card_avg_amount"] * 0.5).where(out["card_transaction_count"] > 1, 0.0)
    out["amount_vs_card_zscore"] = ((out["TransactionAmt"] - out["card_avg_amount"]) / (out["card_amount_std"] + eps)).clip(-20, 20).replace([np.inf, -np.inf], 0).fillna(0)
    out["card_max_amount_ratio"] = out["amount_vs_card_avg"].clip(0, 20)
    out["card_min_amount_ratio"] = out["amount_vs_card_avg"].where(out["card_transaction_count"] > 0, 0.0).clip(0, 20)
    out["card_txn_rate_1h"] = out["card_txn_count_1h"]
    out["card_txn_rate_24h"] = out["card_txn_count_24h"] / 24.0
    out["device_card_share_ratio"] = (out["card_device_transaction_count"] / (out["device_profile_count"] + eps)).clip(0, 1).fillna(0)
    out["amount_x_velocity_1h"] = out["TransactionAmt_log"] * np.log1p(out["card_txn_count_1h"])
    out["amount_x_velocity_24h"] = out["TransactionAmt_log"] * np.log1p(out["card_txn_count_24h"])
    out["amount_x_device_cards"] = out["TransactionAmt_log"] * np.log1p(out["device_profile_unique_cards"])
    return out


def recall_at_k(y, p, rate):
    k = max(1, int(len(y) * rate))
    idx = np.argsort(p)[::-1][:k]
    return float(np.asarray(y)[idx].sum() / max(np.asarray(y).sum(), 1))


def summarize(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    return {
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(auc(rc, pr)),
        "Recall_at_1pct": recall_at_k(y, p, 0.01),
        "Recall_at_5pct": recall_at_k(y, p, 0.05),
        "Recall_at_10pct": recall_at_k(y, p, 0.10),
    }


def main():
    features = pd.read_csv(DATA_DIR / "feature_store.csv")
    labels = pd.read_csv(RAW_DIR / "train_transaction.csv", usecols=["TransactionID", "isFraud"])
    df = features.merge(labels, on="TransactionID").sort_values("TransactionDT").reset_index(drop=True)
    df = add_v2_features(df)

    test = df.iloc[int(0.85 * len(df)):]
    y = test["isFraud"].to_numpy()

    baseline = xgb.XGBClassifier()
    baseline.load_model(str(MODELS_DIR / "fraud_xgboost.json"))
    v2 = xgb.XGBClassifier()
    v2.load_model(str(MODELS_DIR / "fraud_xgboost_v2.json"))

    base_p = baseline.predict_proba(test[BASE_FEATURES])[:, 1]
    v2_p = v2.predict_proba(test[V2_FEATURES])[:, 1]

    result = {"baseline": summarize(y, base_p), "v2": summarize(y, v2_p)}
    result["delta_v2_minus_baseline"] = {
        key: result["v2"][key] - result["baseline"][key] for key in result["baseline"]
    }

    EVAL_DIR.mkdir(exist_ok=True)
    with open(EVAL_DIR / "model_v2_comparison.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
