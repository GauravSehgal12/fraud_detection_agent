import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

NEW_FEATURES = [
    "amount_vs_card_avg_log",
    "card_amount_std",
    "amount_vs_card_zscore",
    "card_max_amount_ratio",
    "card_min_amount_ratio",
    "card_txn_rate_1h",
    "card_txn_rate_24h",
    "device_card_share_ratio",
    "amount_x_velocity_1h",
    "amount_x_velocity_24h",
    "amount_x_device_cards",
]

FEATURES = BASE_FEATURES + NEW_FEATURES


def load_data():
    features = pd.read_csv(DATA_DIR / "feature_store.csv")
    labels = pd.read_csv(RAW_DIR / "train_transaction.csv", usecols=["TransactionID", "isFraud"])
    df = features.merge(labels, on="TransactionID", how="inner")
    return df.sort_values("TransactionDT").reset_index(drop=True)


def add_v2_features(df):
    out = df.copy()
    eps = 1e-6

    out["amount_vs_card_avg_log"] = np.log1p(np.maximum(out["amount_vs_card_avg"], 0.0))

    # The feature store may not contain historical standard deviation/max/min.
    # Build stable approximations from the available card-level statistics.
    out["card_amount_std"] = (
        out["card_avg_amount"] * 0.5
    ).where(out["card_transaction_count"] > 1, 0.0)

    out["amount_vs_card_zscore"] = (
        (out["TransactionAmt"] - out["card_avg_amount"])
        / (out["card_amount_std"] + eps)
    ).clip(-20, 20).replace([np.inf, -np.inf], 0).fillna(0)

    # Conservative ratios; zero when no usable historical amount exists.
    out["card_max_amount_ratio"] = out["amount_vs_card_avg"].clip(0, 20)
    out["card_min_amount_ratio"] = (
        out["amount_vs_card_avg"]
        .where(out["card_transaction_count"] > 0, 0.0)
        .clip(0, 20)
    )

    out["card_txn_rate_1h"] = out["card_txn_count_1h"] / 1.0
    out["card_txn_rate_24h"] = out["card_txn_count_24h"] / 24.0

    out["device_card_share_ratio"] = (
        out["card_device_transaction_count"]
        / (out["device_profile_count"] + eps)
    ).clip(0, 1).fillna(0)

    out["amount_x_velocity_1h"] = (
        out["TransactionAmt_log"] * np.log1p(out["card_txn_count_1h"])
    )
    out["amount_x_velocity_24h"] = (
        out["TransactionAmt_log"] * np.log1p(out["card_txn_count_24h"])
    )
    out["amount_x_device_cards"] = (
        out["TransactionAmt_log"] * np.log1p(out["device_profile_unique_cards"])
    )

    return out


def temporal_split(df):
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(auc(recall, precision)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1_Score": float(f1_score(y, pred, zero_division=0)),
        "False_Positive_Rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "False_Negative_Rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def main():
    EVAL_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    df = add_v2_features(load_data())
    train, val, test = temporal_split(df)

    X_train, y_train = train[FEATURES], train["isFraud"]
    X_val, y_val = val[FEATURES], val["isFraud"]
    X_test, y_test = test[FEATURES], test["isFraud"]

    # Keep the existing model untouched. This is an independently trained V2 candidate.
    model = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    result = {
        "model": "fraud_xgboost_v2",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "validation": metrics(y_val, val_probs),
        "test": metrics(y_test, test_probs),
    }

    model.save_model(MODELS_DIR / "fraud_xgboost_v2.json")
    with open(MODELS_DIR / "features_v2.json", "w") as f:
        json.dump(FEATURES, f, indent=2)
    with open(EVAL_DIR / "model_v2_evaluation.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
