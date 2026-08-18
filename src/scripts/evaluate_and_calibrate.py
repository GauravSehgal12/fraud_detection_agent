import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve, IsotonicRegression
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "eval_results"

FEATURE_COLUMNS = [
    "TransactionAmt",
    "TransactionAmt_log",
    "transaction_hour",
    "transaction_day",
    "has_identity",
    "missing_card_info",
    "missing_address",
    "missing_value_count",
    "card_transaction_count",
    "card_avg_amount",
    "amount_vs_card_avg",
    "new_card",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "has_device_info",
    "device_profile_count",
    "device_profile_unique_cards",
    "new_device_profile",
    "card_device_transaction_count",
    "card_device_seen_before",
    "device_unique_cards_historical",
]

def load_data():
    logger.info("Loading feature store...")
    feature_store_path = DATA_DIR / "feature_store.csv"
    if not feature_store_path.exists():
        raise FileNotFoundError(f"Feature store not found at {feature_store_path}")
    
    features_df = pd.read_csv(feature_store_path)
    
    logger.info("Loading labels from train_transaction.csv...")
    transaction_path = RAW_DIR / "train_transaction.csv"
    if not transaction_path.exists():
        raise FileNotFoundError(f"Transaction data not found at {transaction_path}")
        
    labels_df = pd.read_csv(transaction_path, usecols=["TransactionID", "isFraud"])
    
    logger.info("Merging features and labels...")
    merged_df = features_df.merge(labels_df, on="TransactionID", how="inner")
    
    # Sort chronologically
    merged_df = merged_df.sort_values(by="TransactionDT").reset_index(drop=True)
    return merged_df

def temporal_split(df: pd.DataFrame):
    logger.info("Performing temporal split (70% Train, 15% Val, 15% Test)...")
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Train set: {len(train_df)} rows")
    logger.info(f"Validation set: {len(val_df)} rows")
    logger.info(f"Test set: {len(test_df)} rows")
    
    return train_df, val_df, test_df

def calculate_recall_at_k(y_true, y_probs, review_rates):
    recalls = {}
    n = len(y_true)
    total_fraud = y_true.sum()
    
    if total_fraud == 0:
        return {r: 0.0 for r in review_rates}
        
    sorted_indices = np.argsort(y_probs)[::-1]
    y_true_sorted = y_true.iloc[sorted_indices].values
    
    for rate in review_rates:
        k = int(n * rate)
        if k == 0:
            recalls[rate] = 0.0
            continue
        
        fraud_caught = y_true_sorted[:k].sum()
        recalls[rate] = float(fraud_caught / total_fraud)
        
    return recalls

def evaluate_model():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_data()
    train_df, val_df, test_df = temporal_split(df)
    
    logger.info("Loading XGBoost model...")
    model_path = MODELS_DIR / "fraud_xgboost.json"
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Ensure correct feature order
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["isFraud"]
    
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["isFraud"]
    
    logger.info("Generating predictions...")
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # ---------------------------------------------------------
    # Risk Calibration
    # ---------------------------------------------------------
    logger.info("Calibrating model using Isotonic Regression...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_probs, y_val)
    
    test_probs_calibrated = calibrator.predict(test_probs)
    
    # ---------------------------------------------------------
    # Calculate Metrics
    # ---------------------------------------------------------
    logger.info("Calculating metrics...")
    
    # Threshold for discrete metrics
    threshold = 0.5
    y_pred = (test_probs_calibrated >= threshold).astype(int)
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_probs_calibrated)
    pr_auc = auc(recall_curve, precision_curve)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, test_probs_calibrated)
    
    # Precision, Recall, F1
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Recall at review rates
    review_rates = [0.01, 0.05, 0.10]
    recalls_at_k = calculate_recall_at_k(y_test, test_probs_calibrated, review_rates)
    
    # Brier Score
    brier_uncalibrated = brier_score_loss(y_test, test_probs)
    brier_calibrated = brier_score_loss(y_test, test_probs_calibrated)
    
    metrics = {
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1_Score": float(f1),
        "False_Positive_Rate": float(fpr),
        "False_Negative_Rate": float(fnr),
        "Confusion_Matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "Recall_at_1%_review": float(recalls_at_k[0.01]),
        "Recall_at_5%_review": float(recalls_at_k[0.05]),
        "Recall_at_10%_review": float(recalls_at_k[0.10]),
        "Brier_Score_Uncalibrated": float(brier_uncalibrated),
        "Brier_Score_Calibrated": float(brier_calibrated),
    }
    
    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------
    logger.info("Generating plots...")
    plt.figure(figsize=(10, 10))
    
    # Calibration Curve Plot
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, test_probs, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, test_probs_calibrated, n_bins=10)
    
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.plot(prob_pred_uncal, prob_true_uncal, marker='.', label='Uncalibrated')
    plt.plot(prob_pred_cal, prob_true_cal, marker='.', label='Isotonic Calibration')
    plt.title('Calibration Curves')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.savefig(EVAL_DIR / "calibration_curve.png")
    plt.close()
    
    # Save metrics
    metrics_path = EVAL_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Evaluation complete. Results saved to {EVAL_DIR}")
    for k, v in metrics.items():
        if isinstance(v, dict):
            logger.info(f"{k}: {v}")
        else:
            logger.info(f"{k}: {v:.4f}")

if __name__ == "__main__":
    evaluate_model()
