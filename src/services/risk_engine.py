from pathlib import Path
from typing import Any

import pandas as pd
import shap

from src.services.transaction_feature_builder import (
    TransactionFeatureBuilder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_STORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "feature_store.csv"
)


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


class RiskEngine:

    def __init__(
        self,
        model,
        feature_builder: TransactionFeatureBuilder | None = None,
    ):
        """
        Risk engine for model risk prediction and feature explainability.
        """
        self.model = model

        # Feature store for existing historical transactions
        if FEATURE_STORE_PATH.exists():
            self.feature_store = pd.read_csv(FEATURE_STORE_PATH)
        else:
            self.feature_store = pd.DataFrame(columns=["TransactionID", *FEATURE_COLUMNS])

        self.feature_builder = feature_builder

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

    def get_transaction(self, transaction_id: int):
        matches = self.feature_store[
            self.feature_store["TransactionID"] == transaction_id
        ]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _prepare_model_input(
        self,
        features: dict[str, Any],
    ) -> pd.DataFrame:
        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in features
        ]

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        X = pd.DataFrame(
            [
                {
                    feature: features[feature]
                    for feature in FEATURE_COLUMNS
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        return X

    def calculate_risk(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Calculate model risk score, risk level, model decision, and SHAP evidence.
        """
        input_completeness = "COMPLETE"

        if isinstance(transaction, int):
            transaction_id = transaction
            existing_transaction = self.get_transaction(transaction_id)

            if existing_transaction is None:
                return {
                    "error": f"Transaction not found in feature store.",
                    "transaction_id": transaction_id,
                }

            features = {
                feature: existing_transaction[feature]
                for feature in FEATURE_COLUMNS
            }
            input_completeness = "COMPLETE"

        elif isinstance(transaction, dict):
            transaction_id = transaction.get("TransactionID")
            if transaction_id is None:
                return {"error": "TransactionID is required."}

            if self.feature_builder is None:
                return {
                    "error": "FeatureBuilder is not configured in RiskEngine.",
                    "transaction_id": transaction_id,
                }

            # Check completeness: If fewer than 20 raw fields supplied, mark PARTIAL
            if len(transaction) < 20 and not transaction.get("synthetic", False):
                input_completeness = "PARTIAL"
            else:
                input_completeness = "COMPLETE"

            try:
                features = self.feature_builder.build(transaction)
            except Exception as exc:
                return {
                    "error": "Failed to generate transaction features.",
                    "transaction_id": transaction_id,
                    "details": str(exc),
                }

        else:
            return {
                "error": "transaction must be either an integer transaction ID or a transaction dictionary."
            }

        try:
            X = self._prepare_model_input(features)
        except Exception as exc:
            return {
                "error": "Failed to prepare model input.",
                "transaction_id": transaction_id,
                "details": str(exc),
            }

        try:
            risk_score = float(self.model.predict_proba(X)[0, 1])
        except Exception as exc:
            return {
                "error": "Model prediction failed.",
                "transaction_id": transaction_id,
                "details": str(exc),
            }

        if risk_score >= 0.90:
            risk_level = "HIGH"
            decision = "REVIEW"
        elif risk_score >= 0.70:
            risk_level = "MEDIUM"
            decision = "REVIEW"
        else:
            risk_level = "LOW"
            decision = "APPROVE"

        # SHAP calculation
        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = shap_values[0]

            evidence = []
            for feature, value, shap_val in zip(
                FEATURE_COLUMNS,
                X.iloc[0].values,
                shap_values,
            ):
                evidence.append(
                    {
                        "feature": feature,
                        "value": float(value),
                        "shap_value": float(shap_val),
                        "abs_shap": abs(float(shap_val)),
                    }
                )

            evidence.sort(key=lambda item: item["abs_shap"], reverse=True)
            evidence = evidence[:5]
        except Exception as exc:
            evidence = [
                {
                    "feature": "SHAP_ERROR",
                    "value": 0.0,
                    "shap_value": 0.0,
                    "abs_shap": 0.0,
                    "error": str(exc),
                }
            ]

        return {
            "transaction_id": int(transaction_id),
            "model_risk": {
                "score": round(risk_score, 4),
                "level": risk_level,
                "decision": decision,
            },
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "decision": decision,
            "input_completeness": input_completeness,
            "evidence": evidence,
            "features": features,
        }