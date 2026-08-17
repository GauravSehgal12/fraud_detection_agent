from pathlib import Path
from typing import Any

import pandas as pd
import shap


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
    ):

        self.model = model

        self.feature_store = pd.read_csv(
            FEATURE_STORE_PATH
        )

        self.explainer = shap.TreeExplainer(
            self.model
        )

    def get_transaction(
        self,
        transaction_id: int
    ):

        matches = self.feature_store[
            self.feature_store["TransactionID"]
            == transaction_id
        ]

        if matches.empty:
            return None

        return matches.iloc[0]

    def calculate_risk(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

        transaction = self.get_transaction(
            transaction_id
        )

        if transaction is None:

            return {
                "error": "Transaction not found."
            }

        

        X = pd.DataFrame(
            [
                transaction[
                    FEATURE_COLUMNS
                ].values
            ],
            columns=FEATURE_COLUMNS
        )

      
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        

        X = X.fillna(0)

        

        risk_score = float(
            self.model.predict_proba(X)[0, 1]
        )

       

        if risk_score >= 0.90:

            risk_level = "HIGH"
            decision = "REVIEW"

        elif risk_score >= 0.70:

            risk_level = "MEDIUM"
            decision = "REVIEW"

        else:

            risk_level = "LOW"
            decision = "APPROVE"

      

        shap_values = self.explainer.shap_values(
            X
        )

        if isinstance(shap_values, list):

            shap_values = shap_values[0]

        shap_values = shap_values[0]

        evidence = []

        for feature, value, shap_value in zip(
            FEATURE_COLUMNS,
            X.iloc[0].values,
            shap_values
        ):

            evidence.append(
                {
                    "feature": feature,
                    "value": float(value),
                    "shap_value": float(shap_value),
                    "abs_shap": abs(
                        float(shap_value)
                    )
                }
            )

       
        evidence.sort(
            key=lambda item: item["abs_shap"],
            reverse=True
        )

        
        evidence = evidence[:5]

        return {
            "transaction_id": transaction_id,
            "risk_score": round(
                risk_score,
                4
            ),
            "risk_level": risk_level,
            "decision": decision,
            "evidence": evidence
        }