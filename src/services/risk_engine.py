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
        Risk engine for fraud prediction.

        Supports two types of transactions:

        1. Existing transaction:
           calculate_risk(transaction_id)

        2. New transaction:
           calculate_risk(transaction_dict)

        For an existing transaction, features are read
        from the feature store.

        For a new transaction, features are generated
        using TransactionFeatureBuilder.
        """

        self.model = model

        # -------------------------------------------------
        # Existing feature store
        # -------------------------------------------------

        self.feature_store = pd.read_csv(
            FEATURE_STORE_PATH
        )

        # -------------------------------------------------
        # Optional feature builder
        # -------------------------------------------------

        self.feature_builder = (
            feature_builder
        )

        # -------------------------------------------------
        # SHAP explainer
        # -------------------------------------------------

        self.explainer = shap.TreeExplainer(
            self.model
        )

    # =====================================================
    # EXISTING TRANSACTION
    # =====================================================

    def get_transaction(
        self,
        transaction_id: int,
    ):

        matches = self.feature_store[
            self.feature_store[
                "TransactionID"
            ]
            == transaction_id
        ]

        if matches.empty:
            return None

        return matches.iloc[0]

    # =====================================================
    # BUILD MODEL INPUT
    # =====================================================

    def _prepare_model_input(
        self,
        features: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Convert generated features into the exact
        dataframe expected by XGBoost.
        """

        missing_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in features
        ]

        if missing_features:

            raise ValueError(
                "Missing model features: "
                f"{missing_features}"
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

        # -------------------------------------------------
        # Ensure numeric
        # -------------------------------------------------

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # -------------------------------------------------
        # Model cannot receive NaN
        # -------------------------------------------------

        X = X.fillna(0)

        return X

    # =====================================================
    # RISK CALCULATION
    # =====================================================

    def calculate_risk(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Calculate fraud risk.

        Accepts either:

        transaction ID:

            calculate_risk(3409570)

        OR a completely new transaction:

            calculate_risk({
                "TransactionID": 1234567,
                "TransactionDT": 10699419,
                "TransactionAmt": 87.302,
                "card1": 12730,
                ...
            })
        """

        # =================================================
        # CASE 1:
        # Existing transaction ID
        # =================================================

        if isinstance(
            transaction,
            int
        ):

            transaction_id = transaction

            existing_transaction = (
                self.get_transaction(
                    transaction_id
                )
            )

            if existing_transaction is None:

                return {
                    "error": (
                        "Transaction not found "
                        "in feature store."
                    ),
                    "transaction_id":
                        transaction_id,
                }

            features = {
                feature:
                    existing_transaction[
                        feature
                    ]
                for feature in FEATURE_COLUMNS
            }

        # =================================================
        # CASE 2:
        # New transaction dictionary
        # =================================================

        elif isinstance(
            transaction,
            dict
        ):

            transaction_id = transaction.get(
                "TransactionID"
            )

            if transaction_id is None:

                return {
                    "error": (
                        "TransactionID is required."
                    )
                }

            if self.feature_builder is None:

                return {
                    "error": (
                        "FeatureBuilder is not "
                        "configured in RiskEngine."
                    ),
                    "transaction_id":
                        transaction_id,
                }

            try:

                features = (
                    self.feature_builder.build(
                        transaction
                    )
                )

            except Exception as exc:

                return {
                    "error": (
                        "Failed to generate "
                        "transaction features."
                    ),
                    "transaction_id":
                        transaction_id,
                    "details":
                        str(exc),
                }

        # =================================================
        # INVALID INPUT
        # =================================================

        else:

            return {
                "error": (
                    "transaction must be either "
                    "an integer transaction ID "
                    "or a transaction dictionary."
                )
            }

        # =================================================
        # Prepare XGBoost input
        # =================================================

        try:

            X = self._prepare_model_input(
                features
            )

        except Exception as exc:

            return {
                "error": (
                    "Failed to prepare "
                    "model input."
                ),
                "transaction_id":
                    transaction_id,
                "details":
                    str(exc),
            }

        # =================================================
        # XGBOOST PREDICTION
        # =================================================

        try:

            risk_score = float(
                self.model.predict_proba(
                    X
                )[0, 1]
            )

        except Exception as exc:

            return {
                "error": (
                    "Model prediction failed."
                ),
                "transaction_id":
                    transaction_id,
                "details":
                    str(exc),
            }

        # =================================================
        # RISK LEVEL
        # =================================================

        if risk_score >= 0.90:

            risk_level = "HIGH"
            decision = "REVIEW"

        elif risk_score >= 0.70:

            risk_level = "MEDIUM"
            decision = "REVIEW"

        else:

            risk_level = "LOW"
            decision = "APPROVE"

        # =================================================
        # SHAP
        # =================================================

        try:

            shap_values = (
                self.explainer.shap_values(
                    X
                )
            )

            # ---------------------------------------------
            # Handle different SHAP output formats
            # ---------------------------------------------

            if isinstance(
                shap_values,
                list
            ):

                shap_values = (
                    shap_values[0]
                )

            shap_values = (
                shap_values[0]
            )

            evidence = []

            for (
                feature,
                value,
                shap_value
            ) in zip(
                FEATURE_COLUMNS,
                X.iloc[0].values,
                shap_values,
            ):

                evidence.append(
                    {
                        "feature":
                            feature,

                        "value":
                            float(value),

                        "shap_value":
                            float(
                                shap_value
                            ),

                        "abs_shap":
                            abs(
                                float(
                                    shap_value
                                )
                            ),
                    }
                )

            # ---------------------------------------------
            # Strongest contributors first
            # ---------------------------------------------

            evidence.sort(
                key=lambda item:
                    item["abs_shap"],
                reverse=True,
            )

            evidence = evidence[:5]

        except Exception as exc:

            # SHAP failure should not prevent
            # risk prediction from being returned.

            evidence = [
                {
                    "feature":
                        "SHAP_ERROR",

                    "value":
                        0.0,

                    "shap_value":
                        0.0,

                    "abs_shap":
                        0.0,

                    "error":
                        str(exc),
                }
            ]

        # =================================================
        # FINAL RESULT
        # =================================================

        return {

            "transaction_id":
                int(transaction_id),

            "risk_score":
                round(
                    risk_score,
                    4
                ),

            "risk_level":
                risk_level,

            "decision":
                decision,

            "evidence":
                evidence,
        }