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

        Supports:

        1. Existing transaction:

            calculate_risk(3409570)

        2. New transaction:

            calculate_risk({
                "TransactionID": 999999999,
                "TransactionDT": 10699419,
                "TransactionAmt": 87.302,
                "card1": 12730,
                "DeviceInfo":
                    "LG-D320 Build/KOT49I.V10a"
            })

        Existing transactions use the persisted
        feature store.

        New transactions use TransactionFeatureBuilder.
        """

        self.model = model

        # =====================================================
        # EXISTING FEATURE STORE
        # =====================================================

        self.feature_store = pd.read_csv(
            FEATURE_STORE_PATH
        )

        # =====================================================
        # FEATURE BUILDER
        # =====================================================

        self.feature_builder = (
            feature_builder
        )

        # =====================================================
        # SHAP
        # =====================================================

        self.explainer = shap.TreeExplainer(
            self.model
        )

    # =========================================================
    # EXISTING TRANSACTION
    # =========================================================

    def get_transaction(
        self,
        transaction_id: int,
    ):
        """
        Get an existing transaction from the
        persisted feature store.
        """

        matches = self.feature_store[
            self.feature_store[
                "TransactionID"
            ]
            == transaction_id
        ]

        if matches.empty:

            return None

        return matches.iloc[0]

    # =========================================================
    # MODEL INPUT
    # =========================================================

    def _prepare_model_input(
        self,
        features: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Convert generated features into the exact
        dataframe expected by XGBoost.
        """

        # -----------------------------------------------------
        # Validate all 21 features
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Create model dataframe
        # -----------------------------------------------------

        X = pd.DataFrame(
            [
                {
                    feature: features[
                        feature
                    ]
                    for feature in FEATURE_COLUMNS
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        # -----------------------------------------------------
        # Convert everything to numeric
        # -----------------------------------------------------

        X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # -----------------------------------------------------
        # XGBoost cannot receive NaN
        # -----------------------------------------------------

        X = X.fillna(0)

        return X

    # =========================================================
    # RISK CALCULATION
    # =========================================================

    def calculate_risk(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Calculate fraud risk.

        Accepts either:

        Existing transaction:

            calculate_risk(3409570)

        OR new transaction:

            calculate_risk({
                "TransactionID": 999999999,
                "TransactionDT": 10699419,
                "TransactionAmt": 87.302,
                "card1": 12730,
                "DeviceInfo":
                    "LG-D320 Build/KOT49I.V10a"
            })
        """

        # =====================================================
        # CASE 1: EXISTING TRANSACTION
        # =====================================================

        if isinstance(
            transaction,
            int,
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

            # -------------------------------------------------
            # Extract the 21 persisted features
            # -------------------------------------------------

            features = {
                feature:
                    existing_transaction[
                        feature
                    ]
                for feature in FEATURE_COLUMNS
            }

        # =====================================================
        # CASE 2: NEW TRANSACTION
        # =====================================================

        elif isinstance(
            transaction,
            dict,
        ):

            transaction_id = transaction.get(
                "TransactionID"
            )

            if transaction_id is None:

                return {
                    "error":
                        "TransactionID is required."
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

                # -------------------------------------------------
                # Generate the exact same 21 features used
                # by the notebook.
                # -------------------------------------------------

                features = (
                    self.feature_builder.build(
                        transaction
                    )
                )

                # =================================================
                # DEBUG
                # =================================================

                print(
                    "\n"
                    "========== FEATURE DEBUG =========="
                )

                print(
                    "Transaction ID:",
                    transaction_id,
                )

                print(
                    "Input transaction:"
                )

                print(
                    transaction
                )

                print(
                    "\nGenerated features:"
                )

                for feature in FEATURE_COLUMNS:

                    print(
                        f"{feature}: "
                        f"{features.get(feature)}"
                    )

                print(
                    "\nMissing model features:"
                )

                print(
                    [
                        feature
                        for feature
                        in FEATURE_COLUMNS
                        if feature
                        not in features
                    ]
                )

                print(
                    "===================================\n"
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

        # =====================================================
        # INVALID INPUT
        # =====================================================

        else:

            return {
                "error": (
                    "transaction must be either "
                    "an integer transaction ID "
                    "or a transaction dictionary."
                )
            }

        # =====================================================
        # PREPARE MODEL INPUT
        # =====================================================

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

        # =====================================================
        # DEBUG MODEL INPUT
        # =====================================================

        if isinstance(
            transaction,
            dict,
        ):

            print(
                "\n"
                "========== MODEL INPUT =========="
            )

            print(X.to_string())

            print(
                "\nmissing_value_count:",
                X.iloc[0][
                    "missing_value_count"
                ],
            )

            print(
                "card_transaction_count:",
                X.iloc[0][
                    "card_transaction_count"
                ],
            )

            print(
                "device_profile_count:",
                X.iloc[0][
                    "device_profile_count"
                ],
            )

            print(
                "device_profile_unique_cards:",
                X.iloc[0][
                    "device_profile_unique_cards"
                ],
            )

            print(
                "device_unique_cards_historical:",
                X.iloc[0][
                    "device_unique_cards_historical"
                ],
            )

            print(
                "=================================\n"
            )

        # =====================================================
        # XGBOOST PREDICTION
        # =====================================================

        try:

            risk_score = float(
                self.model.predict_proba(
                    X
                )[0, 1]
            )

        except Exception as exc:

            return {
                "error":
                    "Model prediction failed.",
                "transaction_id":
                    transaction_id,
                "details":
                    str(exc),
            }

        # =====================================================
        # RISK LEVEL
        # =====================================================

        if risk_score >= 0.90:

            risk_level = "HIGH"
            decision = "REVIEW"

        elif risk_score >= 0.70:

            risk_level = "MEDIUM"
            decision = "REVIEW"

        else:

            risk_level = "LOW"
            decision = "APPROVE"

        # =====================================================
        # SHAP
        # =====================================================

        try:

            shap_values = (
                self.explainer.shap_values(
                    X
                )
            )

            # -------------------------------------------------
            # Handle different SHAP output formats
            # -------------------------------------------------

            if isinstance(
                shap_values,
                list,
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
                shap_value,
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

            # -------------------------------------------------
            # Strongest contributors first
            # -------------------------------------------------

            evidence.sort(
                key=lambda item:
                    item["abs_shap"],
                reverse=True,
            )

            evidence = evidence[:5]

        except Exception as exc:

            # -------------------------------------------------
            # SHAP failure should not prevent risk prediction
            # -------------------------------------------------

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

        # =====================================================
        # FINAL RESULT
        # =====================================================

        return {
            "transaction_id":
                int(transaction_id),

            "risk_score":
                round(
                    risk_score,
                    4,
                ),

            "risk_level":
                risk_level,

            "decision":
                decision,

            "evidence":
                evidence,
        }