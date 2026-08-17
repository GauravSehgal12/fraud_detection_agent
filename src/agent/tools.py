from typing import Any

import pandas as pd

from src.services.risk_engine import RiskEngine


class FraudInvestigationTools:

    def __init__(
        self,
        transactions: pd.DataFrame,
        risk_assessments: dict | None = None,
        model=None
    ):
        """
        Tools used by the fraud investigation agent.

        Parameters
        ----------
        transactions:
            Investigation history dataframe.

        risk_assessments:
            Kept for backward compatibility.
            Dynamic risk assessment is now generated
            by RiskEngine.

        model:
            Loaded XGBoost fraud model.
        """

        self.transactions = transactions

        self.risk_assessments = (
            risk_assessments
            if risk_assessments is not None
            else {}
        )

        # -----------------------------------------
        # Dynamic Risk Engine
        # -----------------------------------------

        self.risk_engine = None

        if model is not None:
            self.risk_engine = RiskEngine(
                model=model
            )

    # =====================================================
    # TRANSACTION
    # =====================================================

    def get_transaction(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

        if self.transactions is None:
            return {
                "error": "Transaction data is not available."
            }

        if not isinstance(
            self.transactions,
            pd.DataFrame
        ):
            return {
                "error": "Transaction data must be a DataFrame."
            }

        matches = self.transactions[
            self.transactions["TransactionID"]
            == transaction_id
        ]

        if matches.empty:

            return {
                "error": "Transaction not found."
            }

        row = matches.iloc[0]

        result = {}

        for column in [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "card1",
            "DeviceInfo"
        ]:

            if column in row.index:

                value = row[column]

                if pd.isna(value):
                    value = None

                elif hasattr(value, "item"):
                    value = value.item()

                result[column] = value

        return result

    # =====================================================
    # DYNAMIC RISK ASSESSMENT
    # =====================================================

    def get_risk_assessment(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

        """
        Dynamically calculate fraud risk using:

        Transaction
             ↓
        21 engineered features
             ↓
        XGBoost
             ↓
        SHAP
             ↓
        Risk score
        """

        if self.risk_engine is None:

            return {
                "error": "Risk engine not initialized."
            }

        result = (
            self.risk_engine.calculate_risk(
                transaction_id
            )
        )

        return result

    # =====================================================
    # CARD HISTORY
    # =====================================================

    def get_card_history(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

        if self.transactions is None:
            return {
                "error": "Transaction data is not available."
            }

        transaction = self.get_transaction(
            transaction_id
        )

        if "error" in transaction:
            return transaction

        card1 = transaction.get(
            "card1"
        )

        transaction_dt = transaction.get(
            "TransactionDT"
        )

        if card1 is None:
            return {
                "error": "Card information unavailable."
            }

        history = self.transactions[
            (
                self.transactions["card1"]
                == card1
            )
            &
            (
                self.transactions["TransactionDT"]
                < transaction_dt
            )
        ].copy()

        if history.empty:

            return {
                "card1": int(card1),
                "transaction_count": 0,
                "average_amount": 0.0,
                "max_amount": 0.0,
                "min_amount": 0.0
            }

        amounts = pd.to_numeric(
            history["TransactionAmt"],
            errors="coerce"
        ).dropna()

        return {
            "card1": int(card1),

            "transaction_count": int(
                len(history)
            ),

            "average_amount": float(
                amounts.mean()
            ) if not amounts.empty else 0.0,

            "max_amount": float(
                amounts.max()
            ) if not amounts.empty else 0.0,

            "min_amount": float(
                amounts.min()
            ) if not amounts.empty else 0.0
        }

    # =====================================================
    # DEVICE HISTORY
    # =====================================================

    def get_device_history(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

        if self.transactions is None:
            return {
                "error": "Transaction data is not available."
            }

        transaction = self.get_transaction(
            transaction_id
        )

        if "error" in transaction:
            return transaction

        device_info = transaction.get(
            "DeviceInfo"
        )

        transaction_dt = transaction.get(
            "TransactionDT"
        )

        if (
            device_info is None
            or pd.isna(device_info)
        ):

            return {
                "device_info": None,
                "transaction_count": 0,
                "unique_cards": 0
            }

        history = self.transactions[
            (
                self.transactions["DeviceInfo"]
                == device_info
            )
            &
            (
                self.transactions["TransactionDT"]
                < transaction_dt
            )
        ].copy()

        if history.empty:

            return {
                "device_info": device_info,
                "transaction_count": 0,
                "unique_cards": 0
            }

        unique_cards = (
            history["card1"]
            .dropna()
            .nunique()
        )

        return {
            "device_info": device_info,

            "transaction_count": int(
                len(history)
            ),

            "unique_cards": int(
                unique_cards
            )
        }

    # =====================================================
    # COMPLETE INVESTIGATION DATA
    # =====================================================

    def collect_investigation_data(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

        """
        Collect all information needed by
        the investigation agent.
        """

        transaction = self.get_transaction(
            transaction_id
        )

        if "error" in transaction:
            return transaction

        risk = self.get_risk_assessment(
            transaction_id
        )

        card_history = self.get_card_history(
            transaction_id
        )

        device_history = self.get_device_history(
            transaction_id
        )

        return {
            "transaction": transaction,
            "risk_assessment": risk,
            "card_history": card_history,
            "device_history": device_history
        }