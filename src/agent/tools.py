from typing import Any
import pandas as pd


class FraudInvestigationTools:

    def __init__(
        self,
        transactions: pd.DataFrame,
        risk_assessments: dict[int, dict] | None = None
    ):
        self.transactions = transactions
        self.risk_assessments = risk_assessments or {}

    def get_transaction(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

        rows = self.transactions[
            self.transactions["TransactionID"]
            == transaction_id
        ]

        if rows.empty:
            return {
                "error": "Transaction not found."
            }

        row = rows.iloc[0]

        return {
            "TransactionID": int(
                row["TransactionID"]
            ),
            "TransactionAmt": float(
                row["TransactionAmt"]
            ),
            "TransactionDT": int(
                row["TransactionDT"]
            ),
            "card1": (
                int(row["card1"])
                if pd.notna(row["card1"])
                else None
            ),
            "DeviceInfo": (
                str(row["DeviceInfo"])
                if pd.notna(row["DeviceInfo"])
                else None
            ),
        }

    def get_card_history(
    self,
    card1: int,
    before_transaction_dt: int | None = None
) -> dict[str, Any]:

        rows = self.transactions[
        self.transactions["card1"] == card1
    ]

        if before_transaction_dt is not None:
            rows = rows[
            rows["TransactionDT"]
            < before_transaction_dt
        ]

        rows = rows.sort_values(
        "TransactionDT"
    )

        if rows.empty:
            return {
            "error": "No historical card activity found."
        }

        return {
        "card1": int(card1),
        "transaction_count": int(len(rows)),
        "average_amount": float(
            rows["TransactionAmt"].mean()
        ),
        "max_amount": float(
            rows["TransactionAmt"].max()
        ),
        "min_amount": float(
            rows["TransactionAmt"].min()
        ),
    }

    def get_device_history(
    self,
    device_info: str,
    before_transaction_dt: int | None = None
) -> dict[str, Any]:

        rows = self.transactions[
        self.transactions["DeviceInfo"]
        == device_info
    ]

        if before_transaction_dt is not None:
            rows = rows[
            rows["TransactionDT"]
            < before_transaction_dt
        ]

        if rows.empty:
            return {
            "error": "No historical device activity found."
        }

        return {
        "device_info": device_info,
        "transaction_count": int(len(rows)),
        "unique_cards": int(
            rows["card1"].nunique()
        ),
    }

    def get_risk_assessment(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

        assessment = self.risk_assessments.get(
            int(transaction_id)
        )

        if assessment is None:
            return {
                "error": "Risk assessment not found."
            }

        return assessment