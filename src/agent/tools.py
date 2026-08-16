from typing import Any

import pandas as pd


class FraudInvestigationTools:

    def __init__(
        self,
        transactions: pd.DataFrame
    ):
        self.transactions = transactions

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
        }


    def get_card_history(
    self,
    card1: int
    ) -> dict[str, Any]:

        rows = self.transactions[
        self.transactions["card1"] == card1
    ].sort_values("TransactionDT")

        if rows.empty:
            return {
            "error": "Card history not found."
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
    device_info: str
) -> dict[str, Any]:

        rows = self.transactions[
        self.transactions["DeviceInfo"]
        == device_info
    ]

        if rows.empty:
            return {
            "error": "Device history not found."
        }

        return {
        "device_info": device_info,
        "transaction_count": int(len(rows)),
        "unique_cards": int(
            rows["card1"].nunique()
        ),
    }