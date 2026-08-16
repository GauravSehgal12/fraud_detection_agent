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