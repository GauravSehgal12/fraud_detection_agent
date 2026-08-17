from typing import Any
import pandas as pd


class ColdStartDetector:

    def __init__(
        self,
        transactions: pd.DataFrame,
        identity: pd.DataFrame | None = None,
    ):
        """
        Cold-start detector for checking historical availability of cards,
        devices, and card-device pairs.

        Historical records must always satisfy:
            historical TransactionDT < current TransactionDT
        """
        self.transactions = transactions
        self.identity = identity

        # Pre-process device history view if identity is provided
        self._prepare_device_view()

    def _prepare_device_view(self):
        """
        Build a historical view linking TransactionID, TransactionDT, card1, and DeviceInfo.
        """
        if (
            self.identity is not None
            and "TransactionID" in self.identity.columns
            and "DeviceInfo" in self.identity.columns
        ):
            device_df = self.identity[["TransactionID", "DeviceInfo"]].dropna(subset=["DeviceInfo"]).drop_duplicates(
                subset=["TransactionID"], keep="first"
            )
            txn_df = self.transactions[["TransactionID", "TransactionDT", "card1"]].drop_duplicates(
                subset=["TransactionID"], keep="first"
            )
            self.device_txn_map = device_df.merge(
                txn_df, on="TransactionID", how="inner"
            )
        elif "DeviceInfo" in self.transactions.columns:
            self.device_txn_map = self.transactions[
                ["TransactionID", "TransactionDT", "card1", "DeviceInfo"]
            ].dropna(subset=["DeviceInfo"])
        else:
            self.device_txn_map = pd.DataFrame(
                columns=["TransactionID", "TransactionDT", "card1", "DeviceInfo"]
            )

    def detect(self, transaction: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze transaction to determine cold-start status.

        Expected fields in transaction dict:
        - card1
        - DeviceInfo (optional)
        - TransactionDT
        """
        card1 = transaction.get("card1")
        device_info = transaction.get("DeviceInfo")
        transaction_dt = transaction.get("TransactionDT")

        # Fallback values if basic info missing
        if card1 is None or transaction_dt is None:
            return {
                "is_new_card": True,
                "is_new_device": True,
                "is_new_card_device_pair": True,
                "card_history_available": False,
                "device_history_available": False,
                "card_device_history_available": False,
            }

        try:
            card1_val = float(card1)
            dt_val = float(transaction_dt)
        except (ValueError, TypeError):
            return {
                "is_new_card": True,
                "is_new_device": True,
                "is_new_card_device_pair": True,
                "card_history_available": False,
                "device_history_available": False,
                "card_device_history_available": False,
            }

        # 1. Card history check (TransactionDT < dt_val)
        card_history_matches = self.transactions[
            (pd.to_numeric(self.transactions["card1"], errors="coerce") == card1_val)
            & (pd.to_numeric(self.transactions["TransactionDT"], errors="coerce") < dt_val)
        ]
        card_history_available = not card_history_matches.empty
        is_new_card = not card_history_available

        # 2. Device history check (TransactionDT < dt_val)
        device_history_available = False
        if device_info is not None and pd.notna(device_info) and str(device_info).strip() != "" and str(device_info).upper() != "UNKNOWN-DEVICE":
            device_matches = self.device_txn_map[
                (self.device_txn_map["DeviceInfo"] == str(device_info))
                & (pd.to_numeric(self.device_txn_map["TransactionDT"], errors="coerce") < dt_val)
            ]
            device_history_available = not device_matches.empty

        is_new_device = not device_history_available

        # 3. Card-Device pair history check (TransactionDT < dt_val)
        card_device_history_available = False
        if card_history_available and device_history_available:
            pair_matches = self.device_txn_map[
                (pd.to_numeric(self.device_txn_map["card1"], errors="coerce") == card1_val)
                & (self.device_txn_map["DeviceInfo"] == str(device_info))
                & (pd.to_numeric(self.device_txn_map["TransactionDT"], errors="coerce") < dt_val)
            ]
            card_device_history_available = not pair_matches.empty

        is_new_card_device_pair = not card_device_history_available

        return {
            "is_new_card": is_new_card,
            "is_new_device": is_new_device,
            "is_new_card_device_pair": is_new_card_device_pair,
            "card_history_available": card_history_available,
            "device_history_available": device_history_available,
            "card_device_history_available": card_device_history_available,
        }
