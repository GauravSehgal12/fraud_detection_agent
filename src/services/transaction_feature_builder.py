# src/services/transaction_feature_builder.py

from typing import Any

import numpy as np
import pandas as pd


class TransactionFeatureBuilder:

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

    CARD_COLUMNS = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
    ]

    ADDRESS_COLUMNS = [
        "addr1",
        "addr2",
    ]

    def __init__(
        self,
        historical_df: pd.DataFrame,
    ):
        """
        historical_df:
            Historical transactions used to calculate
            card/device behavioral features.

        IMPORTANT:
            Only transactions before the new transaction
            should be used for behavioral features.
        """

        if not isinstance(
            historical_df,
            pd.DataFrame
        ):
            raise TypeError(
                "historical_df must be a pandas DataFrame"
            )

        required_columns = [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "card1",
        ]

        missing = [
            column
            for column in required_columns
            if column not in historical_df.columns
        ]

        if missing:
            raise ValueError(
                f"Historical dataframe is missing columns: "
                f"{missing}"
            )

        self.historical_df = historical_df.copy()

    # =====================================================
    # BASIC FEATURES
    # =====================================================

    def _basic_features(
        self,
        transaction: dict[str, Any],
    ) -> dict[str, float]:

        amount = float(
            transaction.get(
                "TransactionAmt",
                0.0
            )
        )

        transaction_dt = float(
            transaction.get(
                "TransactionDT",
                0.0
            )
        )

        # TransactionDT in IEEE-CIS is measured
        # in seconds from the reference start.
        transaction_hour = (
            int(transaction_dt // 3600)
            % 24
        )

        transaction_day = int(
            transaction_dt // 86400
        )

        # -----------------------------------------
        # Identity
        # -----------------------------------------

        identity_columns = [
            column
            for column in transaction.keys()
            if column.startswith("id_")
        ]

        has_identity = int(
            any(
                transaction.get(column) is not None
                and not pd.isna(
                    transaction.get(column)
                )
                for column in identity_columns
            )
        )

        # -----------------------------------------
        # Missing card information
        # -----------------------------------------

        card_values = [
            transaction.get(column)
            for column in self.CARD_COLUMNS
        ]

        missing_card_info = int(
            any(
                value is None
                or pd.isna(value)
                for value in card_values
            )
        )

        # -----------------------------------------
        # Missing address
        # -----------------------------------------

        address_values = [
            transaction.get(column)
            for column in self.ADDRESS_COLUMNS
        ]

        missing_address = int(
            any(
                value is None
                or pd.isna(value)
                for value in address_values
            )
        )

        # -----------------------------------------
        # Missing value count
        # -----------------------------------------

        # For a complete raw transaction this counts
        # missing values in the supplied transaction.
        #
        # If the transaction contains only the minimum
        # real-time fields, pass the complete expected
        # transaction schema to build().

        missing_value_count = sum(
            value is None
            or pd.isna(value)
            for value in transaction.values()
        )

        return {
            "TransactionAmt": amount,

            "TransactionAmt_log": float(
                np.log1p(amount)
            ),

            "transaction_hour": float(
                transaction_hour
            ),

            "transaction_day": float(
                transaction_day
            ),

            "has_identity": float(
                has_identity
            ),

            "missing_card_info": float(
                missing_card_info
            ),

            "missing_address": float(
                missing_address
            ),

            "missing_value_count": float(
                missing_value_count
            ),
        }

    # =====================================================
    # CARD HISTORY
    # =====================================================

    def _card_history(
        self,
        transaction: dict[str, Any],
        transaction_dt: float,
    ) -> pd.DataFrame:

        card1 = transaction.get(
            "card1"
        )

        if card1 is None:
            return self.historical_df.iloc[0:0]

        history = self.historical_df[
            (
                self.historical_df["card1"]
                == card1
            )
            &
            (
                self.historical_df["TransactionDT"]
                < transaction_dt
            )
        ]

        return history

    # =====================================================
    # CARD FEATURES
    # =====================================================

    def _card_features(
        self,
        transaction: dict[str, Any],
        transaction_dt: float,
    ) -> dict[str, float]:

        amount = float(
            transaction.get(
                "TransactionAmt",
                0.0
            )
        )

        history = self._card_history(
            transaction,
            transaction_dt
        )

        transaction_count = len(
            history
        )

        if transaction_count > 0:

            amounts = pd.to_numeric(
                history["TransactionAmt"],
                errors="coerce"
            ).dropna()

            if not amounts.empty:
                average_amount = float(
                    amounts.mean()
                )
            else:
                average_amount = 0.0

        else:

            average_amount = 0.0

        # -----------------------------------------
        # Amount relative to historical average
        # -----------------------------------------

        if average_amount > 0:

            amount_vs_card_avg = (
                amount / average_amount
            )

        else:

            amount_vs_card_avg = 0.0

        # -----------------------------------------
        # New card
        # -----------------------------------------

        new_card = int(
            transaction_count == 0
        )

        return {
            "card_transaction_count": float(
                transaction_count
            ),

            "card_avg_amount": float(
                average_amount
            ),

            "amount_vs_card_avg": float(
                amount_vs_card_avg
            ),

            "new_card": float(
                new_card
            ),
        }

    # =====================================================
    # CARD VELOCITY
    # =====================================================

    def _card_velocity_features(
        self,
        transaction: dict[str, Any],
        transaction_dt: float,
    ) -> dict[str, float]:

        history = self._card_history(
            transaction,
            transaction_dt
        )

        one_hour_start = (
            transaction_dt - 3600
        )

        twenty_four_hour_start = (
            transaction_dt - 86400
        )

        history_1h = history[
            history["TransactionDT"]
            >= one_hour_start
        ]

        history_24h = history[
            history["TransactionDT"]
            >= twenty_four_hour_start
        ]

        return {
            "card_txn_count_1h": float(
                len(history_1h)
            ),

            "card_txn_count_24h": float(
                len(history_24h)
            ),
        }

    # =====================================================
    # DEVICE HISTORY
    # =====================================================

    def _device_history(
        self,
        transaction: dict[str, Any],
        transaction_dt: float,
    ) -> pd.DataFrame:

        device_info = transaction.get(
            "DeviceInfo"
        )

        if (
            device_info is None
            or pd.isna(device_info)
        ):

            return self.historical_df.iloc[0:0]

        if "DeviceInfo" not in self.historical_df.columns:

            return self.historical_df.iloc[0:0]

        history = self.historical_df[
            (
                self.historical_df["DeviceInfo"]
                == device_info
            )
            &
            (
                self.historical_df["TransactionDT"]
                < transaction_dt
            )
        ]

        return history

    # =====================================================
    # DEVICE FEATURES
    # =====================================================

    def _device_features(
        self,
        transaction: dict[str, Any],
        transaction_dt: float,
    ) -> dict[str, float]:

        device_info = transaction.get(
            "DeviceInfo"
        )

        has_device_info = int(
            device_info is not None
            and not pd.isna(device_info)
            and str(device_info).strip() != ""
        )

        if not has_device_info:

            return {
                "has_device_info": 0.0,
                "device_profile_count": 0.0,
                "device_profile_unique_cards": 0.0,
                "new_device_profile": 1.0,
                "device_unique_cards_historical": 0.0,
            }

        history = self._device_history(
            transaction,
            transaction_dt
        )

        profile_count = len(
            history
        )

        if "card1" in history.columns:

            unique_cards = int(
                history["card1"]
                .dropna()
                .nunique()
            )

        else:

            unique_cards = 0

        new_device_profile = int(
            profile_count == 0
        )

        return {
            "has_device_info": 1.0,

            "device_profile_count": float(
                profile_count
            ),

            "device_profile_unique_cards": float(
                unique_cards
            ),

            "new_device_profile": float(
                new_device_profile
            ),

            "device_unique_cards_historical": float(
                unique_cards
            ),
        }

    # =====================================================
    # CARD + DEVICE FEATURES
    # =====================================================

    def _card_device_features(
        self,
        transaction: dict[str, Any],
        transaction_dt: float,
    ) -> dict[str, float]:

        card1 = transaction.get(
            "card1"
        )

        device_info = transaction.get(
            "DeviceInfo"
        )

        if (
            card1 is None
            or device_info is None
            or pd.isna(device_info)
        ):

            return {
                "card_device_transaction_count": 0.0,
                "card_device_seen_before": 0.0,
            }

        if "DeviceInfo" not in self.historical_df.columns:

            return {
                "card_device_transaction_count": 0.0,
                "card_device_seen_before": 0.0,
            }

        history = self.historical_df[
            (
                self.historical_df["card1"]
                == card1
            )
            &
            (
                self.historical_df["DeviceInfo"]
                == device_info
            )
            &
            (
                self.historical_df["TransactionDT"]
                < transaction_dt
            )
        ]

        transaction_count = len(
            history
        )

        return {
            "card_device_transaction_count": float(
                transaction_count
            ),

            "card_device_seen_before": float(
                int(transaction_count > 0)
            ),
        }

    # =====================================================
    # MAIN BUILD FUNCTION
    # =====================================================

    def build(
        self,
        transaction: dict[str, Any],
    ) -> dict[str, float]:
        """
        Generate the 21 model features for a transaction.

        The transaction itself does NOT need to exist in
        historical_df.

        Historical card/device information is calculated
        only from transactions occurring before it.
        """

        if not isinstance(
            transaction,
            dict
        ):
            raise TypeError(
                "transaction must be a dictionary"
            )

        if "TransactionDT" not in transaction:
            raise ValueError(
                "TransactionDT is required"
            )

        if "TransactionAmt" not in transaction:
            raise ValueError(
                "TransactionAmt is required"
            )

        if "card1" not in transaction:
            raise ValueError(
                "card1 is required"
            )

        transaction_dt = float(
            transaction["TransactionDT"]
        )

        features = {}

        # -----------------------------------------
        # Basic
        # -----------------------------------------

        features.update(
            self._basic_features(
                transaction
            )
        )

        # -----------------------------------------
        # Card
        # -----------------------------------------

        features.update(
            self._card_features(
                transaction,
                transaction_dt
            )
        )

        # -----------------------------------------
        # Card velocity
        # -----------------------------------------

        features.update(
            self._card_velocity_features(
                transaction,
                transaction_dt
            )
        )

        # -----------------------------------------
        # Device
        # -----------------------------------------

        features.update(
            self._device_features(
                transaction,
                transaction_dt
            )
        )

        # -----------------------------------------
        # Card + device
        # -----------------------------------------

        features.update(
            self._card_device_features(
                transaction,
                transaction_dt
            )
        )

        # -----------------------------------------
        # Ensure exact model columns
        # -----------------------------------------

        missing_features = [
            column
            for column in self.FEATURE_COLUMNS
            if column not in features
        ]

        if missing_features:

            raise RuntimeError(
                "Feature builder failed to create: "
                f"{missing_features}"
            )

        # -----------------------------------------
        # Return ONLY the 21 model features
        # in the correct order
        # -----------------------------------------

        ordered_features = {
            column: float(
                features[column]
            )
            for column in self.FEATURE_COLUMNS
        }

        return ordered_features

    # =====================================================
    # DATAFRAME OUTPUT
    # =====================================================

    def build_dataframe(
        self,
        transaction: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Return the generated features as a
        one-row DataFrame.
        """

        features = self.build(
            transaction
        )

        return pd.DataFrame(
            [features],
            columns=self.FEATURE_COLUMNS
        )