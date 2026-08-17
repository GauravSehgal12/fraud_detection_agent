# src/services/transaction_feature_builder.py

from typing import Any

import numpy as np
import pandas as pd


class TransactionFeatureBuilder:

    # =========================================================
    # EXACT 21 FEATURES USED BY XGBOOST
    # =========================================================

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

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        historical_df: pd.DataFrame,
        identity: pd.DataFrame | None = None,
        raw_columns: list[str] | None = None,
    ):
        """
        historical_df:
            Historical transaction dataframe used for
            behavioral feature generation.

        identity:
            Identity dataframe used to calculate
            has_identity.

        raw_columns:
            Original transaction columns used for
            missing_value_count.
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
                "historical_df is missing columns: "
                f"{missing}"
            )

        self.historical_df = (
            historical_df.copy()
        )

        # -----------------------------------------------------
        # Identity transaction IDs
        # -----------------------------------------------------

        if identity is not None:

            if (
                "TransactionID"
                not in identity.columns
            ):
                raise ValueError(
                    "identity must contain "
                    "'TransactionID'"
                )

            self.identity_ids = set(
                identity["TransactionID"]
                .dropna()
                .astype("int64")
                .tolist()
            )

        else:

            self.identity_ids = set()

        # -----------------------------------------------------
        # Original raw schema
        # -----------------------------------------------------

        if raw_columns is not None:
            self.raw_columns = list(
                raw_columns
            )
        else:
            self.raw_columns = list(
                historical_df.columns
            )

    # =========================================================
    # BASIC FEATURES
    # =========================================================

    def _basic_features(
        self,
        transaction: dict[str, Any],
    ) -> dict[str, float]:

        transaction_id = transaction.get(
            "TransactionID"
        )

        transaction_dt = float(
            transaction.get(
                "TransactionDT",
                0.0
            )
        )

        amount = float(
            transaction.get(
                "TransactionAmt",
                0.0
            )
        )

        # -----------------------------------------------------
        # Transaction amount log
        # -----------------------------------------------------

        transaction_amt_log = float(
            np.log1p(amount)
        )

        # -----------------------------------------------------
        # Hour
        # -----------------------------------------------------

        transaction_hour = int(
            (transaction_dt // 3600) % 24
        )

        # -----------------------------------------------------
        # Day
        # -----------------------------------------------------

        transaction_day = int(
            transaction_dt // 86400
        )

        # -----------------------------------------------------
        # Identity
        # -----------------------------------------------------

        raw_transaction_id = transaction.get(
            "TransactionID"
        )

        try:
            transaction_id = int(
                raw_transaction_id
            ) if raw_transaction_id is not None else None
        except (
            TypeError,
            ValueError
        ):
            transaction_id = None

        has_identity = int(
            transaction_id is not None
            and transaction_id
            in self.identity_ids
        )

        # -----------------------------------------------------
        # Missing card information
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Missing address
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Missing value count
        #
        # IMPORTANT:
        # raw_columns must be transactions.columns
        # when creating the builder.
        # -----------------------------------------------------

        missing_value_count = 0

        for column in self.raw_columns:

            value = transaction.get(
                column
            )

            if (
                value is None
                or pd.isna(value)
            ):
                missing_value_count += 1

        return {

            "TransactionAmt":
                amount,

            "TransactionAmt_log":
                transaction_amt_log,

            "transaction_hour":
                float(transaction_hour),

            "transaction_day":
                float(transaction_day),

            "has_identity":
                float(has_identity),

            "missing_card_info":
                float(missing_card_info),

            "missing_address":
                float(missing_address),

            "missing_value_count":
                float(missing_value_count),
        }

    # =========================================================
    # CARD HISTORY
    # =========================================================

    def _card_history(
        self,
        transaction: dict[str, Any],
        transaction_dt: float,
    ) -> pd.DataFrame:

        card1 = transaction.get(
            "card1"
        )

        if (
            card1 is None
            or pd.isna(card1)
        ):
            return self.historical_df.iloc[
                0:0
            ]

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

    # =========================================================
    # CARD BEHAVIOR
    # =========================================================

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

        card_transaction_count = len(
            history
        )

        # -----------------------------------------------------
        # Historical average amount
        # -----------------------------------------------------

        if card_transaction_count > 0:

            amounts = pd.to_numeric(
                history["TransactionAmt"],
                errors="coerce"
            ).dropna()

            if len(amounts) > 0:

                card_avg_amount = float(
                    amounts.mean()
                )

            else:

                card_avg_amount = 0.0

        else:

            card_avg_amount = 0.0

        # -----------------------------------------------------
        # Amount vs average
        # -----------------------------------------------------

        if card_avg_amount > 0:

            amount_vs_card_avg = (
                amount /
                card_avg_amount
            )

        else:

            amount_vs_card_avg = 0.0

        # -----------------------------------------------------
        # New card
        # -----------------------------------------------------

        new_card = int(
            card_transaction_count == 0
        )

        return {

            "card_transaction_count":
                float(
                    card_transaction_count
                ),

            "card_avg_amount":
                float(
                    card_avg_amount
                ),

            "amount_vs_card_avg":
                float(
                    amount_vs_card_avg
                ),

            "new_card":
                float(
                    new_card
                ),
        }

    # =========================================================
    # CARD VELOCITY
    # =========================================================

    def _card_velocity_features(
        self,
        transaction: dict[str, Any],
        transaction_dt: float,
    ) -> dict[str, float]:

        history = self._card_history(
            transaction,
            transaction_dt
        )

        # -----------------------------------------------------
        # Previous 1 hour
        # -----------------------------------------------------

        history_1h = history[
            history["TransactionDT"]
            >= transaction_dt - 3600
        ]

        # -----------------------------------------------------
        # Previous 24 hours
        # -----------------------------------------------------

        history_24h = history[
            history["TransactionDT"]
            >= transaction_dt - 86400
        ]

        return {

            "card_txn_count_1h":
                float(
                    len(history_1h)
                ),

            "card_txn_count_24h":
                float(
                    len(history_24h)
                ),
        }

    # =========================================================
    # DEVICE HISTORY
    # =========================================================

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
            or str(device_info).strip() == ""
        ):
            return self.historical_df.iloc[
                0:0
            ]

        if (
            "DeviceInfo"
            not in self.historical_df.columns
        ):
            return self.historical_df.iloc[
                0:0
            ]

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

    # =========================================================
    # DEVICE PROFILE FEATURES
    # =========================================================

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

                "has_device_info":
                    0.0,

                "device_profile_count":
                    0.0,

                "device_profile_unique_cards":
                    0.0,

                "new_device_profile":
                    1.0,
            }

        history = self._device_history(
            transaction,
            transaction_dt
        )

        device_profile_count = len(
            history
        )

        if "card1" in history.columns:

            device_profile_unique_cards = int(
                history["card1"]
                .dropna()
                .nunique()
            )

        else:

            device_profile_unique_cards = 0

        new_device_profile = int(
            device_profile_count == 0
        )

        return {

            "has_device_info":
                1.0,

            "device_profile_count":
                float(
                    device_profile_count
                ),

            "device_profile_unique_cards":
                float(
                    device_profile_unique_cards
                ),

            "new_device_profile":
                float(
                    new_device_profile
                ),
        }

    # =========================================================
    # CARD + DEVICE FEATURES
    # =========================================================

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

        # -----------------------------------------------------
        # Missing card/device
        # -----------------------------------------------------

        if (
            card1 is None
            or pd.isna(card1)
            or device_info is None
            or pd.isna(device_info)
            or str(device_info).strip() == ""
        ):

            return {

                "card_device_transaction_count":
                    0.0,

                "card_device_seen_before":
                    0.0,

                "device_unique_cards_historical":
                    0.0,
            }

        if (
            "DeviceInfo"
            not in self.historical_df.columns
        ):

            return {

                "card_device_transaction_count":
                    0.0,

                "card_device_seen_before":
                    0.0,

                "device_unique_cards_historical":
                    0.0,
            }

        # -----------------------------------------------------
        # Exact card + device historical records
        # -----------------------------------------------------

        card_device_history = self.historical_df[
            (
                self.historical_df["DeviceInfo"]
                == device_info
            )
            &
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

        card_device_transaction_count = len(
            card_device_history
        )

        card_device_seen_before = int(
            card_device_transaction_count > 0
        )

        # -----------------------------------------------------
        # REPRODUCE ORIGINAL TRAINING LOGIC
        #
        # Original:
        #
        # df.sort_values(
        #     [
        #         "DeviceInfo",
        #         "card1",
        #         "TransactionDT"
        #     ]
        # )
        #
        # Then:
        #
        # for _, group in df[valid].groupby(
        #     "DeviceInfo",
        #     sort=False
        # ):
        #
        #     cards_seen = set()
        #
        #     positions = group.index.to_numpy()
        #     cards = group["card1"].to_numpy()
        #
        #     for card in cards:
        #
        #         counts.append(
        #             len(cards_seen)
        #         )
        #
        #         cards_seen.add(card)
        #
        # Therefore the feature for the current card
        # depends on cards with LOWER card1 values
        # in the sorted device group.
        # -----------------------------------------------------

        device_history = self.historical_df[
            (
                self.historical_df["DeviceInfo"]
                == device_info
            )
            
            &
            (
                self.historical_df["card1"]
                .notna()
            )
        ].copy()

        # -----------------------------------------------------
        # Sort exactly as original training function
        # -----------------------------------------------------

        device_history = (
            device_history
            .sort_values(
                [
                    "DeviceInfo",
                    "card1",
                    "TransactionDT",
                ]
            )
            .reset_index(drop=True)
        )

        # -----------------------------------------------------
        # Reproduce cards_seen logic.
        #
        # We need the count immediately before
        # the current card1 group.
        # -----------------------------------------------------

        cards_seen = set()

        device_unique_cards_historical = 0

        for historical_card in (
            device_history["card1"]
        ):

            # If we've reached the current
            # card's position in sorted card order,
            # stop BEFORE adding the current card.
            if historical_card >= card1:

                break

            if pd.notna(historical_card):

                cards_seen.add(
                    historical_card
                )

        device_unique_cards_historical = len(
            cards_seen
        )

        return {

            "card_device_transaction_count":
                float(
                    card_device_transaction_count
                ),

            "card_device_seen_before":
                float(
                    card_device_seen_before
                ),

            "device_unique_cards_historical":
                float(
                    device_unique_cards_historical
                ),
        }

    # =========================================================
    # MAIN BUILD
    # =========================================================

    def build(
        self,
        transaction: dict[str, Any],
    ) -> dict[str, float]:
        """
        Generate the exact 21 model features
        for one transaction.

        The transaction itself does not need to
        already exist in the historical dataframe.

        All behavioral features use transactions
        that occurred before the current
        TransactionDT.
        """

        if not isinstance(
            transaction,
            dict
        ):
            raise TypeError(
                "transaction must be a dictionary"
            )

        required_fields = [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "card1",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in transaction
        ]

        if missing_fields:

            raise ValueError(
                "Transaction is missing required "
                f"fields: {missing_fields}"
            )

        transaction_dt = float(
            transaction["TransactionDT"]
        )

        features = {}

        # -----------------------------------------------------
        # Basic
        # -----------------------------------------------------

        features.update(
            self._basic_features(
                transaction
            )
        )

        # -----------------------------------------------------
        # Card behavior
        # -----------------------------------------------------

        features.update(
            self._card_features(
                transaction,
                transaction_dt
            )
        )

        # -----------------------------------------------------
        # Card velocity
        # -----------------------------------------------------

        features.update(
            self._card_velocity_features(
                transaction,
                transaction_dt
            )
        )

        # -----------------------------------------------------
        # Device profile
        # -----------------------------------------------------

        features.update(
            self._device_features(
                transaction,
                transaction_dt
            )
        )

        # -----------------------------------------------------
        # Card + device
        # -----------------------------------------------------

        features.update(
            self._card_device_features(
                transaction,
                transaction_dt
            )
        )

        # -----------------------------------------------------
        # Validate feature completeness
        # -----------------------------------------------------

        missing_features = [
            feature
            for feature in self.FEATURE_COLUMNS
            if feature not in features
        ]

        if missing_features:

            raise RuntimeError(
                "Feature builder failed to create: "
                f"{missing_features}"
            )

        # -----------------------------------------------------
        # Return in exact model order
        # -----------------------------------------------------

        return {
            feature: float(
                features[feature]
            )
            for feature in self.FEATURE_COLUMNS
        }

    # =========================================================
    # DATAFRAME OUTPUT
    # =========================================================

    def build_dataframe(
        self,
        transaction: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Build one-row DataFrame in the exact
        XGBoost feature order.
        """

        features = self.build(
            transaction
        )

        return pd.DataFrame(
            [features],
            columns=self.FEATURE_COLUMNS
        )