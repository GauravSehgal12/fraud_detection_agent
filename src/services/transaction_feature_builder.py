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

    # =========================================================
    # ORIGINAL TRANSACTION COLUMNS USED BY NOTEBOOK
    # =========================================================

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
        Parameters
        ----------
        historical_df:
            Historical transaction dataframe used for
            behavioral features.

            This dataframe must contain:

                TransactionID
                TransactionDT
                TransactionAmt
                card1
                DeviceInfo

        identity:
            IEEE-CIS identity dataframe.

        raw_columns:
            ORIGINAL train_transaction.csv columns.

            IMPORTANT:
            These must be the 394 transaction columns,
            NOT the 434 merged transaction + identity columns.
        """

        # =====================================================
        # VALIDATE HISTORICAL DATA
        # =====================================================

        if not isinstance(
            historical_df,
            pd.DataFrame,
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

        # =====================================================
        # IDENTITY IDS
        # =====================================================

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
                identity[
                    "TransactionID"
                ]
                .dropna()
                .astype("int64")
                .tolist()
            )

        else:

            self.identity_ids = set()

        # =====================================================
        # ORIGINAL TRANSACTION SCHEMA
        # =====================================================

        if raw_columns is None:

            self.raw_columns = list(
                historical_df.columns
            )

        else:

            self.raw_columns = list(
                raw_columns
            )

        # -----------------------------------------------------
        # IMPORTANT
        #
        # If somebody accidentally passes the merged
        # transaction + identity schema, remove identity-only
        # columns so missing_value_count still uses the
        # original transaction schema.
        # -----------------------------------------------------

        if identity is not None:

            identity_only_columns = {
                column
                for column in identity.columns
                if column != "TransactionID"
            }

            self.raw_columns = [
                column
                for column in self.raw_columns
                if column
                not in identity_only_columns
            ]

        print(
            "\n========== FEATURE BUILDER =========="
        )

        print(
            "Historical columns:",
            len(
                historical_df.columns
            )
        )

        print(
            "Feature builder raw columns:",
            len(
                self.raw_columns
            )
        )

        print(
            "Identity rows:",
            len(identity)
            if identity is not None
            else 0
        )

        print(
            "=====================================\n"
        )

    # =========================================================
    # BASIC FEATURES
    # =========================================================

    def _basic_features(
        self,
        transaction: dict[str, Any],
    ) -> dict[str, float]:

        # -----------------------------------------------------
        # Transaction amount
        # -----------------------------------------------------

        amount = float(
            transaction.get(
                "TransactionAmt",
                0.0,
            )
        )

        # -----------------------------------------------------
        # Transaction DT
        # -----------------------------------------------------

        transaction_dt = float(
            transaction.get(
                "TransactionDT",
                0.0,
            )
        )

        # -----------------------------------------------------
        # Transaction amount log
        # -----------------------------------------------------

        transaction_amt_log = float(
            np.log1p(amount)
        )

        # -----------------------------------------------------
        # Transaction hour
        # -----------------------------------------------------

        transaction_hour = int(
            (transaction_dt // 3600) % 24
        )

        # -----------------------------------------------------
        # Transaction day
        # -----------------------------------------------------

        transaction_day = int(
            transaction_dt // 86400
        )

        # =====================================================
        # IDENTITY
        # =====================================================

        raw_transaction_id = (
            transaction.get(
                "TransactionID"
            )
        )

        try:

            transaction_id = (
                int(raw_transaction_id)
                if raw_transaction_id is not None
                else None
            )

        except (
            TypeError,
            ValueError,
        ):

            transaction_id = None

        has_identity = int(
            transaction_id is not None
            and transaction_id
            in self.identity_ids
        )

        # =====================================================
        # MISSING CARD INFORMATION
        # =====================================================

        card_values = [
            transaction.get(
                column
            )
            for column in self.CARD_COLUMNS
        ]

        missing_card_info = int(
            any(
                value is None
                or pd.isna(value)
                for value in card_values
            )
        )

        # =====================================================
        # MISSING ADDRESS
        # =====================================================

        address_values = [
            transaction.get(
                column
            )
            for column in self.ADDRESS_COLUMNS
        ]

        missing_address = int(
            any(
                value is None
                or pd.isna(value)
                for value in address_values
            )
        )

        # =====================================================
        # MISSING VALUE COUNT
        #
        # IMPORTANT:
        #
        # This uses ONLY the original transaction columns.
        #
        # train_transaction.csv
        #     ~394 columns
        #
        # NOT:
        #
        # transaction + identity
        #     ~434 columns
        # =====================================================

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
                float(
                    missing_value_count
                ),
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
                self.historical_df[
                    "card1"
                ]
                == card1
            )
            &
            (
                self.historical_df[
                    "TransactionDT"
                ]
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
                0.0,
            )
        )

        history = self._card_history(
            transaction,
            transaction_dt,
        )

        card_transaction_count = len(
            history
        )

        # -----------------------------------------------------
        # Historical average
        # -----------------------------------------------------

        if card_transaction_count > 0:

            amounts = pd.to_numeric(
                history[
                    "TransactionAmt"
                ],
                errors="coerce",
            ).dropna()

            if not amounts.empty:

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
                amount
                / card_avg_amount
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
            transaction_dt,
        )

        # -----------------------------------------------------
        # Previous 1 hour
        # -----------------------------------------------------

        history_1h = history[
            history[
                "TransactionDT"
            ]
            >= transaction_dt - 3600
        ]

        # -----------------------------------------------------
        # Previous 24 hours
        # -----------------------------------------------------

        history_24h = history[
            history[
                "TransactionDT"
            ]
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
                self.historical_df[
                    "DeviceInfo"
                ]
                == device_info
            )
            &
            (
                self.historical_df[
                    "TransactionDT"
                ]
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
            transaction_dt,
        )

        device_profile_count = len(
            history
        )

        if "card1" in history.columns:

            device_profile_unique_cards = int(
                history[
                    "card1"
                ]
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

        # =====================================================
        # CARD + DEVICE HISTORY
        # =====================================================

        card_device_history = (
            self.historical_df[
                (
                    self.historical_df[
                        "DeviceInfo"
                    ]
                    == device_info
                )
                &
                (
                    self.historical_df[
                        "card1"
                    ]
                    == card1
                )
                &
                (
                    self.historical_df[
                        "TransactionDT"
                    ]
                    < transaction_dt
                )
            ]
        )

        card_device_transaction_count = len(
            card_device_history
        )

        card_device_seen_before = int(
            card_device_transaction_count > 0
        )

        # =====================================================
        # DEVICE UNIQUE CARDS HISTORICAL
        #
        # Reproduce the training logic:
        #
        # sort by:
        # DeviceInfo
        # card1
        # TransactionDT
        #
        # Count unique cards before the current card1.
        # =====================================================

        device_history = (
            self.historical_df[
                (
                    self.historical_df[
                        "DeviceInfo"
                    ]
                    == device_info
                )
                &
                (
                    self.historical_df[
                        "card1"
                    ].notna()
                )
            ]
            .copy()
        )

        if device_history.empty:

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
                    0.0,
            }

        # -----------------------------------------------------
        # Sort exactly like training
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
            .reset_index(
                drop=True
            )
        )

        # -----------------------------------------------------
        # Count cards before current card1
        # -----------------------------------------------------

        cards_seen = set()

        for historical_card in (
            device_history[
                "card1"
            ]
        ):

            if (
                pd.isna(
                    historical_card
                )
            ):
                continue

            if historical_card >= card1:
                break

            cards_seen.add(
                historical_card
            )

        device_unique_cards_historical = (
            len(cards_seen)
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
        Generate the exact 21 model features.
        """

        if not isinstance(
            transaction,
            dict,
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
            transaction[
                "TransactionDT"
            ]
        )

        features = {}

        # =====================================================
        # BASIC
        # =====================================================

        features.update(
            self._basic_features(
                transaction
            )
        )

        # =====================================================
        # CARD
        # =====================================================

        features.update(
            self._card_features(
                transaction,
                transaction_dt,
            )
        )

        # =====================================================
        # CARD VELOCITY
        # =====================================================

        features.update(
            self._card_velocity_features(
                transaction,
                transaction_dt,
            )
        )

        # =====================================================
        # DEVICE
        # =====================================================

        features.update(
            self._device_features(
                transaction,
                transaction_dt,
            )
        )

        # =====================================================
        # CARD + DEVICE
        # =====================================================

        features.update(
            self._card_device_features(
                transaction,
                transaction_dt,
            )
        )

        # =====================================================
        # VALIDATE
        # =====================================================

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

        # =====================================================
        # EXACT MODEL ORDER
        # =====================================================

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

        features = self.build(
            transaction
        )

        return pd.DataFrame(
            [features],
            columns=self.FEATURE_COLUMNS,
        )