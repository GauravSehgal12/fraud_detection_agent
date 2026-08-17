from typing import Any

import pandas as pd

from src.services.risk_engine import RiskEngine
from src.services.transaction_feature_builder import (
    TransactionFeatureBuilder,
)


class FraudInvestigationTools:

    def __init__(
        self,
        transactions: pd.DataFrame,
        identity: pd.DataFrame | None = None,
        risk_assessments: dict | None = None,
        model=None,
    ):
        """
        Fraud investigation tools.

        transactions:
            ORIGINAL transaction dataframe.
            Must contain the complete raw transaction schema.

        identity:
            IEEE-CIS identity dataframe.

        model:
            Loaded XGBoost model.
        """

        # =====================================================
        # DATA
        # =====================================================

        self.transactions = transactions
        self.identity = identity

        self.risk_assessments = (
            risk_assessments
            if risk_assessments is not None
            else {}
        )

        # =====================================================
        # VALIDATION
        # =====================================================

        if not isinstance(
            self.transactions,
            pd.DataFrame,
        ):
            raise TypeError(
                "transactions must be a pandas DataFrame."
            )

        required_columns = [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "card1",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.transactions.columns
        ]

        if missing_columns:

            raise ValueError(
                "Transactions dataframe is missing "
                f"required columns: {missing_columns}"
            )

        if self.identity is not None:

            if not isinstance(
                self.identity,
                pd.DataFrame,
            ):
                raise TypeError(
                    "identity must be a pandas DataFrame."
                )

            if (
                "TransactionID"
                not in self.identity.columns
            ):

                raise ValueError(
                    "identity dataframe must contain "
                    "'TransactionID'."
                )

        # =====================================================
        # CREATE FEATURE-BUILDER HISTORY
        # =====================================================

        # IMPORTANT:
        #
        # Keep ALL original transaction columns.
        #
        # Do NOT reduce this dataframe to only:
        #
        # TransactionID
        # TransactionDT
        # TransactionAmt
        # card1
        #
        # The feature builder needs the original schema.
        # =====================================================

        history = self.transactions.copy()

        # =====================================================
        # ADD DEVICE INFO FROM IDENTITY
        # =====================================================

        if (
            self.identity is not None
            and "TransactionID"
            in self.identity.columns
            and "DeviceInfo"
            in self.identity.columns
        ):

            device_data = self.identity[
                [
                    "TransactionID",
                    "DeviceInfo",
                ]
            ].copy()

            device_data = (
                device_data
                .drop_duplicates(
                    subset=[
                        "TransactionID"
                    ],
                    keep="first",
                )
            )

            # If DeviceInfo already exists in the
            # transaction dataframe, keep the existing
            # column and fill only missing values from
            # identity.

            if "DeviceInfo" in history.columns:

                identity_device_map = (
                    device_data.set_index(
                        "TransactionID"
                    )["DeviceInfo"]
                )

                history["DeviceInfo"] = (
                    history["DeviceInfo"]
                    .fillna(
                        history[
                            "TransactionID"
                        ].map(
                            identity_device_map
                        )
                    )
                )

            else:

                history = history.merge(
                    device_data,
                    on="TransactionID",
                    how="left",
                    sort=False,
                )

        else:

            if "DeviceInfo" not in history.columns:

                history["DeviceInfo"] = None

        # =====================================================
        # FEATURE BUILDER
        # =====================================================

        self.feature_builder = (
            TransactionFeatureBuilder(
                historical_df=history,
                identity=self.identity,

                # IMPORTANT:
                # raw_columns must be the original
                # transaction schema.
                raw_columns=(
                    self.transactions
                    .columns
                    .tolist()
                ),
            )
        )

        # =====================================================
        # DEBUG
        # =====================================================

        print(
            "\n========== FEATURE BUILDER =========="
        )

        print(
            "Transaction columns:",
            len(
                self.transactions.columns
            ),
        )

        print(
            "Feature builder raw columns:",
            len(
                self.feature_builder.raw_columns
            ),
        )

        print(
            "Identity rows:",
            len(self.identity)
            if self.identity is not None
            else 0,
        )

        print(
            "=====================================\n"
        )

        # =====================================================
        # RISK ENGINE
        # =====================================================

        self.risk_engine = None

        if model is not None:

            self.risk_engine = RiskEngine(
                model=model,
                feature_builder=(
                    self.feature_builder
                ),
            )

    # =========================================================
    # FIND HISTORICAL BASE TRANSACTION
    # =========================================================

    def _find_base_transaction(
        self,
        transaction: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Find the historical transaction that corresponds
        to a new/synthetic transaction.

        Matching strategy:

            card1
              +
            TransactionDT

        This is useful for the synthetic transaction used
        in the notebook:

            original:
                TransactionID = 3409570

            new:
                TransactionID = 999999999

        where the transaction characteristics are retained
        but selected fields are overridden.
        """

        card1 = transaction.get(
            "card1"
        )

        transaction_dt = transaction.get(
            "TransactionDT"
        )

        if (
            card1 is None
            or transaction_dt is None
        ):
            return None

        try:

            card1_numeric = float(
                card1
            )

            transaction_dt_numeric = float(
                transaction_dt
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        # -----------------------------------------------------
        # Match card1 + TransactionDT
        # -----------------------------------------------------

        matches = self.transactions[
            (
                pd.to_numeric(
                    self.transactions["card1"],
                    errors="coerce",
                )
                == card1_numeric
            )
            &
            (
                pd.to_numeric(
                    self.transactions[
                        "TransactionDT"
                    ],
                    errors="coerce",
                )
                == transaction_dt_numeric
            )
        ]

        if matches.empty:

            return None

        # -----------------------------------------------------
        # Return complete original row
        # -----------------------------------------------------

        row = matches.iloc[0]

        result = row.to_dict()

        return result

    # =========================================================
    # ENRICH NEW TRANSACTION
    # =========================================================

    def _prepare_new_transaction(
        self,
        transaction: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Prepare a new transaction for feature generation.

        If a matching historical transaction exists using
        card1 + TransactionDT, use that complete raw row as
        the base and overwrite it with the values supplied
        by the API.

        This preserves the original 434-column structure.
        """

        # -----------------------------------------------------
        # Try to find matching historical transaction
        # -----------------------------------------------------

        base_transaction = (
            self._find_base_transaction(
                transaction
            )
        )

        if base_transaction is not None:

            # Start with complete 434-column row
            enriched = base_transaction.copy()

            # -------------------------------------------------
            # Override with API values
            # -------------------------------------------------

            enriched.update(
                transaction
            )

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Keep the NEW TransactionID.
            # -------------------------------------------------

            if (
                "TransactionID"
                in transaction
            ):

                enriched[
                    "TransactionID"
                ] = transaction[
                    "TransactionID"
                ]

            print(
                "\n"
                "========== NEW TRANSACTION =========="
            )

            print(
                "Matched historical transaction:"
            )

            print(
                base_transaction.get(
                    "TransactionID"
                )
            )

            print(
                "New transaction ID:"
            )

            print(
                transaction.get(
                    "TransactionID"
                )
            )

            print(
                "Raw fields:",
                len(enriched),
            )

            print(
                "======================================\n"
            )

            return enriched

        # -----------------------------------------------------
        # No historical match
        # -----------------------------------------------------

        enriched = transaction.copy()

        print(
            "\n"
            "========== NEW TRANSACTION =========="
        )

        print(
            "No historical base transaction "
            "found."
        )

        print(
            "Using supplied transaction only."
        )

        print(
            "Fields:",
            len(enriched),
        )

        print(
            "======================================\n"
        )

        return enriched

    # =========================================================
    # GET TRANSACTION
    # =========================================================

    def get_transaction(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Get transaction information.

        Existing transaction:

            get_transaction(3409570)

        New transaction:

            get_transaction({
                "TransactionID": 999999999,
                "TransactionDT": 10699419,
                "TransactionAmt": 87.302,
                "card1": 12730,
                "DeviceInfo":
                    "LG-D320 Build/KOT49I.V10a"
            })
        """

        # =====================================================
        # EXISTING TRANSACTION
        # =====================================================

        if isinstance(
            transaction,
            int,
        ):

            matches = self.transactions[
                self.transactions[
                    "TransactionID"
                ]
                == transaction
            ]

            if matches.empty:

                return {
                    "error":
                        "Transaction not found."
                }

            row = matches.iloc[0]

            result = {}

            for column in [
                "TransactionID",
                "TransactionDT",
                "TransactionAmt",
                "card1",
            ]:

                if column in row.index:

                    value = row[column]

                    if pd.isna(value):

                        value = None

                    elif hasattr(
                        value,
                        "item",
                    ):

                        value = value.item()

                    result[column] = value

            # -------------------------------------------------
            # DeviceInfo
            # -------------------------------------------------

            result["DeviceInfo"] = None

            if (
                self.identity is not None
                and "TransactionID"
                in self.identity.columns
                and "DeviceInfo"
                in self.identity.columns
            ):

                identity_match = self.identity[
                    self.identity[
                        "TransactionID"
                    ]
                    == transaction
                ]

                if not identity_match.empty:

                    value = (
                        identity_match.iloc[0][
                            "DeviceInfo"
                        ]
                    )

                    if pd.isna(value):

                        value = None

                    result["DeviceInfo"] = value

            elif "DeviceInfo" in row.index:

                value = row[
                    "DeviceInfo"
                ]

                if pd.isna(value):

                    value = None

                result["DeviceInfo"] = value

            return result

        # =====================================================
        # NEW TRANSACTION
        # =====================================================

        if isinstance(
            transaction,
            dict,
        ):

            transaction_id = (
                transaction.get(
                    "TransactionID"
                )
            )

            if transaction_id is None:

                return {
                    "error":
                        "TransactionID is required."
                }

            # -------------------------------------------------
            # Return the user-facing transaction.
            #
            # We do NOT return all 434 columns here.
            # The enriched version is used only for risk
            # feature generation.
            # -------------------------------------------------

            result = {}

            for column in [
                "TransactionID",
                "TransactionDT",
                "TransactionAmt",
                "card1",
                "DeviceInfo",
            ]:

                if column in transaction:

                    value = transaction[
                        column
                    ]

                    if (
                        value is None
                        or pd.isna(value)
                    ):

                        value = None

                    elif hasattr(
                        value,
                        "item",
                    ):

                        value = value.item()

                    result[column] = value

            return result

        return {
            "error": (
                "Transaction must be either "
                "an integer transaction ID or "
                "a transaction dictionary."
            )
        }

    # =========================================================
    # RISK ASSESSMENT
    # =========================================================

    def get_risk_assessment(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Calculate dynamic fraud risk.
        """

        if self.risk_engine is None:

            return {
                "error":
                    "Risk engine not initialized."
            }

        # =====================================================
        # EXISTING TRANSACTION
        # =====================================================

        if isinstance(
            transaction,
            int,
        ):

            risk_input = transaction

        # =====================================================
        # NEW TRANSACTION
        # =====================================================

        elif isinstance(
            transaction,
            dict,
        ):

            risk_input = (
                self._prepare_new_transaction(
                    transaction
                )
            )

        else:

            return {
                "error": (
                    "Invalid transaction type."
                )
            }

        # =====================================================
        # RISK ENGINE
        # =====================================================

        try:

            result = (
                self.risk_engine.calculate_risk(
                    risk_input
                )
            )

        except Exception as exc:

            return {
                "error":
                    "Risk calculation failed.",
                "details":
                    str(exc),
            }

        return result

    # =========================================================
    # CARD HISTORY
    # =========================================================

    def get_card_history(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return historical card behavior.

        Only transactions before the current
        TransactionDT are included.
        """

        transaction_data = (
            self.get_transaction(
                transaction
            )
        )

        if "error" in transaction_data:

            return transaction_data

        card1 = transaction_data.get(
            "card1"
        )

        transaction_dt = (
            transaction_data.get(
                "TransactionDT"
            )
        )

        if (
            card1 is None
            or transaction_dt is None
        ):

            return {
                "error":
                    "Card information unavailable."
            }

        history = self.transactions[
            (
                self.transactions[
                    "card1"
                ]
                == card1
            )
            &
            (
                self.transactions[
                    "TransactionDT"
                ]
                < transaction_dt
            )
        ].copy()

        if history.empty:

            return {
                "card1": int(card1),
                "transaction_count": 0,
                "average_amount": 0.0,
                "max_amount": 0.0,
                "min_amount": 0.0,
            }

        amounts = pd.to_numeric(
            history[
                "TransactionAmt"
            ],
            errors="coerce",
        ).dropna()

        return {
            "card1":
                int(card1),

            "transaction_count":
                int(len(history)),

            "average_amount":
                float(
                    amounts.mean()
                )
                if not amounts.empty
                else 0.0,

            "max_amount":
                float(
                    amounts.max()
                )
                if not amounts.empty
                else 0.0,

            "min_amount":
                float(
                    amounts.min()
                )
                if not amounts.empty
                else 0.0,
        }

    # =========================================================
    # DEVICE HISTORY
    # =========================================================

    def get_device_history(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return historical device behavior.
        """

        transaction_data = (
            self.get_transaction(
                transaction
            )
        )

        if "error" in transaction_data:

            return transaction_data

        device_info = (
            transaction_data.get(
                "DeviceInfo"
            )
        )

        transaction_dt = (
            transaction_data.get(
                "TransactionDT"
            )
        )

        if (
            device_info is None
            or transaction_dt is None
        ):

            return {
                "device_info": None,
                "transaction_count": 0,
                "unique_cards": 0,
            }

        # =====================================================
        # DEVICE HISTORY FROM IDENTITY
        # =====================================================

        if (
            self.identity is None
            or "TransactionID"
            not in self.identity.columns
            or "DeviceInfo"
            not in self.identity.columns
        ):

            return {
                "device_info":
                    device_info,

                "transaction_count":
                    0,

                "unique_cards":
                    0,
            }

        device_data = self.identity[
            [
                "TransactionID",
                "DeviceInfo",
            ]
        ].copy()

        device_data = (
            device_data
            .drop_duplicates(
                subset=[
                    "TransactionID"
                ],
                keep="first",
            )
        )

        transaction_data_df = (
            self.transactions[
                [
                    "TransactionID",
                    "TransactionDT",
                    "card1",
                ]
            ]
            .drop_duplicates(
                subset=[
                    "TransactionID"
                ],
                keep="first",
            )
        )

        device_history = (
            device_data.merge(
                transaction_data_df,
                on="TransactionID",
                how="inner",
            )
        )

        history = device_history[
            (
                device_history[
                    "DeviceInfo"
                ]
                == device_info
            )
            &
            (
                device_history[
                    "TransactionDT"
                ]
                < transaction_dt
            )
        ].copy()

        if history.empty:

            return {
                "device_info":
                    device_info,

                "transaction_count":
                    0,

                "unique_cards":
                    0,
            }

        unique_cards = (
            history[
                "card1"
            ]
            .dropna()
            .nunique()
        )

        return {
            "device_info":
                device_info,

            "transaction_count":
                int(
                    len(history)
                ),

            "unique_cards":
                int(
                    unique_cards
                ),
        }

    # =========================================================
    # COMPLETE INVESTIGATION DATA
    # =========================================================

    def collect_investigation_data(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Collect all deterministic investigation data.
        """

        transaction_data = (
            self.get_transaction(
                transaction
            )
        )

        if "error" in transaction_data:

            return transaction_data

        risk = (
            self.get_risk_assessment(
                transaction
            )
        )

        if "error" in risk:

            return risk

        card_history = (
            self.get_card_history(
                transaction
            )
        )

        if "error" in card_history:

            return card_history

        device_history = (
            self.get_device_history(
                transaction
            )
        )

        if "error" in device_history:

            return device_history

        return {
            "transaction":
                transaction_data,

            "risk_assessment":
                risk,

            "card_history":
                card_history,

            "device_history":
                device_history,
        }