import json
from pathlib import Path

import pandas as pd

from src.services.model_loader import ModelLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

TRANSACTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train_transaction.csv"
)

IDENTITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train_identity.csv"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "investigation_history.csv"
)

RISK_PATH = (
    PROJECT_ROOT
    / "data"
    / "risk_assessments.json"
)


class AppContainer:

    def __init__(self):

        # ====================================================
        # MODEL
        # ====================================================

        self.model = None
        self.features = None

        # ====================================================
        # DATA
        # ====================================================

        # Raw transaction dataframe
        self.raw_transactions = None

        # Identity dataframe
        self.identity = None

        # Transaction + identity merged dataframe
        self.transactions = None

        # Existing investigation history
        self.investigation_history = None

        # Persisted risk assessments
        self.risk_assessments = None

        # Agent
        self.agent = None

    # ========================================================
    # MODEL
    # ========================================================

    def load_model(self):

        print(
            "\nLoading model..."
        )

        loader = ModelLoader()

        loaded = loader.load_all()

        self.model = loaded["model"]

        self.features = loaded["features"]

        print(
            "XGBoost model loaded."
        )

        print(
            "Model features:",
            len(self.features)
        )

    # ========================================================
    # DATA
    # ========================================================

    def load_data(self):

        # ====================================================
        # CHECK FILES
        # ====================================================

        if not TRANSACTIONS_PATH.exists():

            raise FileNotFoundError(
                "Transaction data not found: "
                f"{TRANSACTIONS_PATH}"
            )

        if not IDENTITY_PATH.exists():

            raise FileNotFoundError(
                "Identity data not found: "
                f"{IDENTITY_PATH}"
            )

        if not HISTORY_PATH.exists():

            raise FileNotFoundError(
                "Investigation history not found: "
                f"{HISTORY_PATH}"
            )

        if not RISK_PATH.exists():

            raise FileNotFoundError(
                "Risk assessments not found: "
                f"{RISK_PATH}"
            )

        # ====================================================
        # LOAD RAW TRANSACTIONS
        # ====================================================

        print(
            "\nLoading ORIGINAL transaction data..."
        )

        self.raw_transactions = pd.read_csv(
            TRANSACTIONS_PATH
        )

        print(
            "Raw transaction shape:",
            self.raw_transactions.shape
        )

        print(
            "Raw transaction columns:",
            len(
                self.raw_transactions.columns
            )
        )

        # ====================================================
        # LOAD IDENTITY
        # ====================================================

        print(
            "\nLoading identity data..."
        )

        self.identity = pd.read_csv(
            IDENTITY_PATH
        )

        print(
            "Identity shape:",
            self.identity.shape
        )

        print(
            "Identity columns:",
            len(
                self.identity.columns
            )
        )

        # ====================================================
        # MERGE TRANSACTION + IDENTITY
        # ====================================================

        print(
            "\nMerging transaction + identity..."
        )

        if "TransactionID" not in (
            self.raw_transactions.columns
        ):

            raise ValueError(
                "Transaction data does not contain "
                "'TransactionID'."
            )

        if "TransactionID" not in (
            self.identity.columns
        ):

            raise ValueError(
                "Identity data does not contain "
                "'TransactionID'."
            )

        # ----------------------------------------------------
        # Avoid duplicate columns
        # ----------------------------------------------------

        identity_columns = [
            column
            for column in self.identity.columns
            if column != "TransactionID"
        ]

        identity_for_merge = (
            self.identity[
                [
                    "TransactionID",
                    *identity_columns,
                ]
            ]
            .drop_duplicates(
                subset=[
                    "TransactionID"
                ],
                keep="first",
            )
        )

        # ----------------------------------------------------
        # Left merge
        # ----------------------------------------------------

        self.transactions = (
            self.raw_transactions.merge(
                identity_for_merge,
                on="TransactionID",
                how="left",
                sort=False,
            )
        )

        print(
            "Merged transaction shape:",
            self.transactions.shape
        )

        print(
            "Merged transaction columns:",
            len(
                self.transactions.columns
            )
        )

        # ====================================================
        # VERIFY MERGED DATA
        # ====================================================

        print(
            "\n========== MERGED DATA CHECK =========="
        )

        print(
            "Raw columns:",
            len(
                self.raw_transactions.columns
            )
        )

        print(
            "Identity columns:",
            len(
                self.identity.columns
            )
        )

        print(
            "Merged columns:",
            len(
                self.transactions.columns
            )
        )

        if "DeviceInfo" in (
            self.transactions.columns
        ):

            print(
                "DeviceInfo: available"
            )

        else:

            print(
                "DeviceInfo: NOT AVAILABLE"
            )

        print(
            "========================================\n"
        )

        # ====================================================
        # LOAD INVESTIGATION HISTORY
        # ====================================================

        print(
            "Loading investigation history..."
        )

        self.investigation_history = (
            pd.read_csv(
                HISTORY_PATH
            )
        )

        print(
            "Investigation history shape:",
            self.investigation_history.shape
        )

        # ====================================================
        # LOAD RISK ASSESSMENTS
        # ====================================================

        print(
            "\nLoading risk assessments..."
        )

        with open(
            RISK_PATH,
            "r"
        ) as f:

            self.risk_assessments = (
                json.load(f)
            )

        # JSON object keys are strings.
        # Convert transaction IDs back to integers.

        self.risk_assessments = {

            int(transaction_id):
                assessment

            for (
                transaction_id,
                assessment
            )
            in self.risk_assessments.items()
        }

        print(
            "Risk assessments loaded:",
            len(
                self.risk_assessments
            )
        )

    # ========================================================
    # LOAD EVERYTHING
    # ========================================================

    def load_all(self):

        print(
            "\n"
            "=========================================="
        )

        print(
            "INITIALIZING APPLICATION CONTAINER"
        )

        print(
            "=========================================="
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.load_model()

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        self.load_data()

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        print(
            "\n========== CONTAINER VALIDATION =========="
        )

        print(
            "Model loaded:",
            self.model is not None
        )

        print(
            "Features loaded:",
            self.features is not None
        )

        print(
            "Raw transactions loaded:",
            self.raw_transactions is not None
        )

        print(
            "Identity loaded:",
            self.identity is not None
        )

        print(
            "Merged transactions loaded:",
            self.transactions is not None
        )

        print(
            "Investigation history loaded:",
            self.investigation_history is not None
        )

        print(
            "Risk assessments loaded:",
            self.risk_assessments is not None
        )

        if self.raw_transactions is not None:

            print(
                "Raw transaction rows:",
                len(
                    self.raw_transactions
                )
            )

            print(
                "Raw transaction columns:",
                len(
                    self.raw_transactions.columns
                )
            )

        if self.identity is not None:

            print(
                "Identity rows:",
                len(
                    self.identity
                )
            )

            print(
                "Identity columns:",
                len(
                    self.identity.columns
                )
            )

        if self.transactions is not None:

            print(
                "Merged rows:",
                len(
                    self.transactions
                )
            )

            print(
                "Merged columns:",
                len(
                    self.transactions.columns
                )
            )

        print(
            "==========================================\n"
        )

        return self

    # ========================================================
    # READY
    # ========================================================

    def is_ready(self) -> bool:

        return (

            self.model is not None

            and self.features is not None

            and self.raw_transactions is not None

            and self.identity is not None

            and self.transactions is not None

            and self.investigation_history is not None

            and self.risk_assessments is not None
        )