import json
from pathlib import Path

import pandas as pd

from src.services.model_loader import ModelLoader
from src.services.cold_start_detector import ColdStartDetector
from src.services.rule_engine import RuleEngine
from src.services.decision_engine import DecisionEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSACTIONS_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
IDENTITY_PATH = PROJECT_ROOT / "data" / "raw" / "train_identity.csv"
HISTORY_PATH = PROJECT_ROOT / "data" / "investigation_history.csv"
RISK_PATH = PROJECT_ROOT / "data" / "risk_assessments.json"

# Only these transaction columns are required at runtime for historical
# behavior, cold-start detection, and transaction lookup. The complete
# original transaction schema is read from the CSV header only so that
# missing_value_count can still use the original 394-column schema.
HISTORICAL_TRANSACTION_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
]

IDENTITY_COLUMNS = ["TransactionID", "DeviceInfo"]

# Explicit dtypes prevent pandas from widening numeric columns while reading
# and substantially reduce the peak memory required by the Docker container.
TRANSACTION_DTYPES = {
    "TransactionID": "int64",
    "TransactionDT": "int64",
    "TransactionAmt": "float32",
    "card1": "int32",
    "card2": "float32",
    "card3": "float32",
    "card4": "string",
    "card5": "float32",
    "card6": "string",
    "addr1": "float32",
    "addr2": "float32",
}

IDENTITY_DTYPES = {"TransactionID": "int64", "DeviceInfo": "string"}


class AppContainer:
    def __init__(self):
        self.model = None
        self.features = None
        self.raw_transactions = None
        self.identity = None
        self.transactions = None
        self.raw_transaction_columns = None
        self.investigation_history = None
        self.risk_assessments = None
        self.cold_start_detector = None
        self.rule_engine = None
        self.decision_engine = None
        self.agent = None

    def load_model(self):
        print("\nLoading model...")
        loader = ModelLoader()
        loaded = loader.load_all()
        self.model = loaded["model"]
        self.features = loaded["features"]
        print("XGBoost model loaded.")
        print(f"Model features: {len(self.features)}")

    def load_data(self):
        for path, label in [
            (TRANSACTIONS_PATH, "Transaction data"),
            (IDENTITY_PATH, "Identity data"),
            (HISTORY_PATH, "Investigation history"),
            (RISK_PATH, "Risk assessments"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        print("\nLoading ORIGINAL transaction schema...")
        self.raw_transaction_columns = pd.read_csv(
            TRANSACTIONS_PATH,
            nrows=0,
            engine="c",
        ).columns.tolist()
        print(f"Original transaction columns: {len(self.raw_transaction_columns)}")

        missing_runtime_columns = [
            column
            for column in HISTORICAL_TRANSACTION_COLUMNS
            if column not in self.raw_transaction_columns
        ]
        if missing_runtime_columns:
            raise ValueError(
                "Transaction CSV is missing runtime columns: "
                f"{missing_runtime_columns}"
            )

        print("\nLoading COMPACT historical transaction data...")
        # Do not use the pyarrow CSV engine here. It can create a large native
        # allocation for this CSV even when usecols is small. The C parser with
        # explicit dtypes has a much lower peak memory footprint in this
        # container and still loads only 11 of the original 394 columns.
        self.raw_transactions = pd.read_csv(
            TRANSACTIONS_PATH,
            usecols=HISTORICAL_TRANSACTION_COLUMNS,
            dtype=TRANSACTION_DTYPES,
            engine="c",
            low_memory=True,
        )
        print(f"Compact transaction shape: {self.raw_transactions.shape}")
        print(f"Transaction columns loaded: {len(self.raw_transactions.columns)}")

        print("\nLoading COMPACT identity data...")
        available_identity_columns = pd.read_csv(
            IDENTITY_PATH,
            nrows=0,
            engine="c",
        ).columns.tolist()
        missing_identity_columns = [
            column for column in IDENTITY_COLUMNS
            if column not in available_identity_columns
        ]
        if missing_identity_columns:
            raise ValueError(
                "Identity CSV is missing runtime columns: "
                f"{missing_identity_columns}"
            )

        self.identity = pd.read_csv(
            IDENTITY_PATH,
            usecols=IDENTITY_COLUMNS,
            dtype=IDENTITY_DTYPES,
            engine="c",
            low_memory=True,
        )
        print(f"Identity shape: {self.identity.shape}")

        print("\nAttaching DeviceInfo to compact transaction history...")
        identity_device = self.identity.drop_duplicates(
            subset=["TransactionID"],
            keep="first",
        )
        self.transactions = self.raw_transactions.merge(
            identity_device,
            on="TransactionID",
            how="left",
            sort=False,
        )

        print(f"Runtime transaction shape: {self.transactions.shape}")
        print(f"Runtime transaction columns: {len(self.transactions.columns)}")
        print(
            "Runtime schema: compact history + DeviceInfo; "
            f"original schema retained separately ({len(self.raw_transaction_columns)} columns)"
        )

        print("\nLoading investigation history...")
        self.investigation_history = pd.read_csv(HISTORY_PATH)

        print("\nLoading risk assessments...")
        with open(RISK_PATH, "r", encoding="utf-8") as f:
            raw_risk = json.load(f)
        self.risk_assessments = {int(k): v for k, v in raw_risk.items()}

    def init_services(self):
        print("\nInitializing ColdStartDetector...")
        self.cold_start_detector = ColdStartDetector(
            transactions=self.transactions,
            identity=self.identity,
        )
        print("Cold-start detector initialized")

        print("Initializing RuleEngine...")
        self.rule_engine = RuleEngine()
        print("Rule engine initialized")

        print("Initializing DecisionEngine...")
        self.decision_engine = DecisionEngine()
        print("Decision engine initialized")

    def load_all(self):
        print("\n==========================================")
        print("INITIALIZING APPLICATION CONTAINER")
        print("==========================================")

        self.load_model()
        self.load_data()
        self.init_services()

        print("\n========== CONTAINER VALIDATION ==========")
        print("Model loaded:", self.model is not None)
        print("Features loaded:", len(self.features) if self.features else 0)
        print("Raw transactions loaded:", self.raw_transactions.shape if self.raw_transactions is not None else None)
        print("Identity loaded:", self.identity.shape if self.identity is not None else None)
        print("Merged runtime transactions loaded:", self.transactions.shape if self.transactions is not None else None)
        print("Original transaction schema columns:", len(self.raw_transaction_columns or []))
        print("Cold-start detector ready:", self.cold_start_detector is not None)
        print("Rule engine ready:", self.rule_engine is not None)
        print("Decision engine ready:", self.decision_engine is not None)
        print("==========================================\n")
        return self

    def is_ready(self) -> bool:
        return (
            self.model is not None
            and self.features is not None
            and self.raw_transactions is not None
            and self.identity is not None
            and self.transactions is not None
            and self.cold_start_detector is not None
            and self.rule_engine is not None
            and self.decision_engine is not None
        )
