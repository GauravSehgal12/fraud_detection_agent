import json
from pathlib import Path

import pandas as pd

from src.services.model_loader import ModelLoader
from src.services.cold_start_detector import ColdStartDetector
from src.services.rule_engine import RuleEngine
from src.services.decision_engine import DecisionEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        self.model = None
        self.features = None

        self.raw_transactions = None
        self.identity = None
        self.transactions = None
        self.investigation_history = None
        self.risk_assessments = None

        # Architecture services
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
        if not TRANSACTIONS_PATH.exists():
            raise FileNotFoundError(f"Transaction data not found: {TRANSACTIONS_PATH}")

        if not IDENTITY_PATH.exists():
            raise FileNotFoundError(f"Identity data not found: {IDENTITY_PATH}")

        if not HISTORY_PATH.exists():
            raise FileNotFoundError(f"Investigation history not found: {HISTORY_PATH}")

        if not RISK_PATH.exists():
            raise FileNotFoundError(f"Risk assessments not found: {RISK_PATH}")

        print("\nLoading ORIGINAL transaction data...")
        self.raw_transactions = pd.read_csv(TRANSACTIONS_PATH)
        print(f"Raw transaction shape: {self.raw_transactions.shape}")

        print("\nLoading identity data...")
        self.identity = pd.read_csv(IDENTITY_PATH)
        print(f"Identity shape: {self.identity.shape}")

        print("\nMerging transaction + identity...")
        if "TransactionID" not in self.raw_transactions.columns:
            raise ValueError("Transaction data does not contain 'TransactionID'.")

        if "TransactionID" not in self.identity.columns:
            raise ValueError("Identity data does not contain 'TransactionID'.")

        identity_columns = [
            col for col in self.identity.columns if col != "TransactionID"
        ]

        identity_for_merge = self.identity[
            ["TransactionID", *identity_columns]
        ].drop_duplicates(subset=["TransactionID"], keep="first")

        self.transactions = self.raw_transactions.merge(
            identity_for_merge,
            on="TransactionID",
            how="left",
            sort=False,
        )

        print(f"Merged transaction shape: {self.transactions.shape}")

        print("\nLoading investigation history...")
        self.investigation_history = pd.read_csv(HISTORY_PATH)

        print("\nLoading risk assessments...")
        with open(RISK_PATH, "r") as f:
            raw_risk = json.load(f)

        self.risk_assessments = {
            int(k): v for k, v in raw_risk.items()
        }

    def init_services(self):
        """
        Initialize ColdStartDetector, RuleEngine, and DecisionEngine.
        """
        print("\nInitializing ColdStartDetector...")
        self.cold_start_detector = ColdStartDetector(
            transactions=self.raw_transactions,
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
        print("Merged transactions loaded:", self.transactions.shape if self.transactions is not None else None)
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