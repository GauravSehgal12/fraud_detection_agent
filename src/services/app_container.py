import json
from pathlib import Path

import pandas as pd

from src.services.model_loader import ModelLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
        self.investigation_history = None
        self.risk_assessments = None
        self.agent = None

    def load_model(self):

        loader = ModelLoader()

        loaded = loader.load_all()

        self.model = loaded["model"]
        self.features = loaded["features"]

    def load_data(self):

        if not HISTORY_PATH.exists():
            raise FileNotFoundError(
                f"Investigation history not found: "
                f"{HISTORY_PATH}"
            )

        if not RISK_PATH.exists():
            raise FileNotFoundError(
                f"Risk assessments not found: "
                f"{RISK_PATH}"
            )

        self.investigation_history = pd.read_csv(
            HISTORY_PATH
        )

        with open(RISK_PATH, "r") as f:
            self.risk_assessments = json.load(f)

        # JSON keys are strings.
        # Convert them back to integer transaction IDs.
        self.risk_assessments = {
            int(transaction_id): assessment
            for transaction_id, assessment
            in self.risk_assessments.items()
        }

    def load_all(self):

        self.load_model()
        self.load_data()

        return self

    def is_ready(self) -> bool:

        return (
            self.model is not None
            and self.features is not None
            and self.investigation_history is not None
            and self.risk_assessments is not None
        )