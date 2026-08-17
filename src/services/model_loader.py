import json
from pathlib import Path

import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    
    / "models"
    / "fraud_xgboost.json"
)

FEATURE_PATH = (
    PROJECT_ROOT
    
    / "models"
    / "features.json"
)


class ModelLoader:

    def __init__(self):

        self.model = None
        self.features = None

    def load_model(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        model = xgb.XGBClassifier()

        model.load_model(
            MODEL_PATH
        )

        self.model = model

        return self.model

    def load_features(self):

        if not FEATURE_PATH.exists():

            raise FileNotFoundError(
                f"Feature configuration not found: "
                f"{FEATURE_PATH}"
            )

        with open(
            FEATURE_PATH,
            "r"
        ) as f:

            self.features = json.load(f)

        return self.features

    def load_all(self):

        self.load_model()
        self.load_features()

        return {
            "model": self.model,
            "features": self.features
        }