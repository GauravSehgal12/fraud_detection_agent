from typing import Optional

from src.services.model_loader import ModelLoader


class AppContainer:

    def __init__(self):
        self.model = None
        self.features = None
        self.agent = None

    def load_model(self):

        loader = ModelLoader()

        loaded = loader.load_all()

        self.model = loaded["model"]
        self.features = loaded["features"]

    def is_ready(self) -> bool:

        return (
            self.model is not None
            and self.features is not None
            and self.agent is not None
        )