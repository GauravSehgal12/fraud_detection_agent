from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

from src.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "models" / "risk_fusion_config.json"
DEFAULT_CALIBRATOR_PATH = PROJECT_ROOT / "models" / "isotonic_calibrator.joblib"


class RiskFusion:
    """Combine calibrated model risk with deterministic behavioral risk."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        calibrator_path: Path = DEFAULT_CALIBRATOR_PATH,
    ):
        self.config_path = config_path
        self.calibrator_path = calibrator_path
        self.model_weight = float(settings.FUSION_MODEL_WEIGHT)
        self.behavioral_weight = float(settings.FUSION_BEHAVIORAL_WEIGHT)
        self.review_threshold = float(settings.FUSION_REVIEW_THRESHOLD)
        self.high_threshold = float(settings.FUSION_HIGH_THRESHOLD)
        self.calibrator = None
        self.calibration_available = False
        self._load_config()
        self._load_calibrator()

    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
                self.model_weight = float(config.get("model_weight", self.model_weight))
                self.behavioral_weight = float(config.get("behavioral_weight", self.behavioral_weight))
                self.review_threshold = float(config.get("review_threshold", self.review_threshold))
                self.high_threshold = float(config.get("high_threshold", self.high_threshold))
            except (OSError, ValueError, TypeError):
                pass

        total = self.model_weight + self.behavioral_weight
        if total <= 0:
            self.model_weight, self.behavioral_weight = 0.80, 0.20
        else:
            self.model_weight /= total
            self.behavioral_weight /= total

    def _load_calibrator(self) -> None:
        if joblib is None or not self.calibrator_path.exists():
            return
        try:
            self.calibrator = joblib.load(self.calibrator_path)
            self.calibration_available = True
        except Exception:
            self.calibrator = None
            self.calibration_available = False

    def calibrate_model_score(self, model_score: float) -> float:
        score = float(np.clip(model_score, 0.0, 1.0))
        if self.calibrator is None:
            return score
        calibrated = self.calibrator.predict(np.asarray([score], dtype=float))[0]
        return float(np.clip(calibrated, 0.0, 1.0))

    def combine(self, model_score: float, behavioral_score: float) -> dict[str, Any]:
        raw_model_score = float(np.clip(model_score, 0.0, 1.0))
        behavioral_score = float(np.clip(behavioral_score, 0.0, 1.0))
        calibrated_model_score = self.calibrate_model_score(raw_model_score)
        final_score = float(
            np.clip(
                self.model_weight * calibrated_model_score
                + self.behavioral_weight * behavioral_score,
                0.0,
                1.0,
            )
        )

        if final_score >= self.high_threshold:
            level = "HIGH"
        elif final_score >= self.review_threshold:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "raw_model_score": round(raw_model_score, 6),
            "calibrated_model_score": round(calibrated_model_score, 6),
            "behavioral_score": round(behavioral_score, 6),
            "final_risk_score": round(final_score, 6),
            "final_risk_level": level,
            "decision": "REVIEW" if final_score >= self.review_threshold else "APPROVE",
            "model_weight": round(self.model_weight, 6),
            "behavioral_weight": round(self.behavioral_weight, 6),
            "review_threshold": round(self.review_threshold, 6),
            "high_threshold": round(self.high_threshold, 6),
            "calibration_available": self.calibration_available,
        }
