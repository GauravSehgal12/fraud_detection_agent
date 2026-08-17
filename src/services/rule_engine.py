from typing import Any
from src.config import settings


class RuleEngine:

    def __init__(
        self,
        high_risk_model_threshold: float = settings.HIGH_RISK_MODEL_THRESHOLD,
        medium_risk_model_threshold: float = settings.MEDIUM_RISK_MODEL_THRESHOLD,
        high_amount_threshold: float = settings.HIGH_AMOUNT_THRESHOLD,
        device_unique_card_threshold: int = settings.DEVICE_UNIQUE_CARD_THRESHOLD,
        card_1h_velocity_threshold: int = settings.CARD_1H_VELOCITY_THRESHOLD,
        card_24h_velocity_threshold: int = settings.CARD_24H_VELOCITY_THRESHOLD,
        unusual_amount_ratio_threshold: float = settings.UNUSUAL_AMOUNT_RATIO_THRESHOLD,
    ):
        self.high_risk_model_threshold = high_risk_model_threshold
        self.medium_risk_model_threshold = medium_risk_model_threshold
        self.high_amount_threshold = high_amount_threshold
        self.device_unique_card_threshold = device_unique_card_threshold
        self.card_1h_velocity_threshold = card_1h_velocity_threshold
        self.card_24h_velocity_threshold = card_24h_velocity_threshold
        self.unusual_amount_ratio_threshold = unusual_amount_ratio_threshold

    def evaluate(
        self,
        transaction: dict[str, Any],
        model_risk_score: float | None = None,
        cold_start_status: dict[str, Any] | None = None,
        features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate deterministic behavioral and risk rules.

        Returns:
            {
                "rules_triggered": [
                    {
                        "rule_id": str,
                        "severity": "HIGH" | "MEDIUM" | "LOW",
                        "reason": str
                    }
                ],
                "behavioral_risk_score": float,
                "behavioral_risk_level": "HIGH" | "MEDIUM" | "LOW"
            }
        """
        rules_triggered = []

        if cold_start_status is None:
            cold_start_status = {}

        if features is None:
            features = {}

        is_new_card = cold_start_status.get("is_new_card", False)
        is_new_device = cold_start_status.get("is_new_device", False)
        card_history_available = cold_start_status.get("card_history_available", False)

        amount = transaction.get("TransactionAmt", features.get("TransactionAmt", 0.0))
        try:
            amount = float(amount)
        except (ValueError, TypeError):
            amount = 0.0

        # 1. Model Risk Rules
        if model_risk_score is not None:
            if model_risk_score >= self.high_risk_model_threshold:
                rules_triggered.append(
                    {
                        "rule_id": "HIGH_MODEL_RISK",
                        "severity": "HIGH",
                        "reason": f"Model risk score ({model_risk_score:.4f}) meets or exceeds high threshold ({self.high_risk_model_threshold}).",
                    }
                )
            elif model_risk_score >= self.medium_risk_model_threshold:
                rules_triggered.append(
                    {
                        "rule_id": "MEDIUM_MODEL_RISK",
                        "severity": "MEDIUM",
                        "reason": f"Model risk score ({model_risk_score:.4f}) meets or exceeds medium threshold ({self.medium_risk_model_threshold}).",
                    }
                )

        # 2. Cold-start & Combination Rules
        if is_new_card and is_new_device:
            if amount >= self.high_amount_threshold:
                rules_triggered.append(
                    {
                        "rule_id": "NEW_CARD_NEW_DEVICE_HIGH_AMOUNT",
                        "severity": "HIGH",
                        "reason": f"Transaction uses both a new card and a new device with a high amount (${amount:.2f} >= ${self.high_amount_threshold:.2f}).",
                    }
                )
            else:
                rules_triggered.append(
                    {
                        "rule_id": "NEW_CARD_NEW_DEVICE",
                        "severity": "HIGH",
                        "reason": "The card and device have no prior observed history.",
                    }
                )
        else:
            if is_new_card:
                rules_triggered.append(
                    {
                        "rule_id": "NEW_CARD",
                        "severity": "MEDIUM",
                        "reason": "The payment card has no prior observed history.",
                    }
                )
            if is_new_device:
                rules_triggered.append(
                    {
                        "rule_id": "NEW_DEVICE",
                        "severity": "MEDIUM",
                        "reason": "The transaction device has no prior observed history.",
                    }
                )

        # 3. Velocity Rules
        card_txn_1h = features.get("card_txn_count_1h", 0)
        card_txn_24h = features.get("card_txn_count_24h", 0)
        if card_txn_1h >= self.card_1h_velocity_threshold or card_txn_24h >= self.card_24h_velocity_threshold:
            rules_triggered.append(
                {
                    "rule_id": "HIGH_CARD_VELOCITY",
                    "severity": "HIGH",
                    "reason": f"High card transaction velocity detected (1h count: {card_txn_1h}, 24h count: {card_txn_24h}).",
                }
            )

        # 4. Device Sharing Rule
        device_unique_cards = features.get("device_profile_unique_cards", 0)
        if device_unique_cards >= self.device_unique_card_threshold:
            rules_triggered.append(
                {
                    "rule_id": "SHARED_DEVICE",
                    "severity": "HIGH",
                    "reason": f"Device is associated with multiple unique cards ({device_unique_cards} >= {self.device_unique_card_threshold}).",
                }
            )

        # 5. Unusual Amount Rule (Only evaluated if card history exists)
        if card_history_available:
            amount_vs_card_avg = features.get("amount_vs_card_avg", 1.0)
            if amount_vs_card_avg >= self.unusual_amount_ratio_threshold:
                rules_triggered.append(
                    {
                        "rule_id": "UNUSUAL_AMOUNT",
                        "severity": "MEDIUM",
                        "reason": f"Transaction amount is significantly higher than historical card average (ratio: {amount_vs_card_avg:.2f}).",
                    }
                )

        # Compute Behavioral Risk Score & Level
        high_severities = [r for r in rules_triggered if r["severity"] == "HIGH"]
        medium_severities = [r for r in rules_triggered if r["severity"] == "MEDIUM"]

        if len(high_severities) >= 2:
            behavioral_risk_score = 0.90
            behavioral_risk_level = "HIGH"
        elif len(high_severities) == 1:
            behavioral_risk_score = 0.85
            behavioral_risk_level = "HIGH"
        elif len(medium_severities) >= 2:
            behavioral_risk_score = 0.65
            behavioral_risk_level = "MEDIUM"
        elif len(medium_severities) == 1:
            behavioral_risk_score = 0.50
            behavioral_risk_level = "MEDIUM"
        else:
            behavioral_risk_score = 0.0
            behavioral_risk_level = "LOW"

        return {
            "rules_triggered": rules_triggered,
            "behavioral_risk_score": round(behavioral_risk_score, 2),
            "behavioral_risk_level": behavioral_risk_level,
        }
