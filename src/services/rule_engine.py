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

        Cold-start design principle:
        - New card/device status is context, not proof of fraud.
        - A new card + new device combination is NOT automatically HIGH risk.
        - A high transaction amount alone is NOT enough to make a cold-start
          transaction HIGH risk.
        - Stronger behavioral evidence such as velocity, shared-device usage,
          or an unusual amount relative to an established card history is kept.
        """
        rules_triggered = []

        if cold_start_status is None:
            cold_start_status = {}
        if features is None:
            features = {}

        is_new_card = bool(cold_start_status.get("is_new_card", False))
        is_new_device = bool(cold_start_status.get("is_new_device", False))
        card_history_available = bool(cold_start_status.get("card_history_available", False))

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

        # 2. Conservative Cold-start Rules
        # New card/device status by itself is not treated as high risk.
        if is_new_card and is_new_device:
            # Do NOT emit the old HIGH NEW_CARD_NEW_DEVICE rule. A completely
            # unseen pair is common in legitimate traffic and has no historical
            # evidence by definition.
            if amount >= self.high_amount_threshold:
                rules_triggered.append(
                    {
                        "rule_id": "NEW_CARD_NEW_DEVICE_HIGH_AMOUNT",
                        "severity": "MEDIUM",
                        "reason": f"Transaction uses both a new card and a new device with a high amount (${amount:.2f} >= ${self.high_amount_threshold:.2f}); no prior history exists, so this is treated as a moderate signal rather than automatic high risk.",
                    }
                )
        else:
            if is_new_card:
                rules_triggered.append(
                    {
                        "rule_id": "NEW_CARD",
                        "severity": "MEDIUM",
                        "reason": "The payment card has no prior observed history; this is treated as a contextual signal, not proof of fraud.",
                    }
                )
            if is_new_device:
                rules_triggered.append(
                    {
                        "rule_id": "NEW_DEVICE",
                        "severity": "MEDIUM",
                        "reason": "The transaction device has no prior observed history; this is treated as a contextual signal, not proof of fraud.",
                    }
                )

        # 3. Velocity Rules
        card_txn_1h = features.get("card_txn_count_1h", 0)
        card_txn_24h = features.get("card_txn_count_24h", 0)
        try:
            card_txn_1h = float(card_txn_1h)
        except (ValueError, TypeError):
            card_txn_1h = 0.0
        try:
            card_txn_24h = float(card_txn_24h)
        except (ValueError, TypeError):
            card_txn_24h = 0.0

        if (
            card_txn_1h >= self.card_1h_velocity_threshold
            or card_txn_24h >= self.card_24h_velocity_threshold
        ):
            rules_triggered.append(
                {
                    "rule_id": "HIGH_CARD_VELOCITY",
                    "severity": "HIGH",
                    "reason": f"High card transaction velocity detected (1h count: {card_txn_1h:g}, 24h count: {card_txn_24h:g}).",
                }
            )

        # 4. Device Sharing Rule
        device_unique_cards = features.get("device_profile_unique_cards", 0)
        try:
            device_unique_cards = float(device_unique_cards)
        except (ValueError, TypeError):
            device_unique_cards = 0.0

        if device_unique_cards >= self.device_unique_card_threshold:
            rules_triggered.append(
                {
                    "rule_id": "SHARED_DEVICE",
                    "severity": "HIGH",
                    "reason": f"Device is associated with multiple unique cards ({device_unique_cards:g} >= {self.device_unique_card_threshold}).",
                }
            )

        # 5. Personalized Unusual Amount Rule
        # Only use amount_vs_card_avg when historical card behavior exists.
        # This avoids penalizing genuinely new cards merely because they have
        # no baseline average.
        if card_history_available:
            amount_vs_card_avg = features.get("amount_vs_card_avg", 1.0)
            try:
                amount_vs_card_avg = float(amount_vs_card_avg)
            except (ValueError, TypeError):
                amount_vs_card_avg = 1.0

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
