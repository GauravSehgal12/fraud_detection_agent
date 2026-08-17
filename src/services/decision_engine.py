from typing import Any


class DecisionEngine:

    def __init__(self):
        """
        Deterministic decision engine combining ML model risk score,
        behavioral rule evaluation, and cold-start signals.
        """
        pass

    def decide(
        self,
        model_risk_score: float,
        model_risk_level: str,
        behavioral_risk_level: str,
        cold_start_status: dict[str, Any] | None = None,
        rules_triggered: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Determine final action: "APPROVE" or "REVIEW".

        Policy:
        - model risk >= 0.90 -> REVIEW
        - model risk >= 0.70 -> REVIEW
        - model LOW + behavioral HIGH -> REVIEW
        - model LOW + behavioral MEDIUM -> REVIEW
        - model LOW + behavioral LOW -> APPROVE
        """
        if rules_triggered is None:
            rules_triggered = []

        if cold_start_status is None:
            cold_start_status = {}

        # High or Medium model risk always triggers REVIEW
        if model_risk_score >= 0.70 or model_risk_level in ["HIGH", "MEDIUM"]:
            return "REVIEW"

        # Model risk is LOW: check behavioral risk & cold start rules
        if behavioral_risk_level in ["HIGH", "MEDIUM"]:
            return "REVIEW"

        # If NEW_CARD_NEW_DEVICE or NEW_CARD_NEW_DEVICE_HIGH_AMOUNT triggered, escalate to REVIEW
        rule_ids = {r.get("rule_id") for r in rules_triggered}
        if "NEW_CARD_NEW_DEVICE" in rule_ids or "NEW_CARD_NEW_DEVICE_HIGH_AMOUNT" in rule_ids:
            return "REVIEW"

        return "APPROVE"
