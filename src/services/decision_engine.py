from typing import Any


class DecisionEngine:
    """Final deterministic policy over a continuous fused risk score."""

    def __init__(
        self,
        review_threshold: float = 0.50,
        high_threshold: float = 0.80,
    ):
        self.review_threshold = float(review_threshold)
        self.high_threshold = float(high_threshold)

    def decide(
        self,
        model_risk_score: float,
        model_risk_level: str | None = None,
        behavioral_risk_level: str | None = None,
        cold_start_status: dict[str, Any] | None = None,
        rules_triggered: list[dict[str, Any]] | None = None,
        final_risk_score: float | None = None,
    ) -> str:
        """
        Return APPROVE or REVIEW from the continuous fused score.

        ``final_risk_score`` is preferred. The older model/behavioral arguments
        remain accepted for backward compatibility with existing callers.
        """
        if final_risk_score is not None:
            return "REVIEW" if float(final_risk_score) >= self.review_threshold else "APPROVE"

        # Backward-compatible fallback for callers that have not migrated to fusion.
        if float(model_risk_score) >= self.review_threshold:
            return "REVIEW"
        if behavioral_risk_level in {"HIGH", "MEDIUM"}:
            return "REVIEW"
        return "APPROVE"

    def classify(self, final_risk_score: float) -> str:
        score = float(final_risk_score)
        if score >= self.high_threshold:
            return "HIGH"
        if score >= self.review_threshold:
            return "MEDIUM"
        return "LOW"
