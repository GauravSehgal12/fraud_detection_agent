import pytest
from src.services.decision_engine import DecisionEngine


def test_decision_engine_high_model_risk():
    engine = DecisionEngine()
    decision = engine.decide(
        model_risk_score=0.92,
        model_risk_level="HIGH",
        behavioral_risk_level="LOW",
    )
    assert decision == "REVIEW"


def test_decision_engine_low_model_high_behavioral():
    engine = DecisionEngine()
    decision = engine.decide(
        model_risk_score=0.0122,
        model_risk_level="LOW",
        behavioral_risk_level="HIGH",
        rules_triggered=[{"rule_id": "NEW_CARD_NEW_DEVICE", "severity": "HIGH"}],
    )
    assert decision == "REVIEW"


def test_decision_engine_low_model_low_behavioral():
    engine = DecisionEngine()
    decision = engine.decide(
        model_risk_score=0.0122,
        model_risk_level="LOW",
        behavioral_risk_level="LOW",
        rules_triggered=[],
    )
    assert decision == "APPROVE"
