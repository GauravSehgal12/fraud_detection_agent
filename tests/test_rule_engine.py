import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.rule_engine import RuleEngine


def test_rule_engine_new_card_new_device_high_amount_is_medium():
    """Cold-start + high amount is a contextual medium signal, not automatic HIGH risk."""
    engine = RuleEngine(high_amount_threshold=500.0)

    result = engine.evaluate(
        transaction={"TransactionAmt": 600.0},
        model_risk_score=0.10,
        cold_start_status={
            "is_new_card": True,
            "is_new_device": True,
            "card_history_available": False,
        },
        features={},
    )

    triggered_ids = [r["rule_id"] for r in result["rules_triggered"]]
    assert "NEW_CARD_NEW_DEVICE_HIGH_AMOUNT" in triggered_ids
    assert result["behavioral_risk_level"] == "MEDIUM"
    assert result["behavioral_risk_score"] == 0.50


def test_rule_engine_high_model_risk():
    engine = RuleEngine(high_risk_model_threshold=0.90)

    result = engine.evaluate(
        transaction={"TransactionAmt": 100.0},
        model_risk_score=0.95,
        cold_start_status={
            "is_new_card": False,
            "is_new_device": False,
            "card_history_available": True,
        },
        features={},
    )

    triggered_ids = [r["rule_id"] for r in result["rules_triggered"]]
    assert "HIGH_MODEL_RISK" in triggered_ids
    assert result["behavioral_risk_level"] == "HIGH"


def test_rule_engine_unusual_amount_requires_history():
    engine = RuleEngine(unusual_amount_ratio_threshold=3.0)

    # Without card history -> UNUSUAL_AMOUNT should NOT trigger
    res_no_history = engine.evaluate(
        transaction={"TransactionAmt": 1000.0},
        model_risk_score=0.05,
        cold_start_status={"card_history_available": False},
        features={"amount_vs_card_avg": 10.0},
    )
    triggered_ids_no_hist = [r["rule_id"] for r in res_no_history["rules_triggered"]]
    assert "UNUSUAL_AMOUNT" not in triggered_ids_no_hist

    # With card history -> UNUSUAL_AMOUNT SHOULD trigger
    res_with_history = engine.evaluate(
        transaction={"TransactionAmt": 1000.0},
        model_risk_score=0.05,
        cold_start_status={"card_history_available": True},
        features={"amount_vs_card_avg": 5.0},
    )
    triggered_ids_with_hist = [r["rule_id"] for r in res_with_history["rules_triggered"]]
    assert "UNUSUAL_AMOUNT" in triggered_ids_with_hist
