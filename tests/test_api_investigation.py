import pytest
from fastapi.testclient import TestClient

from app import app, container
from src.agent.tools import FraudInvestigationTools
from src.agent.fraud_agent import FraudInvestigationAgent
from src.api.route import set_agent


@pytest.fixture(scope="module")
def test_client():
    container.load_all()

    tools = FraudInvestigationTools(
        transactions=container.raw_transactions,
        identity=container.identity,
        risk_assessments=container.risk_assessments,
        model=container.model,
        cold_start_detector=container.cold_start_detector,
        rule_engine=container.rule_engine,
        decision_engine=container.decision_engine,
    )

    class MockLLM:
        model = "mock-model"
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            return (
                "MODEL RISK:\nLOW\n\nMODEL SCORE:\n0.0122\n\nMODEL DECISION:\nAPPROVE\n\n"
                "BEHAVIORAL RISK:\nHIGH\n\nFINAL DECISION:\nREVIEW\n\n"
                "KEY EVIDENCE:\n- Cold start card and device\n\n"
                "CARD BEHAVIOR:\n- No history\n\n"
                "DEVICE BEHAVIOR:\n- No history\n\n"
                "INVESTIGATION SUMMARY:\nMock analysis complete.\n\n"
                "RECOMMENDED ACTION:\nManual Analyst Review Required."
            )

    agent = FraudInvestigationAgent(tools=tools, llm=MockLLM())
    set_agent(agent)

    with TestClient(app) as client:
        yield client, tools


def test_existing_transaction_3409570(test_client):
    client, tools = test_client
    response = client.post("/api/v1/investigate", json={"transaction_id": 3409570})
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == 3409570
    assert "risk" in data
    assert "behavioral_risk" in data
    assert "final_decision" in data
    assert "evidence" in data
    assert len(data["evidence"]) <= 5


def test_validated_synthetic_transaction_999999999(test_client):
    client, tools = test_client

    # Synthetic transaction payload corresponding to historical 3409570
    synthetic_payload = {
        "TransactionID": 999999999,
        "TransactionDT": 10699419,
        "TransactionAmt": 87.302,
        "card1": 12730,
        "DeviceInfo": "LG-D320 Build/KOT49I.V10a",
    }

    # Verify features directly built from tools.feature_builder
    features = tools.feature_builder.build(tools._prepare_new_transaction(synthetic_payload))

    assert features["missing_value_count"] == 95
    assert features["card_transaction_count"] == 157
    assert abs(features["card_avg_amount"] - 45.337777) < 0.1
    assert features["device_profile_count"] == 46
    assert features["device_profile_unique_cards"] == 27
    assert features["device_unique_cards_historical"] == 25

    response = client.post("/api/v1/investigate", json={"transaction": synthetic_payload})
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == 999999999


def test_cold_start_transaction_999999997(test_client):
    client, tools = test_client
    cold_payload = {
        "TransactionID": 999999997,
        "TransactionDT": 10699421,
        "TransactionAmt": 500.0,
        "card1": 99999,
        "DeviceInfo": "UNKNOWN-DEVICE",
    }

    response = client.post("/api/v1/investigate", json={"transaction": cold_payload})
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == 999999997
    assert data["cold_start"]["is_new_card"] is True
    assert data["cold_start"]["is_new_device"] is True
    assert data["cold_start"]["is_new_card_device_pair"] is True
    assert data["final_decision"] == "REVIEW"


def test_missing_transaction_returns_404(test_client):
    client, tools = test_client
    response = client.post("/api/v1/investigate", json={"transaction_id": 9999999999})
    assert response.status_code == 404


def test_invalid_input_returns_400(test_client):
    client, tools = test_client
    response = client.post("/api/v1/investigate", json={"transaction_id": -5})
    assert response.status_code == 422  # Pydantic gt=0 validation error


def test_feedback_endpoint(test_client):
    client, tools = test_client
    feedback_payload = {
        "label": "CONFIRMED_FRAUD",
        "analyst_comment": "Unauthorized card usage confirmed",
    }
    response = client.post("/api/v1/investigations/3409570/feedback", json=feedback_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "recorded"
    assert data["transaction_id"] == 3409570
    assert data["label"] == "CONFIRMED_FRAUD"
