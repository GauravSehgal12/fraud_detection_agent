from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    InvestigationRequest,
    InvestigationResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModelRiskInfo,
    BehavioralRiskInfo,
    ColdStartInfo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_FILE_PATH = PROJECT_ROOT / "data" / "feedback_history.json"

router = APIRouter()
agent = None


def set_agent(fraud_agent):
    global agent
    agent = fraud_agent


@router.get("/health")
def health():
    if agent is None:
        return {"status": "starting"}
    return {"status": "healthy"}


@router.post(
    "/investigate",
    response_model=InvestigationResponse,
)
def investigate(request: InvestigationRequest):
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Fraud agent is not initialized.",
        )

    if request.transaction_id is None and request.transaction is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'transaction_id' or 'transaction'.",
        )

    if request.transaction_id is not None and request.transaction is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of 'transaction_id' or 'transaction'.",
        )

    if request.transaction_id is not None:
        transaction_input = request.transaction_id
        transaction_id = request.transaction_id
    else:
        transaction_input = request.transaction
        if not isinstance(transaction_input, dict):
            raise HTTPException(
                status_code=400,
                detail="Transaction must be a JSON object.",
            )

        transaction_id = transaction_input.get("TransactionID")
        if transaction_id is None:
            raise HTTPException(
                status_code=400,
                detail="New transaction must contain TransactionID.",
            )

        try:
            transaction_id = int(transaction_id)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="TransactionID must be an integer.",
            )

        if transaction_id <= 0:
            raise HTTPException(
                status_code=400,
                detail="TransactionID must be greater than 0.",
            )

    try:
        investigation_data = agent.investigate(transaction_input)
        report = agent.generate_report(transaction_input)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Investigation failed: {exc}",
        ) from exc

    if isinstance(investigation_data, dict) and "error" in investigation_data:
        err_msg = investigation_data["error"]
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail="Transaction not found.")
        raise HTTPException(status_code=400, detail=err_msg)

    risk_info = investigation_data.get("risk", {})
    behavioral_info = investigation_data.get("behavioral_risk", {})
    cold_start_info = investigation_data.get("cold_start", {})

    return InvestigationResponse(
        transaction_id=int(transaction_id),
        risk=ModelRiskInfo(
            model_score=risk_info.get("model_score", 0.0),
            model_level=risk_info.get("model_level", "LOW"),
            model_decision=risk_info.get("model_decision", "APPROVE"),
        ),
        behavioral_risk=BehavioralRiskInfo(
            score=behavioral_info.get("score", 0.0),
            level=behavioral_info.get("level", "LOW"),
            rules_triggered=behavioral_info.get("rules_triggered", []),
        ),
        final_decision=investigation_data.get("final_decision", "APPROVE"),
        cold_start=ColdStartInfo(
            is_new_card=cold_start_info.get("is_new_card", True),
            is_new_device=cold_start_info.get("is_new_device", True),
            is_new_card_device_pair=cold_start_info.get("is_new_card_device_pair", True),
            card_history_available=cold_start_info.get("card_history_available", False),
            device_history_available=cold_start_info.get("device_history_available", False),
        ),
        input_completeness=investigation_data.get("input_completeness", "COMPLETE"),
        evidence=investigation_data.get("evidence", []),
        report=report,
    )


@router.post(
    "/investigations/{transaction_id}/feedback",
    response_model=FeedbackResponse,
)
def record_feedback(transaction_id: int, feedback: FeedbackRequest):
    allowed_labels = {
        "CONFIRMED_FRAUD",
        "FALSE_POSITIVE",
        "LEGITIMATE",
        "NEEDS_MORE_INFORMATION",
    }
    if feedback.label not in allowed_labels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid label. Must be one of: {sorted(list(allowed_labels))}",
        )

    timestamp_str = datetime.now(timezone.utc).isoformat()
    record = {
        "transaction_id": transaction_id,
        "analyst_label": feedback.label,
        "analyst_comment": feedback.analyst_comment,
        "model_score": feedback.model_score,
        "behavioral_score": feedback.behavioral_score,
        "final_decision": feedback.final_decision,
        "model_version": feedback.model_version,
        "timestamp": timestamp_str,
    }

    # Load existing feedback
    history = []
    if FEEDBACK_FILE_PATH.exists():
        try:
            with open(FEEDBACK_FILE_PATH, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(record)

    # Save updated feedback
    FEEDBACK_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE_PATH, "w") as f:
        json.dump(history, f, indent=2)

    return FeedbackResponse(
        status="recorded",
        transaction_id=transaction_id,
        label=feedback.label,
        timestamp=timestamp_str,
    )