from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    InvestigationRequest,
    InvestigationResponse
)

router = APIRouter()

agent = None


def set_agent(fraud_agent):

    global agent

    agent = fraud_agent


@router.get("/health")
def health():

    if agent is None:

        return {
            "status": "starting"
        }

    return {
        "status": "healthy"
    }


@router.post(
    "/investigate",
    response_model=InvestigationResponse
)
def investigate(
    request: InvestigationRequest
):

    if agent is None:

        raise HTTPException(
            status_code=503,
            detail="Fraud agent is not initialized."
        )

    try:

        report = agent.generate_report(
            request.transaction_id
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Investigation failed."
        ) from exc

    if report == "Transaction not found.":

        raise HTTPException(
            status_code=404,
            detail="Transaction not found."
        )

    return InvestigationResponse(
        transaction_id=request.transaction_id,
        report=report
    )