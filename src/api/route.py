from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    InvestigationRequest,
    InvestigationResponse
)

router = APIRouter()


# This will be assigned when the application starts
agent = None


def set_agent(fraud_agent):

    global agent

    agent = fraud_agent


@router.get("/health")
def health():

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

    report = agent.generate_report(
        request.transaction_id
    )

    if report == "Transaction not found.":

        raise HTTPException(
            status_code=404,
            detail="Transaction not found."
        )

    return InvestigationResponse(
        transaction_id=request.transaction_id,
        report=report
    )