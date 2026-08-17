from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    InvestigationRequest,
    InvestigationResponse,
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
    response_model=InvestigationResponse,
)
def investigate(
    request: InvestigationRequest,
):

    # =====================================================
    # CHECK AGENT
    # =====================================================

    if agent is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Fraud agent is not initialized."
            ),
        )

    # =====================================================
    # CHECK INPUT
    # =====================================================

    if (
        request.transaction_id is None
        and request.transaction is None
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Provide either "
                "'transaction_id' or "
                "'transaction'."
            ),
        )

    # Don't allow both at the same time
    if (
        request.transaction_id is not None
        and request.transaction is not None
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Provide only one of "
                "'transaction_id' or "
                "'transaction'."
            ),
        )

    # =====================================================
    # EXISTING TRANSACTION
    # =====================================================

    if request.transaction_id is not None:

        transaction_input = (
            request.transaction_id
        )

        transaction_id = (
            request.transaction_id
        )

    # =====================================================
    # NEW TRANSACTION
    # =====================================================

    else:

        transaction_input = (
            request.transaction
        )

        if not isinstance(
            transaction_input,
            dict,
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Transaction must be "
                    "a JSON object."
                ),
            )

        transaction_id = (
            transaction_input.get(
                "TransactionID"
            )
        )

        if transaction_id is None:

            raise HTTPException(
                status_code=400,
                detail=(
                    "New transaction must "
                    "contain TransactionID."
                ),
            )

        try:

            transaction_id = int(
                transaction_id
            )

        except (
            TypeError,
            ValueError,
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "TransactionID must "
                    "be an integer."
                ),
            )

        if transaction_id <= 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "TransactionID must "
                    "be greater than 0."
                ),
            )

    # =====================================================
    # GENERATE REPORT
    # =====================================================

    try:

        report = agent.generate_report(
            transaction_input
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Investigation failed."
            ),
        ) from exc

    # =====================================================
    # HANDLE AGENT ERROR
    # =====================================================

    if (
        isinstance(report, str)
        and report == "Transaction not found."
    ):

        raise HTTPException(
            status_code=404,
            detail="Transaction not found.",
        )

    # =====================================================
    # RETURN RESPONSE
    # =====================================================

    return InvestigationResponse(
        transaction_id=int(
            transaction_id
        ),
        report=report,
    )