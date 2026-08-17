from typing import Any

from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):

    # Existing transaction
    transaction_id: int | None = Field(
        default=None,
        gt=0,
        description="Existing transaction ID to investigate",
    )

    # New transaction
    transaction: dict[str, Any] | None = Field(
        default=None,
        description="New transaction data to investigate",
    )


class InvestigationResponse(BaseModel):

    transaction_id: int

    report: str