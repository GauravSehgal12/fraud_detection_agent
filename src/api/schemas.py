from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):

    transaction_id: int = Field(
        ...,
        gt=0,
        description="Transaction ID to investigate"
    )


class InvestigationResponse(BaseModel):

    transaction_id: int
    report: str