from typing import Any
from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):

    transaction_id: int | None = Field(
        default=None,
        gt=0,
        description="Existing transaction ID to investigate",
    )

    transaction: dict[str, Any] | None = Field(
        default=None,
        description="New transaction data to investigate",
    )


class ModelRiskInfo(BaseModel):
    model_score: float
    model_level: str
    model_decision: str


class BehavioralRiskInfo(BaseModel):
    score: float
    level: str
    rules_triggered: list[dict[str, Any]]


class ColdStartInfo(BaseModel):
    is_new_card: bool
    is_new_device: bool
    is_new_card_device_pair: bool
    card_history_available: bool
    device_history_available: bool


class InvestigationResponse(BaseModel):
    transaction_id: int
    risk: ModelRiskInfo
    behavioral_risk: BehavioralRiskInfo
    final_decision: str
    cold_start: ColdStartInfo
    input_completeness: str
    evidence: list[dict[str, Any]]
    report: str


class FeedbackRequest(BaseModel):
    label: str = Field(
        ...,
        description="Allowed labels: CONFIRMED_FRAUD, FALSE_POSITIVE, LEGITIMATE, NEEDS_MORE_INFORMATION",
    )
    analyst_comment: str | None = Field(
        default=None,
        description="Analyst comment",
    )


class FeedbackResponse(BaseModel):
    status: str
    transaction_id: int
    label: str
    timestamp: str