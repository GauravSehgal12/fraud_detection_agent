def classify_risk(
    probability: float
) -> str:

    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


def make_risk_decision(
    probability: float
) -> dict:

    risk_level = classify_risk(
        probability
    )

    if risk_level == "HIGH":
        decision = "REVIEW"

    elif risk_level == "MEDIUM":
        decision = "MONITOR"

    else:
        decision = "APPROVE"

    return {
        "risk_score": round(
            float(probability),
            4
        ),
        "risk_level": risk_level,
        "decision": decision,
    }


def build_risk_assessment(
    transaction_id: int,
    probability: float,
    evidence: list[dict]
) -> dict:

    decision = make_risk_decision(
        probability
    )

    return {
        "transaction_id": int(transaction_id),
        **decision,
        "evidence": evidence
    }