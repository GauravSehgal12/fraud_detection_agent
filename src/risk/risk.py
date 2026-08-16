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