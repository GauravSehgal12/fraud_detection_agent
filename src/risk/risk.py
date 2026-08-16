def classify_risk(
    probability: float
) -> str:

    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


