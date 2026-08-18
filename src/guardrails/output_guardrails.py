import re
from typing import Any


class OutputGuardrail:

    def validate(
        self,
        report: str,
        risk_assessment: dict[str, Any]
    ) -> str:


        forbidden_phrases = [
            "definitely fraudulent",
            "certainly fraudulent",
            "100% fraud",
            "confirmed fraud",
            "guaranteed fraud"
        ]

        report_lower = report.lower()

        for phrase in forbidden_phrases:

            if phrase in report_lower:

                raise ValueError(
                    "Output guardrail blocked an "
                    "unsupported fraud claim."
                )

      

        expected_score = str(
            risk_assessment["risk_score"]
        )

        if expected_score not in report:

            raise ValueError(
                "Output guardrail: risk score "
                "does not match Risk Engine."
            )

        

        expected_level = (
            risk_assessment["risk_level"]
        )

        if expected_level not in report:

            raise ValueError(
                "Output guardrail: risk level "
                "does not match Risk Engine."
            )

      

        expected_decision = (
            risk_assessment["decision"]
        )

        if expected_decision not in report:

            raise ValueError(
                "Output guardrail: decision "
                "does not match Risk Engine."
            )

     

        secret_patterns = [
            r"gsk_[A-Za-z0-9_-]+",
            r"sk-[A-Za-z0-9_-]+"
        ]

        for pattern in secret_patterns:

            if re.search(
                pattern,
                report
            ):

                raise ValueError(
                    "Output guardrail blocked "
                    "potential secret exposure."
                )

        return report