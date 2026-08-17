import json
from typing import Any

from src.agent.tools import FraudInvestigationTools
from src.agent.llm import FraudLLM
from src.agent.prompt import INVESTIGATION_PROMPT


def build_llm_context(
    investigation: dict
) -> str:

    return json.dumps(
        investigation,
        indent=2,
        default=str
    )


class FraudInvestigationAgent:

    def __init__(
        self,
        tools: FraudInvestigationTools,
        llm: FraudLLM
    ):

        self.tools = tools
        self.llm = llm

    def investigate(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

        # 1. Get transaction
        transaction = self.tools.get_transaction(
            transaction_id
        )

        if "error" in transaction:
            return transaction

        # 2. Get risk assessment
        risk = self.tools.get_risk_assessment(
            transaction_id
        )

        if "error" in risk:
            return risk

        # 3. Get card history
        card_history = self.tools.get_card_history(
            transaction["card1"],
            transaction["TransactionDT"]
        )

        # 4. Get device history
        device_history = None

        if transaction["DeviceInfo"]:

            device_history = (
                self.tools.get_device_history(
                    transaction["DeviceInfo"],
                    transaction["TransactionDT"]
                )
            )

        # 5. Return complete investigation
        return {
            "transaction": transaction,
            "risk_assessment": risk,
            "card_history": card_history,
            "device_history": device_history
        }

    def generate_report(
        self,
        transaction_id: int
    ) -> str:

        # Run deterministic investigation
        investigation = self.investigate(
            transaction_id
        )

        # Handle errors
        if "error" in investigation:
            return investigation["error"]

        # Convert investigation data to JSON
        context = build_llm_context(
            investigation
        )

        # Send ONLY the investigation evidence
        # to the LLM
        report = self.llm.generate(
            system_prompt=INVESTIGATION_PROMPT,
            user_prompt=(
                "Investigate this transaction using "
                "ONLY the supplied evidence.\n\n"
                f"{context}"
            )
        )

        return report