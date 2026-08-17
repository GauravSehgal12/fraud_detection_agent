import json
from typing import Any

from src.agent.tools import FraudInvestigationTools
from src.agent.llm import FraudLLM
from src.agent.prompt import INVESTIGATION_PROMPT
from src.guardrails.output_guardrails import OutputGuardrail

from src.guardrails.input_guardrails import (
    InputGuardrail
)


def build_llm_context(
    investigation: dict
) -> str:
    """
    Convert the deterministic investigation package
    into a JSON string for the LLM.
    """

    return json.dumps(
        investigation,
        indent=2,
        default=str
    )


class FraudInvestigationAgent:

    def __init__(
        self,
        tools: FraudInvestigationTools,
        llm: FraudLLM,
        input_guardrail: InputGuardrail
    ):
        """
        Initialize the fraud investigation agent.
        """

        self.tools = tools
        self.llm = llm
        self.input_guardrail = input_guardrail
        self.guardrail = OutputGuardrail()

    def investigate(
        self,
        transaction_id: int
    ) -> dict[str, Any]:
        """
        Collect deterministic evidence for a transaction.

        The LLM is NOT involved in this step.
        """

       

        transaction = self.tools.get_transaction(
            transaction_id
        )

        if "error" in transaction:
            return transaction

     

        risk_assessment = (
            self.tools.get_risk_assessment(
                transaction_id
            )
        )

        if "error" in risk_assessment:
            return risk_assessment

       

        card_history = (
            self.tools.get_card_history(
                transaction["card1"],
                transaction["TransactionDT"]
            )
        )

       

        device_history = None

        device_info = transaction.get(
            "DeviceInfo"
        )

        if device_info:

            device_history = (
                self.tools.get_device_history(
                    device_info,
                    transaction["TransactionDT"]
                )
            )

       

        return {
            "transaction": transaction,

            "risk_assessment": risk_assessment,

            "card_history": card_history,

            "device_history": device_history
        }

    def generate_report(
        self,
        transaction_id: int
    ) -> str:
        
        """
        Generate a human-readable fraud investigation
        report using the LLM.

        The LLM only receives evidence collected by
        the deterministic investigation layer.
        """
        transaction_id = (
        self.input_guardrail
        .validate_transaction_id(
            transaction_id
        )
    )

      

        investigation = self.investigate(
            transaction_id
        )

         

        if "error" in investigation:

            return investigation["error"]

        

        context = build_llm_context(
            investigation
        )

       

        report = self.llm.generate(
            system_prompt=INVESTIGATION_PROMPT,

            user_prompt=(
                "Investigate the following transaction "
                "using ONLY the supplied evidence.\n\n"

                "Do not invent information.\n"
                "Do not modify the risk score.\n"
                "Do not modify the risk level.\n"
                "Do not modify the decision.\n\n"

                f"{context}"
            )
        )

     

        validated_report = (
            self.guardrail.validate(
                report=report,
                risk_assessment=(
                    investigation[
                        "risk_assessment"
                    ]
                )
            )
        )

       

        return validated_report