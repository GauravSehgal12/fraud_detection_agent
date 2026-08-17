import json
from typing import Any

from src.agent.tools import FraudInvestigationTools
from src.agent.llm import FraudLLM
from src.agent.prompt import INVESTIGATION_PROMPT

from src.guardrails.output_guardrails import OutputGuardrail
from src.guardrails.input_guardrails import InputGuardrail


def build_llm_context(investigation: dict) -> str:
    """
    Convert the deterministic investigation package into a formatted JSON string for the LLM.
    """
    return json.dumps(investigation, indent=2, default=str)


class FraudInvestigationAgent:

    def __init__(
        self,
        tools: FraudInvestigationTools,
        llm: FraudLLM,
    ):
        self.tools = tools
        self.llm = llm

        self.input_guardrail = InputGuardrail()
        self.guardrail = OutputGuardrail()

    def investigate(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Collect deterministic evidence for a transaction.
        The LLM is NOT involved in calculating scores or decisions.
        """
        if isinstance(transaction, int):
            transaction_id = self.input_guardrail.validate_transaction_id(transaction)
            return self.tools.collect_investigation_data(transaction_id)

        if isinstance(transaction, dict):
            transaction_id = transaction.get("TransactionID")
            if transaction_id is None:
                return {"error": "TransactionID is required."}

            try:
                validated_id = self.input_guardrail.validate_transaction_id(int(transaction_id))
            except Exception as exc:
                return {"error": f"Invalid TransactionID: {exc}"}

            return self.tools.collect_investigation_data(transaction)

        return {
            "error": "Transaction must be either an integer transaction ID or a transaction dictionary."
        }

    def generate_report(
        self,
        transaction: int | dict[str, Any],
    ) -> str:
        """
        Generate a human-readable fraud investigation report using the LLM.
        """
        investigation = self.investigate(transaction)

        if "error" in investigation:
            return investigation["error"]

        context = build_llm_context(investigation)

        user_prompt = (
            "Investigate the following financial transaction using ONLY "
            "the supplied evidence.\n\n"
            "Do not invent information.\n"
            "Do not modify the model risk score.\n"
            "Do not modify the model risk level.\n"
            "Do not modify the behavioral risk level.\n"
            "Do not modify the final decision.\n"
            "Do not introduce evidence that is not present in the supplied data.\n\n"
            f"{context}"
        )

        report = self.llm.generate(
            system_prompt=INVESTIGATION_PROMPT,
            user_prompt=user_prompt,
        )

        # Validate LLM output against deterministic values
        risk_summary = {
            "risk_score": investigation.get("risk", {}).get("model_score", 0.0),
            "risk_level": investigation.get("risk", {}).get("model_level", "LOW"),
            "decision": investigation.get("final_decision", "APPROVE"),
        }

        validated_report = self.guardrail.validate(
            report=report,
            risk_assessment=risk_summary,
        )

        return validated_report