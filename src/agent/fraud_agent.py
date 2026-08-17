import json
from typing import Any

from src.agent.tools import FraudInvestigationTools
from src.agent.llm import FraudLLM
from src.agent.prompt import INVESTIGATION_PROMPT

from src.guardrails.output_guardrails import OutputGuardrail
from src.guardrails.input_guardrails import InputGuardrail


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
        default=str,
    )


class FraudInvestigationAgent:

    def __init__(
        self,
        tools: FraudInvestigationTools,
        llm: FraudLLM,
    ):
        """
        Initialize the fraud investigation agent.
        """

        self.tools = tools
        self.llm = llm

        # Input validation
        self.input_guardrail = InputGuardrail()

        # Output validation
        self.guardrail = OutputGuardrail()

    # =====================================================
    # INVESTIGATION
    # =====================================================

    def investigate(
        self,
        transaction: int | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Collect deterministic evidence for a transaction.

        Supports:

        1. Existing transaction:

            investigate(3409570)

        2. New transaction:

            investigate({
                "TransactionID": 999999999,
                ...
            })

        The LLM is NOT involved in this step.
        """

        # =================================================
        # EXISTING TRANSACTION
        # =================================================

        if isinstance(
            transaction,
            int,
        ):

            transaction_id = (
                self.input_guardrail
                .validate_transaction_id(
                    transaction
                )
            )

            # ---------------------------------------------
            # 1. Get transaction
            # ---------------------------------------------

            transaction_data = (
                self.tools.get_transaction(
                    transaction_id
                )
            )

            if (
                isinstance(
                    transaction_data,
                    dict,
                )
                and "error"
                in transaction_data
            ):
                return transaction_data

            # ---------------------------------------------
            # 2. Risk assessment
            # ---------------------------------------------

            risk_assessment = (
                self.tools.get_risk_assessment(
                    transaction_id
                )
            )

            if (
                isinstance(
                    risk_assessment,
                    dict,
                )
                and "error"
                in risk_assessment
            ):
                return risk_assessment

            # ---------------------------------------------
            # 3. Card history
            # ---------------------------------------------

            card_history = (
                self.tools.get_card_history(
                    transaction_id
                )
            )

            if (
                isinstance(
                    card_history,
                    dict,
                )
                and "error"
                in card_history
            ):
                return card_history

            # ---------------------------------------------
            # 4. Device history
            # ---------------------------------------------

            device_history = (
                self.tools.get_device_history(
                    transaction_id
                )
            )

            if (
                isinstance(
                    device_history,
                    dict,
                )
                and "error"
                in device_history
            ):
                return device_history

            return {
                "transaction":
                    transaction_data,

                "risk_assessment":
                    risk_assessment,

                "card_history":
                    card_history,

                "device_history":
                    device_history,
            }

        # =================================================
        # NEW TRANSACTION
        # =================================================

        if isinstance(
            transaction,
            dict,
        ):

            transaction_id = transaction.get(
                "TransactionID"
            )

            if transaction_id is None:

                return {
                    "error":
                        "TransactionID is required."
                }

            try:

                transaction_id = (
                    self.input_guardrail
                    .validate_transaction_id(
                        int(transaction_id)
                    )
                )

            except Exception as exc:

                return {
                    "error":
                        f"Invalid TransactionID: {exc}"
                }

            # ---------------------------------------------
            # NEW TRANSACTION
            #
            # The tools layer must handle the dictionary
            # for dynamic feature generation.
            # ---------------------------------------------

            transaction_data = (
                self.tools.get_transaction(
                    transaction
                )
            )

            if (
                isinstance(
                    transaction_data,
                    dict,
                )
                and "error"
                in transaction_data
            ):
                return transaction_data

            # ---------------------------------------------
            # Dynamic risk assessment
            # ---------------------------------------------

            risk_assessment = (
                self.tools.get_risk_assessment(
                    transaction
                )
            )

            if (
                isinstance(
                    risk_assessment,
                    dict,
                )
                and "error"
                in risk_assessment
            ):
                return risk_assessment

            # ---------------------------------------------
            # Card history
            # ---------------------------------------------

            card_history = (
                self.tools.get_card_history(
                    transaction
                )
            )

            if (
                isinstance(
                    card_history,
                    dict,
                )
                and "error"
                in card_history
            ):
                return card_history

            # ---------------------------------------------
            # Device history
            # ---------------------------------------------

            device_history = (
                self.tools.get_device_history(
                    transaction
                )
            )

            if (
                isinstance(
                    device_history,
                    dict,
                )
                and "error"
                in device_history
            ):
                return device_history

            return {
                "transaction":
                    transaction_data,

                "risk_assessment":
                    risk_assessment,

                "card_history":
                    card_history,

                "device_history":
                    device_history,
            }

        # =================================================
        # INVALID INPUT
        # =================================================

        return {
            "error": (
                "Transaction must be either "
                "an integer transaction ID or "
                "a transaction dictionary."
            )
        }

    # =====================================================
    # REPORT GENERATION
    # =====================================================

    def generate_report(
        self,
        transaction: int | dict[str, Any],
    ) -> str:
        """
        Generate a human-readable fraud investigation
        report using the LLM.

        Supports both existing and new transactions.
        """

        # ---------------------------------------------
        # Collect deterministic evidence
        # ---------------------------------------------

        investigation = self.investigate(
            transaction
        )

        # ---------------------------------------------
        # Handle investigation error
        # ---------------------------------------------

        if "error" in investigation:

            return investigation["error"]

        # ---------------------------------------------
        # Convert evidence to LLM context
        # ---------------------------------------------

        context = build_llm_context(
            investigation
        )

        # ---------------------------------------------
        # Generate report
        # ---------------------------------------------

        report = self.llm.generate(
            system_prompt=INVESTIGATION_PROMPT,

            user_prompt=(
                "Investigate the following "
                "financial transaction using ONLY "
                "the supplied evidence.\n\n"

                "Do not invent information.\n"
                "Do not modify the risk score.\n"
                "Do not modify the risk level.\n"
                "Do not modify the decision.\n"
                "Do not introduce evidence that is "
                "not present in the supplied data.\n\n"

                f"{context}"
            ),
        )

        # ---------------------------------------------
        # Validate LLM output
        # ---------------------------------------------

        validated_report = (
            self.guardrail.validate(
                report=report,

                risk_assessment=(
                    investigation[
                        "risk_assessment"
                    ]
                ),
            )
        )

        return validated_report