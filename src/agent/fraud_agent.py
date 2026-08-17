from typing import Any

from src.agent.tools import FraudInvestigationTools


class FraudInvestigationAgent:

    def __init__(
        self,
        tools: FraudInvestigationTools
    ):
        self.tools = tools

    def investigate(
        self,
        transaction_id: int
    ) -> dict[str, Any]:

       

        transaction = self.tools.get_transaction(
            transaction_id
        )

        if "error" in transaction:
            return transaction

        

        risk = self.tools.get_risk_assessment(
            transaction_id
        )

        if "error" in risk:
            return risk


        card_history = self.tools.get_card_history(
            transaction["card1"],
            transaction["TransactionDT"]
        )

       

        device_history = None

        if transaction["DeviceInfo"]:

            device_history = (
                self.tools.get_device_history(
                    transaction["DeviceInfo"],
                    transaction["TransactionDT"]
                )
            )


        return {
            "transaction": transaction,
            "risk_assessment": risk,
            "card_history": card_history,
            "device_history": device_history
        }