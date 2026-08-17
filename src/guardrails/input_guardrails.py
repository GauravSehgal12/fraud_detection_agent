import re


class InputGuardrail:

    def validate_transaction_id(
        self,
        transaction_id
    ) -> int:

    
        if isinstance(
            transaction_id,
            bool
        ):
            raise ValueError(
                "Invalid transaction ID."
            )

        try:

            transaction_id = int(
                transaction_id
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                "Transaction ID must be an integer."
            )

       

        if transaction_id <= 0:

            raise ValueError(
                "Transaction ID must be positive."
            )

        return transaction_id