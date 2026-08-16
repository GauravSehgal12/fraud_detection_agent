import pandas as pd
import numpy as np


def add_card_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create historical card-level behavioral features.

    IMPORTANT:
    Features are calculated using only transactions
    that occurred BEFORE the current transaction.
    """

    df = df.sort_values(
        ["card1", "TransactionDT"]
    ).copy()

    # Previous transaction count for this card
    df["card_transaction_count"] = (
        df.groupby("card1")
        .cumcount()
    )

    # Previous average transaction amount
    card_amount_sum = (
        df.groupby("card1")["TransactionAmt"]
        .cumsum()
    )

    df["card_avg_amount"] = (
        card_amount_sum - df["TransactionAmt"]
    ) / df["card_transaction_count"].replace(
        0, np.nan
    )

    # Amount relative to historical card average
    df["amount_vs_card_avg"] = (
        df["TransactionAmt"] /
        df["card_avg_amount"]
    )

    # First transaction for this card?
    df["new_card"] = (
        df["card_transaction_count"] == 0
    ).astype("int8")

    return df