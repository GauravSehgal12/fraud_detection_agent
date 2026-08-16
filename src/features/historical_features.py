import pandas as pd


HISTORY_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "DeviceInfo",
]


def prepare_history(
    transactions: pd.DataFrame,
    identity: pd.DataFrame
) -> pd.DataFrame:

    history = transactions[
        [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "card1",
        ]
    ].copy()

    device_data = identity[
        [
            "TransactionID",
            "DeviceInfo",
        ]
    ].copy()

    history = history.merge(
        device_data,
        on="TransactionID",
        how="left",
        sort=False,
    )

    return history