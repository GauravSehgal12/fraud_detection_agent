import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):
    """
    Split transactions chronologically.

    Train      = earliest 70%
    Validation = next 15%
    Test       = latest 15%
    """

    df = df.sort_values("TransactionDT").reset_index(drop=True)

    n = len(df)

    train_end = int(n * train_ratio)
    validation_end = int(
        n * (train_ratio + validation_ratio)
    )

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    return train_df, validation_df, test_df