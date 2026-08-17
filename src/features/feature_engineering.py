import pandas as pd
import numpy as np


def add_basic_features(
    self,
    historical_df: pd.DataFrame,
    identity: pd.DataFrame,
    raw_columns: list[str] | None = None,
) -> pd.DataFrame:

    df = historical_df.copy()
    
    identity_ids = set(identity["TransactionID"])

    df["has_identity"] = (
        df["TransactionID"].isin(identity_ids)
    ).astype("int8")

   
    df["TransactionAmt_log"] = np.log1p(
        df["TransactionAmt"]
    )

   
    df["transaction_hour"] = (
        (df["TransactionDT"] // 3600) % 24
    ).astype("int8")

    df["transaction_day"] = (
        df["TransactionDT"] // (24 * 3600)
    ).astype("int32")

   

    df["missing_card_info"] = (
        df[
            ["card1", "card2", "card3", "card4",
             "card5", "card6"]
        ]
        .isna()
        .any(axis=1)
        .astype("int8")
    )

    df["missing_address"] = (
        df[["addr1", "addr2"]]
        .isna()
        .any(axis=1)
        .astype("int8")
    )

 

    raw_cols = raw_columns if raw_columns is not None else df.columns.tolist()
    df["missing_value_count"] = (
        df[raw_cols].isna().sum(axis=1).astype("int16")
    )

    return df