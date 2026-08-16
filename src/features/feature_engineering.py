import pandas as pd
import numpy as np


def add_basic_features(
    transactions: pd.DataFrame,
    identity: pd.DataFrame
) -> pd.DataFrame:

    df = transactions.copy()

    
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

 

    df["missing_value_count"] = (
        df.isna().sum(axis=1)
    ).astype("int16")

    return df