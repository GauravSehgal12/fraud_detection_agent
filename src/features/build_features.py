import pandas as pd

from .feature_engineering import add_basic_features
from .behavioural_features import (
    add_card_behavior_features,
    add_card_velocity_features,
    add_device_profile_features,
    add_card_device_features,
)


def build_historical_features(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build leakage-safe historical features.

    For every transaction, behavioral features are calculated
    using only transactions that occurred before it.
    """

    # --------------------------------------------------
    # 1. Basic transaction features
    # --------------------------------------------------

    merged_df = transactions.merge(
        identity,
        on="TransactionID",
        how="left"
    )
    
    df = add_basic_features(
        merged_df,
        historical_df=merged_df,
        identity=identity,
    )

    # --------------------------------------------------
    # 2. Add DeviceInfo
    # --------------------------------------------------

    device_data = identity[
        [
            "TransactionID",
            "DeviceInfo"
        ]
    ].copy()

    df = df.merge(
        device_data,
        on="TransactionID",
        how="left"
    )

    # --------------------------------------------------
    # 3. Card behavioral features
    # --------------------------------------------------

    df = add_card_behavior_features(df)

    # --------------------------------------------------
    # 4. Card velocity
    # --------------------------------------------------

    df = add_card_velocity_features(df)

    # --------------------------------------------------
    # 5. Device profile features
    # --------------------------------------------------

    df = add_device_profile_features(df)

    # --------------------------------------------------
    # 6. Card-device relationship features
    # --------------------------------------------------

    df = add_card_device_features(df)

    # --------------------------------------------------
    # 7. Restore chronological order
    # --------------------------------------------------

    df = (
        df.sort_values("TransactionDT")
        .reset_index(drop=True)
    )

    return df