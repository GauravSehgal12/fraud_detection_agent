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

def add_card_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical card transaction velocity features.

    TransactionDT is measured in seconds.
    The current transaction is NOT included.
    """

    df = (
        df.sort_values(["card1", "TransactionDT"])
        .reset_index(drop=True)
        .copy()
    )

    velocity_1h = np.zeros(len(df), dtype=np.int32)
    velocity_24h = np.zeros(len(df), dtype=np.int32)

    for _, group in df.groupby("card1", sort=False):

        times = group["TransactionDT"].to_numpy()
        positions = np.arange(len(times))

        # Previous 1 hour = 3,600 seconds
        left_1h = np.searchsorted(
            times,
            times - 3600,
            side="left"
        )

        velocity_1h[group.index.to_numpy()] = (
            positions - left_1h
        )

        # Previous 24 hours = 86,400 seconds
        left_24h = np.searchsorted(
            times,
            times - 86400,
            side="left"
        )

        velocity_24h[group.index.to_numpy()] = (
            positions - left_24h
        )

    df["card_txn_count_1h"] = velocity_1h
    df["card_txn_count_24h"] = velocity_24h

    return df



def add_device_profile_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = (
        df.sort_values(
            ["DeviceInfo", "TransactionDT"]
        )
        .reset_index(drop=True)
        .copy()
    )

    # ---------------------------------------------
    # Device information available?
    # ---------------------------------------------

    df["has_device_info"] = (
        df["DeviceInfo"].notna()
    ).astype("int8")

    # ---------------------------------------------
    # Previous transactions with same device profile
    #
    # Missing DeviceInfo is deliberately excluded.
    # We don't want every NaN to become one "device".
    # ---------------------------------------------

    valid_device = df["DeviceInfo"].notna()

    df["device_profile_count"] = 0

    df.loc[valid_device, "device_profile_count"] = (
        df.loc[valid_device]
        .groupby("DeviceInfo")
        .cumcount()
    )

    df["device_profile_count"] = (
        df["device_profile_count"]
        .astype("int32")
    )

    # ---------------------------------------------
    # Previous unique cards associated with the
    # same device profile
    # ---------------------------------------------

    unique_cards = np.zeros(
        len(df),
        dtype=np.int32
    )

    valid_positions = np.where(
        valid_device.to_numpy()
    )[0]

    valid_df = df.loc[valid_device]

    for _, group in valid_df.groupby(
        "DeviceInfo",
        sort=False
    ):

        cards_seen = set()

        positions = group.index.to_numpy()

        cards = group["card1"].to_numpy()

        counts = []

        for card in cards:

            counts.append(
                len(cards_seen)
            )

            if pd.notna(card):
                cards_seen.add(card)

        unique_cards[positions] = counts

    df["device_profile_unique_cards"] = (
        unique_cards
    )

    # ---------------------------------------------
    # New device profile
    # ---------------------------------------------

    df["new_device_profile"] = (
        (
            df["has_device_info"] == 1
        )
        &
        (
            df["device_profile_count"] == 0
        )
    ).astype("int8")

    return df

def add_card_device_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create historical card-device relationship features.

    Only transactions occurring before the current transaction
    are used.
    """

    df = (
        df.sort_values(
            ["DeviceInfo", "card1", "TransactionDT"]
        )
        .reset_index(drop=True)
        .copy()
    )

    # Only valid device profiles participate
    valid = (
        df["DeviceInfo"].notna()
        & df["card1"].notna()
    )

    # --------------------------------------------------
    # 1. Previous transactions for this card-device pair
    # --------------------------------------------------

    df["card_device_transaction_count"] = 0

    df.loc[valid, "card_device_transaction_count"] = (
        df.loc[valid]
        .groupby(
            ["DeviceInfo", "card1"]
        )
        .cumcount()
    )

    df["card_device_transaction_count"] = (
        df["card_device_transaction_count"]
        .astype("int32")
    )

    # --------------------------------------------------
    # 2. Has this card-device combination appeared before?
    # --------------------------------------------------

    df["card_device_seen_before"] = (
        df["card_device_transaction_count"] > 0
    ).astype("int8")

    # --------------------------------------------------
    # 3. Number of unique cards previously associated
    #    with this DeviceInfo
    # --------------------------------------------------

    unique_cards = np.zeros(
        len(df),
        dtype=np.int32
    )

    for _, group in df[valid].groupby(
        "DeviceInfo",
        sort=False
    ):

        cards_seen = set()

        positions = group.index.to_numpy()
        cards = group["card1"].to_numpy()

        counts = []

        for card in cards:

            # Count BEFORE adding current card
            counts.append(
                len(cards_seen)
            )

            if pd.notna(card):
                cards_seen.add(card)

        unique_cards[positions] = counts

    df["device_unique_cards_historical"] = (
        unique_cards
    )

    return df