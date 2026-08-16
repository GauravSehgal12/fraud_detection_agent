INITIAL_FEATURES = [
    # Transaction
    "TransactionAmt",
    "TransactionAmt_log",

    # Time
    "transaction_hour",
    "transaction_day",

    # Identity
    "has_identity",

    # Missingness
    "missing_card_info",
    "missing_address",
    "missing_value_count",

    # Card behavior
    "card_transaction_count",
    "card_avg_amount",
    "amount_vs_card_avg",
    "new_card",
    "card_txn_count_1h",
    "card_txn_count_24h",

    # Device profile
    "has_device_info",
    "device_profile_count",
    "device_profile_unique_cards",
    "new_device_profile",

    # Card-device relationship
    "card_device_transaction_count",
    "card_device_seen_before",
    "device_unique_cards_historical",
]