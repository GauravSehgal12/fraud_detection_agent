import pytest
import pandas as pd
from src.services.cold_start_detector import ColdStartDetector


@pytest.fixture
def mock_data():
    transactions = pd.DataFrame(
        [
            {"TransactionID": 101, "TransactionDT": 1000, "card1": 11111, "TransactionAmt": 50.0},
            {"TransactionID": 102, "TransactionDT": 2000, "card1": 11111, "TransactionAmt": 100.0},
            {"TransactionID": 103, "TransactionDT": 1500, "card1": 22222, "TransactionAmt": 200.0},
        ]
    )

    identity = pd.DataFrame(
        [
            {"TransactionID": 101, "DeviceInfo": "iOS-Device"},
            {"TransactionID": 102, "DeviceInfo": "iOS-Device"},
            {"TransactionID": 103, "DeviceInfo": "Android-Device"},
        ]
    )

    return transactions, identity


def test_cold_start_existing_card_and_device(mock_data):
    txns, identity = mock_data
    detector = ColdStartDetector(txns, identity)

    # Transaction at DT=3000 for card1=11111 and DeviceInfo="iOS-Device"
    result = detector.detect(
        {
            "card1": 11111,
            "DeviceInfo": "iOS-Device",
            "TransactionDT": 3000,
        }
    )

    assert result["is_new_card"] is False
    assert result["is_new_device"] is False
    assert result["is_new_card_device_pair"] is False
    assert result["card_history_available"] is True
    assert result["device_history_available"] is True
    assert result["card_device_history_available"] is True


def test_cold_start_new_card_new_device(mock_data):
    txns, identity = mock_data
    detector = ColdStartDetector(txns, identity)

    # Transaction for card1=99999 and DeviceInfo="UNKNOWN-DEVICE"
    result = detector.detect(
        {
            "TransactionID": 999999997,
            "TransactionDT": 10699421,
            "TransactionAmt": 500.0,
            "card1": 99999,
            "DeviceInfo": "UNKNOWN-DEVICE",
        }
    )

    assert result["is_new_card"] is True
    assert result["is_new_device"] is True
    assert result["is_new_card_device_pair"] is True
    assert result["card_history_available"] is False
    assert result["device_history_available"] is False
    assert result["card_device_history_available"] is False


def test_cold_start_existing_card_new_device(mock_data):
    txns, identity = mock_data
    detector = ColdStartDetector(txns, identity)

    result = detector.detect(
        {
            "card1": 11111,
            "DeviceInfo": "New-Brand-Phone",
            "TransactionDT": 3000,
        }
    )

    assert result["is_new_card"] is False
    assert result["is_new_device"] is True
    assert result["is_new_card_device_pair"] is True


def test_cold_start_new_card_known_device(mock_data):
    txns, identity = mock_data
    detector = ColdStartDetector(txns, identity)

    result = detector.detect(
        {
            "card1": 88888,
            "DeviceInfo": "iOS-Device",
            "TransactionDT": 3000,
        }
    )

    assert result["is_new_card"] is True
    assert result["is_new_device"] is False
    assert result["is_new_card_device_pair"] is True
