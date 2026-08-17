from typing import Any
import pandas as pd

from src.services.risk_engine import RiskEngine
from src.services.transaction_feature_builder import TransactionFeatureBuilder
from src.services.cold_start_detector import ColdStartDetector
from src.services.rule_engine import RuleEngine
from src.services.decision_engine import DecisionEngine


class FraudInvestigationTools:

    def __init__(
        self,
        transactions: pd.DataFrame,
        identity: pd.DataFrame | None = None,
        risk_assessments: dict | None = None,
        model=None,
        cold_start_detector: ColdStartDetector | None = None,
        rule_engine: RuleEngine | None = None,
        decision_engine: DecisionEngine | None = None,
    ):
        """
        Fraud investigation tools with deterministic rules and decision engines.
        """
        self.transactions = transactions
        self.identity = identity
        self.risk_assessments = risk_assessments if risk_assessments is not None else {}

        if not isinstance(self.transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame.")

        required_columns = ["TransactionID", "TransactionDT", "TransactionAmt", "card1"]
        missing_columns = [col for col in required_columns if col not in self.transactions.columns]
        if missing_columns:
            raise ValueError(f"Transactions dataframe is missing required columns: {missing_columns}")

        # Feature Builder history setup
        history = self.transactions.copy()

        if (
            self.identity is not None
            and "TransactionID" in self.identity.columns
            and "DeviceInfo" in self.identity.columns
        ):
            device_data = self.identity[["TransactionID", "DeviceInfo"]].copy()
            device_data = device_data.drop_duplicates(subset=["TransactionID"], keep="first")

            if "DeviceInfo" in history.columns:
                identity_device_map = device_data.set_index("TransactionID")["DeviceInfo"]
                history["DeviceInfo"] = history["DeviceInfo"].fillna(
                    history["TransactionID"].map(identity_device_map)
                )
            else:
                history = history.merge(device_data, on="TransactionID", how="left", sort=False)
        else:
            if "DeviceInfo" not in history.columns:
                history["DeviceInfo"] = None

        self.feature_builder = TransactionFeatureBuilder(
            historical_df=history,
            identity=self.identity,
            raw_columns=self.transactions.columns.tolist(),
        )

        # Risk Engine
        self.risk_engine = None
        if model is not None:
            self.risk_engine = RiskEngine(
                model=model,
                feature_builder=self.feature_builder,
            )

        # Architectural Engines
        self.cold_start_detector = (
            cold_start_detector
            if cold_start_detector is not None
            else ColdStartDetector(self.transactions, self.identity)
        )
        self.rule_engine = rule_engine if rule_engine is not None else RuleEngine()
        self.decision_engine = decision_engine if decision_engine is not None else DecisionEngine()

    def _find_base_transaction(self, transaction: dict[str, Any]) -> dict[str, Any] | None:
        card1 = transaction.get("card1")
        transaction_dt = transaction.get("TransactionDT")
        if card1 is None or transaction_dt is None:
            return None

        try:
            card1_numeric = float(card1)
            transaction_dt_numeric = float(transaction_dt)
        except (TypeError, ValueError):
            return None

        matches = self.transactions[
            (pd.to_numeric(self.transactions["card1"], errors="coerce") == card1_numeric)
            & (pd.to_numeric(self.transactions["TransactionDT"], errors="coerce") == transaction_dt_numeric)
        ]

        if matches.empty:
            return None

        return matches.iloc[0].to_dict()

    def _prepare_new_transaction(self, transaction: dict[str, Any]) -> dict[str, Any]:
        base_transaction = self._find_base_transaction(transaction)

        if base_transaction is not None:
            enriched = base_transaction.copy()
            enriched.update(transaction)
            if "TransactionID" in transaction:
                enriched["TransactionID"] = transaction["TransactionID"]
            enriched["synthetic"] = True
            return enriched

        enriched = transaction.copy()
        enriched["synthetic"] = False
        return enriched

    def get_transaction(self, transaction: int | dict[str, Any]) -> dict[str, Any]:
        if isinstance(transaction, int):
            matches = self.transactions[self.transactions["TransactionID"] == transaction]
            if matches.empty:
                return {"error": "Transaction not found."}

            row = matches.iloc[0]
            result = {}
            for col in ["TransactionID", "TransactionDT", "TransactionAmt", "card1"]:
                if col in row.index:
                    val = row[col]
                    result[col] = None if pd.isna(val) else (val.item() if hasattr(val, "item") else val)

            result["DeviceInfo"] = None
            if (
                self.identity is not None
                and "TransactionID" in self.identity.columns
                and "DeviceInfo" in self.identity.columns
            ):
                id_match = self.identity[self.identity["TransactionID"] == transaction]
                if not id_match.empty:
                    val = id_match.iloc[0]["DeviceInfo"]
                    result["DeviceInfo"] = None if pd.isna(val) else val
            elif "DeviceInfo" in row.index:
                val = row["DeviceInfo"]
                result["DeviceInfo"] = None if pd.isna(val) else val

            return result

        if isinstance(transaction, dict):
            transaction_id = transaction.get("TransactionID")
            if transaction_id is None:
                return {"error": "TransactionID is required."}

            result = {}
            for col in ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "DeviceInfo"]:
                if col in transaction:
                    val = transaction[col]
                    result[col] = None if (val is None or pd.isna(val)) else (val.item() if hasattr(val, "item") else val)
            return result

        return {"error": "Invalid transaction input."}

    def get_risk_assessment(self, transaction: int | dict[str, Any]) -> dict[str, Any]:
        if self.risk_engine is None:
            return {"error": "Risk engine not initialized."}

        if isinstance(transaction, int):
            risk_input = transaction
        elif isinstance(transaction, dict):
            risk_input = self._prepare_new_transaction(transaction)
        else:
            return {"error": "Invalid transaction type."}

        return self.risk_engine.calculate_risk(risk_input)

    def get_card_history(self, transaction: int | dict[str, Any]) -> dict[str, Any]:
        tx_data = self.get_transaction(transaction)
        if "error" in tx_data:
            return tx_data

        card1 = tx_data.get("card1")
        transaction_dt = tx_data.get("TransactionDT")

        if card1 is None or transaction_dt is None:
            return {"error": "Card information unavailable."}

        history = self.transactions[
            (self.transactions["card1"] == card1)
            & (self.transactions["TransactionDT"] < transaction_dt)
        ].copy()

        if history.empty:
            return {
                "card1": int(card1),
                "transaction_count": 0,
                "average_amount": 0.0,
                "max_amount": 0.0,
                "min_amount": 0.0,
            }

        amounts = pd.to_numeric(history["TransactionAmt"], errors="coerce").dropna()
        return {
            "card1": int(card1),
            "transaction_count": int(len(history)),
            "average_amount": float(amounts.mean()) if not amounts.empty else 0.0,
            "max_amount": float(amounts.max()) if not amounts.empty else 0.0,
            "min_amount": float(amounts.min()) if not amounts.empty else 0.0,
        }

    def get_device_history(self, transaction: int | dict[str, Any]) -> dict[str, Any]:
        tx_data = self.get_transaction(transaction)
        if "error" in tx_data:
            return tx_data

        device_info = tx_data.get("DeviceInfo")
        transaction_dt = tx_data.get("TransactionDT")

        if device_info is None or transaction_dt is None:
            return {"device_info": None, "transaction_count": 0, "unique_cards": 0}

        if (
            self.identity is None
            or "TransactionID" not in self.identity.columns
            or "DeviceInfo" not in self.identity.columns
        ):
            return {"device_info": device_info, "transaction_count": 0, "unique_cards": 0}

        device_data = self.identity[["TransactionID", "DeviceInfo"]].drop_duplicates(subset=["TransactionID"], keep="first")
        txn_df = self.transactions[["TransactionID", "TransactionDT", "card1"]].drop_duplicates(subset=["TransactionID"], keep="first")
        device_history = device_data.merge(txn_df, on="TransactionID", how="inner")

        history = device_history[
            (device_history["DeviceInfo"] == device_info)
            & (device_history["TransactionDT"] < transaction_dt)
        ].copy()

        if history.empty:
            return {"device_info": device_info, "transaction_count": 0, "unique_cards": 0}

        unique_cards = history["card1"].dropna().nunique()
        return {
            "device_info": device_info,
            "transaction_count": int(len(history)),
            "unique_cards": int(unique_cards),
        }

    # Deterministic Architecture Tools
    def get_cold_start_status(self, transaction: int | dict[str, Any]) -> dict[str, Any]:
        tx_data = self.get_transaction(transaction)
        if "error" in tx_data:
            return {
                "is_new_card": True,
                "is_new_device": True,
                "is_new_card_device_pair": True,
                "card_history_available": False,
                "device_history_available": False,
                "card_device_history_available": False,
            }
        return self.cold_start_detector.detect(tx_data)

    def get_behavioral_risk(self, transaction: int | dict[str, Any]) -> dict[str, Any]:
        tx_data = self.get_transaction(transaction)
        cold_start = self.get_cold_start_status(transaction)
        risk_res = self.get_risk_assessment(transaction)

        model_risk_score = risk_res.get("risk_score") if "error" not in risk_res else None
        features = risk_res.get("features") if "error" not in risk_res else {}

        return self.rule_engine.evaluate(
            transaction=tx_data,
            model_risk_score=model_risk_score,
            cold_start_status=cold_start,
            features=features,
        )

    def get_rules_triggered(self, transaction: int | dict[str, Any]) -> list[dict[str, Any]]:
        behavioral_risk = self.get_behavioral_risk(transaction)
        return behavioral_risk.get("rules_triggered", [])

    def get_final_decision(self, transaction: int | dict[str, Any]) -> str:
        risk_res = self.get_risk_assessment(transaction)
        model_score = risk_res.get("risk_score", 0.0) if "error" not in risk_res else 0.0
        model_level = risk_res.get("risk_level", "LOW") if "error" not in risk_res else "LOW"

        behavioral_res = self.get_behavioral_risk(transaction)
        behavioral_level = behavioral_res.get("behavioral_risk_level", "LOW")
        rules_triggered = behavioral_res.get("rules_triggered", [])
        cold_start = self.get_cold_start_status(transaction)

        return self.decision_engine.decide(
            model_risk_score=model_score,
            model_risk_level=model_level,
            behavioral_risk_level=behavioral_level,
            cold_start_status=cold_start,
            rules_triggered=rules_triggered,
        )

    def collect_investigation_data(self, transaction: int | dict[str, Any]) -> dict[str, Any]:
        tx_data = self.get_transaction(transaction)
        if "error" in tx_data:
            return tx_data

        risk_res = self.get_risk_assessment(transaction)
        if "error" in risk_res:
            return risk_res

        cold_start = self.get_cold_start_status(transaction)
        behavioral_risk = self.get_behavioral_risk(transaction)
        card_history = self.get_card_history(transaction)
        device_history = self.get_device_history(transaction)
        final_decision = self.get_final_decision(transaction)

        model_risk_info = {
            "model_score": risk_res.get("risk_score", 0.0),
            "model_level": risk_res.get("risk_level", "LOW"),
            "model_decision": risk_res.get("decision", "APPROVE"),
        }

        behavioral_risk_info = {
            "score": behavioral_risk.get("behavioral_risk_score", 0.0),
            "level": behavioral_risk.get("behavioral_risk_level", "LOW"),
            "rules_triggered": behavioral_risk.get("rules_triggered", []),
        }

        cold_start_info = {
            "is_new_card": cold_start.get("is_new_card", True),
            "is_new_device": cold_start.get("is_new_device", True),
            "is_new_card_device_pair": cold_start.get("is_new_card_device_pair", True),
            "card_history_available": cold_start.get("card_history_available", False),
            "device_history_available": cold_start.get("device_history_available", False),
        }

        return {
            "transaction_id": tx_data.get("TransactionID"),
            "transaction": tx_data,
            "risk": model_risk_info,
            "behavioral_risk": behavioral_risk_info,
            "final_decision": final_decision,
            "cold_start": cold_start_info,
            "input_completeness": risk_res.get("input_completeness", "COMPLETE"),
            "evidence": risk_res.get("evidence", []),
            "card_history": card_history,
            "device_history": device_history,
        }