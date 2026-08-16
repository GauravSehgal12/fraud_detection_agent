import numpy as np
import pandas as pd
import shap


class RiskExplanationEngine:

    def __init__(self, model):

        self.model = model

        self.explainer = shap.TreeExplainer(
            model
        )

    def _describe_feature(
        self,
        feature,
        value,
        shap_value
    ):

        direction = (
            "increases"
            if shap_value > 0
            else "decreases"
        )

        if feature == "TransactionAmt":

            explanation = (
                f"Transaction amount is "
                f"{value:.2f}; this feature "
                f"{direction} model risk."
            )

        elif feature == "card_avg_amount":

            explanation = (
                f"Historical average amount for "
                f"the card is {value:.2f}; this "
                f"feature {direction} model risk."
            )

        elif feature == "amount_vs_card_avg":

            explanation = (
                f"Current transaction amount is "
                f"{value:.2f}x the historical "
                f"card average."
            )

        elif feature == "card_transaction_count":

            explanation = (
                f"The card has {int(value)} "
                f"previous transactions in the "
                f"observed history."
            )

        elif feature == "card_txn_count_1h":

            explanation = (
                f"The card has {int(value)} "
                f"previous transactions within "
                f"the preceding hour."
            )

        elif feature == "card_txn_count_24h":

            explanation = (
                f"The card has {int(value)} "
                f"previous transactions within "
                f"the preceding 24 hours."
            )

        elif feature == "device_unique_cards_historical":

            explanation = (
                f"The device profile was previously "
                f"associated with approximately "
                f"{int(value)} different cards."
            )

        elif feature == "device_profile_count":

            explanation = (
                f"The device profile has appeared "
                f"in {int(value)} previous transactions."
            )

        elif feature == "device_profile_unique_cards":

            explanation = (
                f"The device profile has previously "
                f"been associated with "
                f"{int(value)} different cards."
            )

        elif feature == "card_device_transaction_count":

            explanation = (
                f"This card-device combination has "
                f"appeared {int(value)} times previously."
            )

        elif feature == "card_device_seen_before":

            explanation = (
                "The card-device relationship "
                "has been observed previously."
                if value == 1
                else
                "The card-device relationship "
                "has not been observed previously."
            )

        elif feature == "has_identity":

            explanation = (
                "Identity information is available."
                if value == 1
                else
                "Identity information is unavailable."
            )

        elif feature == "has_device_info":

            explanation = (
                "Device profile information is available."
                if value == 1
                else
                "Device profile information is unavailable."
            )

        elif feature == "missing_address":

            explanation = (
                "Address information is missing."
                if value == 1
                else
                "Address information is present."
            )

        elif feature == "missing_card_info":

            explanation = (
                "Some card information is missing."
                if value == 1
                else
                "Card information is present."
            )

        elif feature == "missing_value_count":

            explanation = (
                f"The transaction contains "
                f"{int(value)} missing values; "
                f"this missing-data pattern "
                f"{direction} model risk."
            )

        elif feature == "transaction_hour":

            explanation = (
                f"Transaction occurred around "
                f"hour {int(value)}."
            )

        elif feature == "transaction_day":

            explanation = (
                f"Transaction occurred on dataset "
                f"day {int(value)}."
            )

        elif feature == "new_card":

            explanation = (
                "This is the card's first observed "
                "transaction."
                if value == 1
                else
                "This card has previous transaction history."
            )

        elif feature == "new_device_profile":

            explanation = (
                "This is the first observed transaction "
                "for this device profile."
                if value == 1
                else
                "This device profile has previous history."
            )

        else:

            explanation = (
                f"{feature} has value {value}; "
                f"this feature {direction} model risk."
            )

        return explanation

    def explain(
        self,
        X: pd.DataFrame,
        top_k: int = 5
    ):

        shap_values = self.explainer.shap_values(X)

        explanations = []

        for row_idx in range(len(X)):

            row = X.iloc[row_idx]

            row_shap = shap_values[row_idx]

            evidence = pd.DataFrame({
                "feature": X.columns,
                "value": row.values,
                "shap_value": row_shap
            })

            evidence["abs_shap"] = (
                evidence["shap_value"].abs()
            )

            evidence = (
                evidence
                .sort_values(
                    "abs_shap",
                    ascending=False
                )
                .head(top_k)
            )

            records = []

            for _, item in evidence.iterrows():

                records.append({
                    "feature": item["feature"],
                    "value": item["value"],
                    "shap_value": float(
                        item["shap_value"]
                    ),
                    "impact": (
                        "increases_risk"
                        if item["shap_value"] > 0
                        else "decreases_risk"
                    ),
                    "explanation": (
                        self._describe_feature(
                            item["feature"],
                            item["value"],
                            item["shap_value"]
                        )
                    )
                })

            explanations.append(records)

        return explanations