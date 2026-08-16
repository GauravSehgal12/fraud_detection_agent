import numpy as np
import pandas as pd
import shap


class RiskExplanationEngine:

    def __init__(self, model):

        self.model = model

        self.explainer = shap.TreeExplainer(
            model
        )

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
                evidence["shap_value"]
                .abs()
            )

            evidence = (
                evidence
                .sort_values(
                    "abs_shap",
                    ascending=False
                )
                .head(top_k)
            )

            explanations.append(
                evidence.to_dict(
                    orient="records"
                )
            )

        return explanations