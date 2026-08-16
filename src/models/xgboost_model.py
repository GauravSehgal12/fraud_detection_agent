from xgboost import XGBClassifier


def build_xgboost_model(y):
    """
    Build an XGBoost fraud classifier.

    scale_pos_weight handles the class imbalance.
    """

    negative = (y == 0).sum()
    positive = (y == 1).sum()

    scale_pos_weight = negative / positive

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,

        objective="binary:logistic",
        eval_metric="aucpr",

        scale_pos_weight=scale_pos_weight,

        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    return model