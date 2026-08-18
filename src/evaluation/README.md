# Fraud System Evaluation

This package evaluates the existing XGBoost model, behavioral RuleEngine, and final DecisionEngine on a chronological holdout.

## Run

From the repository root:

```bash
python -m src.evaluation.evaluate
```

The runner expects `data/feature_store.csv` to already contain the leakage-safe historical features and `isFraud` labels.

## Outputs

The command writes:

- `eval_results/system_evaluation.json` — complete model/rule/decision evaluation
- `eval_results/threshold_analysis.csv` — precision/recall/F1 and review rate across model thresholds
- `eval_results/cold_start_segments.json` — model performance for existing/new card and device combinations

## Evaluation design

The feature store is sorted by `TransactionDT`, then split into:

- 70% earliest transactions: train period
- 15% next transactions: validation period
- 15% latest transactions: test period

The model is evaluated with ROC-AUC, PR-AUC, precision, recall, F1, false-positive/false-negative rates, and recall/precision at 1%, 5%, and 10% review capacity.

The RuleEngine is evaluated separately from the model. The final DecisionEngine is evaluated as a binary `APPROVE`/`REVIEW` system.

Cold-start segments are evaluated separately:

- existing card + existing device
- existing card + new device
- new card + existing device
- new card + new device

## Tests

Run:

```bash
pytest tests/test_evaluation.py
```

Do not interpret the model score as a calibrated probability unless the calibration results support that interpretation.
