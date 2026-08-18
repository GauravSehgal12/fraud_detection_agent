# Financial Fraud Detection Agent --- Implementation Change Specification

## Goal

Upgrade the existing fraud project into a production-oriented
architecture without breaking the currently working XGBoost + SHAP +
FastAPI pipeline.

Current working flow:

``` text
API
 -> transaction enrichment
 -> TransactionFeatureBuilder
 -> 21 XGBoost features
 -> XGBoost
 -> SHAP
 -> FraudInvestigationAgent
 -> report
```

Target flow:

``` text
API
 -> input validation
 -> transaction enrichment
 -> feature builder
 -> XGBoost model risk
 -> SHAP
 -> cold-start detector
 -> deterministic rule engine
 -> deterministic decision engine
 -> LLM investigation report
 -> output guardrails
 -> structured API response
```

The implementation must preserve the existing model's 21 features and
their order.

------------------------------------------------------------------------

# 1. Current Project Facts

The current system has:

-   FastAPI
-   `AppContainer`
-   `FraudInvestigationTools`
-   `TransactionFeatureBuilder`
-   `RiskEngine`
-   `FraudInvestigationAgent`
-   XGBoost
-   SHAP
-   transaction history
-   identity data
-   card history
-   device history

The current XGBoost features are:

``` python
FEATURE_COLUMNS = [
    "TransactionAmt",
    "TransactionAmt_log",
    "transaction_hour",
    "transaction_day",
    "has_identity",
    "missing_card_info",
    "missing_address",
    "missing_value_count",
    "card_transaction_count",
    "card_avg_amount",
    "amount_vs_card_avg",
    "new_card",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "has_device_info",
    "device_profile_count",
    "device_profile_unique_cards",
    "new_device_profile",
    "card_device_transaction_count",
    "card_device_seen_before",
    "device_unique_cards_historical",
]
```

Do not rename, reorder, remove, or silently redefine these features.

------------------------------------------------------------------------

# 2. Important Existing Fix

The feature builder now correctly calculates:

``` text
missing_value_count
```

from the original transaction schema rather than the merged
transaction + identity schema.

Current data:

``` text
transaction data: approximately 394 columns
identity data: approximately 41 columns
merged data: approximately 434 columns
```

For the validated synthetic transaction based on historical transaction
`3409570`, the model input is now:

``` text
missing_value_count = 95
card_transaction_count = 157
card_avg_amount ≈ 45.337777
device_profile_count = 46
device_profile_unique_cards = 27
device_unique_cards_historical = 25
```

Do not break this behavior.

------------------------------------------------------------------------

# 3. Main Problem: Cold Start

A genuinely new transaction can contain only:

``` json
{
  "TransactionID": 999999997,
  "TransactionDT": 10699421,
  "TransactionAmt": 500.0,
  "card1": 99999,
  "DeviceInfo": "UNKNOWN-DEVICE"
}
```

The current system can generate:

``` text
card_transaction_count = 0
card_avg_amount = 0
new_card = 1
device_profile_count = 0
new_device_profile = 1
missing_value_count = 390
```

The model can then return LOW risk even though the transaction has no
historical card/device context.

This is not necessarily an XGBoost bug. It is a cold-start /
feature-semantics problem.

The solution is to separate:

``` text
MODEL RISK
```

from:

``` text
BEHAVIORAL / INVESTIGATION RISK
```

Do not arbitrarily change the XGBoost prediction.

------------------------------------------------------------------------

# 4. Create ColdStartDetector

Create:

``` text
src/services/cold_start_detector.py
```

Responsibilities:

``` text
is_new_card
is_new_device
is_new_card_device_pair
card_history_available
device_history_available
card_device_history_available
```

Historical records must always satisfy:

``` text
historical TransactionDT < current TransactionDT
```

Return:

``` json
{
  "is_new_card": true,
  "is_new_device": true,
  "is_new_card_device_pair": true,
  "card_history_available": false,
  "device_history_available": false,
  "card_device_history_available": false
}
```

The detector must not calculate fraud probability and must not make the
final decision.

------------------------------------------------------------------------

# 5. Handle Partial Transactions Correctly

A five-field API request must not automatically be interpreted as a
fully populated transaction with 389 genuine missing source fields.

Distinguish:

``` text
COMPLETE
```

from:

``` text
PARTIAL
```

For example:

``` json
{
  "input_completeness": "PARTIAL"
}
```

A complete transaction should calculate real missing values from the
source schema.

A partial transaction should expose that it is partial rather than
silently treating omitted API fields as actual source-dataset missing
values.

Do not change the trained model's feature semantics without retraining.

------------------------------------------------------------------------

# 6. Preserve Transaction Enrichment

Keep these cases separate:

### Existing transaction

``` text
3409570
```

Load the real historical transaction.

### Synthetic transaction based on historical data

If the project intentionally creates a synthetic transaction from a
historical row, explicitly record:

``` text
base_transaction_id
overridden_fields
synthetic = true
```

### Truly new transaction

If no historical base transaction exists:

``` text
synthetic = false
base_transaction_id = null
```

Use the supplied fields and historical behavioral lookups.

Never pretend that a new transaction already exists in the historical
dataset.

------------------------------------------------------------------------

# 7. Create RuleEngine

Create:

``` text
src/services/rule_engine.py
```

The rule engine must be deterministic.

It should inspect:

-   model risk
-   transaction amount
-   card history
-   device history
-   cold-start status
-   velocity
-   card/device relationships

Initial rules:

### NEW_CARD

``` text
is_new_card == true
```

Severity:

``` text
MEDIUM
```

### NEW_DEVICE

``` text
is_new_device == true
```

Severity:

``` text
MEDIUM
```

### NEW_CARD_NEW_DEVICE

``` text
is_new_card == true
AND is_new_device == true
```

Severity:

``` text
HIGH
```

### NEW_CARD_NEW_DEVICE_HIGH_AMOUNT

``` text
is_new_card == true
AND is_new_device == true
AND TransactionAmt >= configured threshold
```

The amount threshold must be configurable.

### HIGH_MODEL_RISK

``` text
risk_score >= 0.90
```

Severity:

``` text
HIGH
```

### MEDIUM_MODEL_RISK

``` text
0.70 <= risk_score < 0.90
```

Severity:

``` text
MEDIUM
```

### HIGH_CARD_VELOCITY

Use:

``` text
card_txn_count_1h
card_txn_count_24h
```

against configurable thresholds.

### SHARED_DEVICE

Use:

``` text
device_profile_unique_cards
```

against a configurable threshold.

### UNUSUAL_AMOUNT

Only calculate amount-vs-history anomaly when card history exists.

If:

``` text
card_history_available == false
```

do not claim the transaction is unusual relative to the card's history.

------------------------------------------------------------------------

# 8. RuleEngine Output

Return structured data:

``` json
{
  "rules_triggered": [
    {
      "rule_id": "NEW_CARD_NEW_DEVICE",
      "severity": "HIGH",
      "reason": "The card and device have no prior observed history."
    }
  ],
  "behavioral_risk_score": 0.85,
  "behavioral_risk_level": "HIGH"
}
```

`behavioral_risk_score` is a deterministic rule score, not an ML
probability.

Do not call it `fraud_probability`.

------------------------------------------------------------------------

# 9. Create DecisionEngine

Create:

``` text
src/services/decision_engine.py
```

Responsibilities:

``` text
model risk
+
behavioral risk
+
cold-start signals
    ->
final decision
```

The LLM must not make this decision.

Initial policy:

``` text
model >= 0.90 -> REVIEW
model >= 0.70 -> REVIEW
model LOW + behavioral HIGH -> REVIEW
model LOW + behavioral LOW -> APPROVE
```

Do not automatically introduce DECLINE rules yet.

Automatic decline should be added only after validation and explicit
business rules.

The existing model-only decision must remain available separately.

------------------------------------------------------------------------

# 10. Modify RiskEngine

Modify:

``` text
src/services/risk_engine.py
```

Keep RiskEngine responsible for:

``` text
feature generation
XGBoost prediction
model risk score
model risk level
model-only decision
SHAP
```

Add:

``` text
input_completeness
cold_start information
```

Example:

``` json
{
  "transaction_id": 999999997,

  "model_risk": {
    "score": 0.0122,
    "level": "LOW",
    "decision": "APPROVE"
  },

  "input_completeness": "PARTIAL",

  "cold_start": {
    "is_new_card": true,
    "is_new_device": true,
    "is_new_card_device_pair": true,
    "card_history_available": false,
    "device_history_available": false,
    "card_device_history_available": false
  },

  "evidence": []
}
```

Do not put business-rule logic into RiskEngine.

------------------------------------------------------------------------

# 11. Modify FraudInvestigationTools

Modify:

``` text
src/agent/tools.py
```

Keep the existing working tools:

``` text
get_transaction()
get_risk_assessment()
get_card_history()
get_device_history()
investigate()
```

Add deterministic tools for:

``` text
get_cold_start_status()
get_behavioral_risk()
get_rules_triggered()
get_final_decision()
```

The tools must calculate facts.

The LLM must not calculate:

``` text
risk score
SHAP
card counts
device counts
rule severity
final decision
```

------------------------------------------------------------------------

# 12. Modify FraudInvestigationAgent

Modify:

``` text
src/agent/fraud_agent.py
```

The report must clearly separate:

``` text
MODEL RISK
BEHAVIORAL RISK
FINAL DECISION
```

Example:

``` text
MODEL RISK:
LOW

MODEL SCORE:
0.0122

MODEL DECISION:
APPROVE

BEHAVIORAL RISK:
HIGH

FINAL DECISION:
REVIEW
```

For the new-card/new-device example, do not produce:

``` text
"No suspicious behavior is present."
```

when deterministic rules identify suspicious signals.

The LLM only summarizes deterministic evidence.

------------------------------------------------------------------------

# 13. LLM Safety Boundary

The LLM must never:

-   invent historical transactions
-   invent SHAP values
-   change the model score
-   change the model risk level
-   invent card history
-   invent device history
-   override the deterministic final decision
-   fabricate identity information
-   make unsupported fraud allegations

The LLM may:

-   summarize
-   explain
-   organize evidence
-   generate an investigation narrative
-   explain why analyst review is recommended

------------------------------------------------------------------------

# 14. Structured API Response

Update the API response while keeping the existing `report` field for
compatibility.

Target structure:

``` json
{
  "transaction_id": 999999999,

  "risk": {
    "model_score": 0.9825,
    "model_level": "HIGH",
    "model_decision": "REVIEW"
  },

  "behavioral_risk": {
    "score": 0.0,
    "level": "LOW",
    "rules_triggered": []
  },

  "final_decision": "REVIEW",

  "cold_start": {
    "is_new_card": false,
    "is_new_device": false,
    "is_new_card_device_pair": false,
    "card_history_available": true,
    "device_history_available": true
  },

  "input_completeness": "COMPLETE",

  "evidence": [],

  "report": "..."
}
```

Frontend clients should use structured fields, not parse the LLM report
to determine the decision.

------------------------------------------------------------------------

# 15. AppContainer

Modify only as necessary:

``` text
src/services/app_container.py
```

Keep separate:

``` text
raw_transactions
identity
merged_transactions
investigation_history
risk_assessments
model
features
agent
```

Do not replace the raw transaction dataframe with the merged dataframe.

The raw 394-column dataframe is required for the original feature
schema.

The merged 434-column dataframe is required for identity/device
investigation.

------------------------------------------------------------------------

# 16. FastAPI Startup

Keep:

``` text
load model
load raw transactions
load identity
merge transaction + identity
load investigation history
load risk assessments
create tools
create LLM
create agent
register agent
```

Initialize:

``` text
ColdStartDetector
RuleEngine
DecisionEngine
```

once at application startup.

Expected logs:

``` text
XGBoost model loaded
Model features: 21
Raw transactions: 590540 x 394
Identity: 144233 x 41
Merged: 590540 x 434
Cold-start detector initialized
Rule engine initialized
Decision engine initialized
Fraud Investigation Agent initialized
Financial Fraud Risk API is ready
```

------------------------------------------------------------------------

# 17. Configuration

Create or extend the existing configuration mechanism.

Do not hard-code business thresholds throughout the codebase.

Use configurable values for:

``` text
HIGH_RISK_MODEL_THRESHOLD=0.90
MEDIUM_RISK_MODEL_THRESHOLD=0.70
HIGH_AMOUNT_THRESHOLD=...
DEVICE_UNIQUE_CARD_THRESHOLD=...
CARD_1H_VELOCITY_THRESHOLD=...
CARD_24H_VELOCITY_THRESHOLD=...
```

Thresholds must be documented as business/rule configuration.

Do not claim they are statistically optimal until validation is
performed.

------------------------------------------------------------------------

# 18. Tests

Add tests for:

### Existing transaction

``` text
3409570
```

Verify:

-   transaction found
-   feature generation works
-   21 features
-   SHAP works
-   card history works
-   device history works

### Validated synthetic transaction

``` text
999999999
```

Verify:

``` text
missing_value_count = 95
card_transaction_count = 157
card_avg_amount ≈ 45.337777
device_profile_count = 46
device_profile_unique_cards = 27
device_unique_cards_historical = 25
```

### New card + new device

``` json
{
  "TransactionID": 999999997,
  "TransactionDT": 10699421,
  "TransactionAmt": 500.0,
  "card1": 99999,
  "DeviceInfo": "UNKNOWN-DEVICE"
}
```

Verify:

``` text
is_new_card = true
is_new_device = true
is_new_card_device_pair = true
```

and verify that behavioral rules can escalate the final decision even if
the model score is LOW.

### Existing card + new device

Verify:

``` text
is_new_card = false
is_new_device = true
```

### New card + known device

Verify:

``` text
is_new_card = true
is_new_device = false
```

### High velocity

Verify velocity rule triggering.

### Shared device

Verify shared-device rule triggering.

### Missing transaction

``` text
9999999999
```

Verify transaction-not-found behavior.

### Invalid input

Verify FastAPI/Pydantic validation errors.

------------------------------------------------------------------------

# 19. Model Evaluation --- Separate Phase

Do not retrain the model during the architectural implementation.

After the application changes are stable, evaluate the model using:

``` text
PR-AUC
ROC-AUC
Precision
Recall
F1
Precision@K
Recall@K
False Positive Rate
False Negative Rate
Confusion Matrix
```

For fraud detection, also measure:

``` text
Recall at 1% review rate
Recall at 5% review rate
Recall at 10% review rate
```

Use temporal validation:

``` text
earlier data -> train
later data -> validation
latest data -> test
```

Do not rely only on random train/test splitting.

------------------------------------------------------------------------

# 20. Risk Calibration --- Later Phase

Evaluate whether model scores are calibrated.

Potential methods:

``` text
Platt scaling
Isotonic regression
```

Evaluate:

``` text
Brier score
Calibration curve
```

Only call the output a probability if calibration supports that
interpretation.

Otherwise use:

``` text
model_risk_score
```

------------------------------------------------------------------------

# 21. Analyst Feedback Loop

Add:

``` http
POST /api/v1/investigations/{transaction_id}/feedback
```

Example:

``` json
{
  "label": "CONFIRMED_FRAUD",
  "analyst_comment": "Unauthorized card usage"
}
```

Allowed labels:

``` text
CONFIRMED_FRAUD
FALSE_POSITIVE
LEGITIMATE
NEEDS_MORE_INFORMATION
```

Store:

``` text
transaction_id
model_score
behavioral_score
final_decision
analyst_label
analyst_comment
timestamp
model_version
```

Do not automatically retrain from feedback.

Use feedback for future evaluation and retraining.

------------------------------------------------------------------------

# 22. Observability

Add structured logging for:

``` text
request_id
transaction_id
model_version
model_score
behavioral_score
final_decision
rules_triggered
cold_start status
latency
LLM model
LLM latency
errors
```

Do not log complete identity records or unnecessary sensitive
transaction data.

------------------------------------------------------------------------

# 23. Dashboard --- Later

After the backend is stable, add a dashboard showing:

``` text
Transaction Details
Model Risk
Behavioral Risk
Final Decision
SHAP Evidence
Card History
Device History
Triggered Rules
AI Investigation Report
Analyst Feedback
```

The dashboard must consume structured API fields rather than parse the
report text.

------------------------------------------------------------------------

# 24. RAG --- Optional Later

Only add RAG if there are useful documents to retrieve, such as:

``` text
fraud policies
investigation procedures
verification guidelines
risk policies
analyst procedures
```

RAG must not override:

``` text
XGBoost model risk
rule engine
decision engine
```

The LLM can use RAG to explain policy-based recommendations.

------------------------------------------------------------------------

# 25. Implementation Order

Implement in this order:

## Phase 1

Cold-start detector.

## Phase 2

Rule engine.

## Phase 3

Decision engine.

## Phase 4

Structured API response.

## Phase 5

Agent/report changes.

## Phase 6

Unit/integration/end-to-end tests.

## Phase 7

Temporal model evaluation.

## Phase 8

Risk calibration.

## Phase 9

Analyst feedback and monitoring.

## Phase 10

Dashboard.

## Phase 11

Optional RAG.

Do not skip directly to RAG or dashboard before the decision
architecture is stable.

------------------------------------------------------------------------

# 26. Non-Goals

Do NOT:

-   immediately retrain XGBoost
-   arbitrarily change risk thresholds
-   let the LLM make the final fraud decision
-   replace SHAP with an LLM explanation
-   invent historical behavior
-   use future transactions for historical features
-   treat every omitted API field as a genuine dataset missing value
-   automatically decline transactions without validated rules
-   add RAG only for the sake of calling the project agentic

------------------------------------------------------------------------

# 27. Final Acceptance Criteria

The implementation is complete when:

1.  Existing transaction `3409570` works.
2.  Synthetic transaction `999999999` still produces the validated
    21-feature values.
3.  New transaction `999999997` is recognized as cold-start.
4.  XGBoost still receives exactly 21 features in the trained order.
5.  Model risk is separate from behavioral risk.
6.  Final decision is deterministic.
7.  LLM cannot override deterministic risk/decision values.
8.  API returns structured risk, behavior, rules, cold-start status,
    final decision, and report.
9.  Existing card/device history tools still work.
10. Unit and integration tests pass.
11. Temporal model evaluation is available as a separate workflow.
12. Feedback can be recorded without automatically retraining the model.

------------------------------------------------------------------------

# 28. Target Architecture

``` text
                         ┌──────────────────┐
                         │    FastAPI       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Input Validation │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Transaction Enrichment   │
                    │ Transaction + Identity   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ Feature Builder  │      │ Cold Start       │
          │ 21 model feats   │      │ Detector         │
          └────────┬─────────┘      └────────┬─────────┘
                   │                         │
                   ▼                         │
          ┌──────────────────┐               │
          │ XGBoost          │               │
          └────────┬─────────┘               │
                   │                         │
                   ▼                         │
          ┌──────────────────┐               │
          │ SHAP             │               │
          └────────┬─────────┘               │
                   │                         │
                   └────────────┬────────────┘
                                ▼
                       ┌──────────────────┐
                       │ Rule Engine      │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Decision Engine  │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Fraud Agent      │
                       │ LLM              │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Output Guardrail │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Structured API   │
                       └──────────────────┘
```

## Core Responsibility Boundary

``` text
XGBoost
= predicts model risk

SHAP
= explains model prediction

ColdStartDetector
= determines history availability

RuleEngine
= detects deterministic behavioral signals

DecisionEngine
= makes final deterministic action

LLM
= explains the investigation
```

Never reverse these responsibilities.
