# Financial Fraud Detection & Investigation Agent

A production-oriented financial fraud risk detection and investigation service built with **XGBoost, FastAPI, rule-based behavioral analysis, SHAP, and a Groq-powered investigation agent**.

The system combines a trained fraud-risk model with transaction-history features, behavioral/rule signals, cold-start analysis, and LLM-assisted investigation tooling. It is packaged as a Docker image, published to GitHub Container Registry (GHCR), tested with GitHub Actions, and deployed to an AWS EC2 instance.

---

## Architecture

```text
                         ┌─────────────────────────┐
                         │       Git Push           │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     GitHub Actions       │
                         │                          │
                         │  • Python tests          │
                         │  • Artifact validation   │
                         │  • Docker smoke test     │
                         │  • Docker image build    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │          GHCR            │
                         │   Docker image :v1       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       AWS EC2            │
                         │                          │
                         │  Docker container        │
                         │       :8000              │
                         └────────────┬────────────┘
                                      │
                     ┌────────────────┴───────────────┐
                     │                                │
                     ▼                                ▼
              FastAPI REST API                 Dashboard / static UI
                     │
                     ▼
          Fraud Detection & Investigation
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
       XGBoost     Rules      LLM Agent
       model      /behavior   (Groq)
```

> **Deployment note:** the current automated CD workflow uses SSH from GitHub Actions to EC2. AWS Systems Manager (SSM) has also been configured and the EC2 instance is registered as an SSM managed node; migration of CD from SSH to SSM is the next hardening step.

---

## Key Features

- **Fraud probability scoring** using an XGBoost model.
- **Behavioral/rule-based risk analysis** alongside the ML model.
- **Cold-start detection** for situations where historical card/device information is unavailable.
- **Transaction velocity features** over 1-hour and 24-hour windows.
- **Card-level historical behavior** including average amount, standard deviation, amount ratios, and transaction rates.
- **Device/card relationship features** and historical device-card behavior.
- **SHAP-based model explainability** in the risk engine.
- **Investigation tools** for combining model, transaction, identity, rule, and risk-assessment information.
- **Groq-powered LLM investigation agent** for investigation-oriented reasoning.
- **FastAPI REST API** with Swagger/OpenAPI documentation.
- **Static dashboard** served by FastAPI.
- **Dockerized production runtime** using Python 3.13 slim.
- **CI/CD** with GitHub Actions, GHCR, and AWS EC2.
- **Production data mounted at runtime**, keeping large datasets out of the Docker image and Git repository.

---

## Machine Learning Model

The deployed production model is the **V1 XGBoost fraud model**.

The model uses engineered transaction, card, device, velocity, and amount-behavior features. The current production artifact reports **21 model features** when loaded by the application.

Representative feature groups include:

- Transaction amount and log-transformed amount
- Transaction hour/day
- Identity and missing-value indicators
- Card transaction count
- Card average amount
- Amount relative to card average
- New-card indicators
- 1-hour and 24-hour card velocity
- Device profile counts
- Unique cards associated with a device
- New-device profile indicator
- Card-device historical activity
- Amount z-score and amount ratios
- Card transaction rates
- Device/card share ratios
- Amount × velocity interaction features

The model is used as the primary risk signal, while deterministic behavioral/rule analysis provides additional investigation context.

### Model validation

The V1 production model was selected for deployment after validation and integration testing. A later experimental fusion model was evaluated, but **V1 remains the deployed production model**.

---

## Risk & Investigation Pipeline

A transaction flows through several layers:

```text
Transaction
    │
    ▼
Feature / historical context
    │
    ├── Card behavior
    ├── Device behavior
    ├── Velocity
    ├── Amount behavior
    └── Cold-start status
    │
    ▼
XGBoost risk score
    │
    ▼
Rule / behavioral engine
    │
    ▼
Decision engine
    │
    ▼
Investigation tools
    │
    ▼
Groq LLM investigation agent
    │
    ▼
Fraud risk / investigation response
```

The behavioral scoring work was intentionally conservative. Generic amount-only evidence and unstable cold-start/device signals were not treated as standalone evidence in the final stable behavioral scorer.

---

## API

The application is a FastAPI service titled:

**Financial Fraud Risk & Investigation API**

API routes are mounted under:

```text
/api/v1
```

The root endpoint is:

```http
GET /
```

Example response:

```json
{
  "service": "Financial Fraud Risk and Investigation Agent",
  "version": "1.0.0",
  "status": "running"
}
```

Swagger/OpenAPI documentation is available at:

```text
http://localhost:8000/docs
```

The application also serves the dashboard/static UI under:

```text
/dashboard
```

---

## Project Structure

The repository is organized around the FastAPI application, service layer, ML model artifacts, tests, Docker configuration, and CI/CD workflows.

```text
fraud_detection_agent/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
│
├── src/
│   ├── agent/
│   │   ├── fraud_agent.py
│   │   ├── llm.py
│   │   └── tools.py
│   │
│   ├── api/
│   │   └── route.py
│   │
│   ├── services/
│   │   ├── app_container.py
│   │   ├── risk_engine.py
│   │   ├── rule_engine.py
│   │   └── ...
│   │
│   └── ...
│
├── models/
│   └── production model artifacts
│
├── static/
│   └── dashboard/static frontend files
│
├── tests/
│   ├── test_cold_start.py
│   ├── test_decision_engine.py
│   ├── test_rule_engine.py
│   └── ...
│
└── .github/
    └── workflows/
        ├── ...
        └── deploy.yml
```

---

## Data

Large production datasets are **not copied into the Docker image**. The Dockerfile explicitly expects runtime data under `/app/data` and the EC2 deployment mounts the external `/app-data` directory into the container. fileciteturn175file0L2-L5

Current EC2 runtime layout:

```text
/app-data/
├── .env
├── investigation_history.csv
├── risk_assessments.json
└── raw/
    ├── train_transaction.csv
    └── train_identity.csv
```

Approximate production data sizes used during deployment:

| File | Approx. size |
|---|---:|
| `train_transaction.csv` | 652 MB |
| `train_identity.csv` | 23 MB |
| `investigation_history.csv` | 16 MB |
| `risk_assessments.json` | 1.5 KB |

The large transaction and identity datasets should **not be committed to GitHub**.

---

## Environment Variables

The application requires the Groq API key for the LLM investigation component.

Create a runtime `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or API keys to GitHub.

For local development, place the `.env` file where your local startup configuration expects it. For production, the EC2 deployment uses:

```text
/app-data/.env
```

and passes it to Docker with `--env-file`.

---

## Requirements

The production requirements currently include:

- Python 3.13
- pandas 2.3.2
- PyArrow
- NumPy 2.1.3
- scikit-learn 1.7.1
- XGBoost 3.0.4
- imbalanced-learn 0.14.0
- joblib 1.5.1
- Groq SDK
- pydantic-settings
- FastAPI
- SHAP 0.48.0
- Uvicorn
- python-dotenv 1.1.1

See `requirements.txt` for the exact dependency specification.

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/GauravSehgal12/fraud_detection_agent.git
cd fraud_detection_agent
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env` with:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Provide the required data

The application expects the transaction, identity, investigation-history, and risk-assessment data described in the **Data** section.

### 6. Start the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

---

## Running Tests

The project uses `pytest` for automated tests.

Run the complete test suite with:

```bash
pytest -q
```

Core tests include:

```bash
pytest -q \
  tests/test_cold_start.py \
  tests/test_decision_engine.py \
  tests/test_rule_engine.py
```

The CI pipeline also performs a Docker smoke test. CI supports a lightweight startup mode through:

```text
CI_SMOKE_TEST=true
```

This allows the image, Python imports, FastAPI wiring, static files, and lightweight application startup to be validated without loading the production dataset or initializing the LLM. fileciteturn172file0L2-L5

---

## Docker

The production image uses:

```dockerfile
FROM python:3.13-slim
```

and installs the system runtime dependency `libgomp1` required by the scientific/ML stack. The image runs Uvicorn on port `8000` as an unprivileged `appuser`. fileciteturn175file0L2-L5

### Build

```bash
docker build -t fraud-detection-agent:v1 .
```

### Run locally

```bash
docker run --rm \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  fraud-detection-agent:v1
```

On Windows PowerShell:

```powershell
docker run --rm -p 8000:8000 `
  --env-file .env `
  -v "${PWD}\data:/app/data" `
  fraud-detection-agent:v1
```

### Production container

The deployed image is published to GHCR as:

```text
ghcr.io/gauravsehgal12/fraud_detection_agent:v1
```

The EC2 deployment mounts:

```text
/app-data:/app/data
```

so production data remains outside the image.

---

## CI/CD

The project uses GitHub Actions for continuous integration and deployment.

### CI

The CI pipeline validates the application before deployment. The current checks include:

- Python tests
- V1 production artifact validation
- Docker smoke test
- Docker image build/publish flow

A recent successful CI run included all three repository checks:

```text
CI / Python tests                         ✅
CI / Validate V1 production artifacts    ✅
Docker CI / docker-smoke-test             ✅
```

### Container Registry

Validated Docker images are published to **GitHub Container Registry (GHCR)**.

Production image:

```text
ghcr.io/gauravsehgal12/fraud_detection_agent:v1
```

### CD to EC2

The deployment workflow runs after the Docker CI workflow succeeds. It currently:

1. Receives EC2 deployment credentials from GitHub Secrets.
2. Connects to EC2.
3. Pulls the validated GHCR image.
4. Verifies the required production data files exist.
5. Stops/removes the existing `fraud-detection-agent` container.
6. Starts the new container with the production `.env` and data volume.
7. Polls the API for successful startup.
8. Prints container logs if the health check fails.

The deployment workflow is configured to verify the application at:

```text
http://127.0.0.1:8000/
```

and waits up to 60 attempts with a 5-second interval. fileciteturn174file0L2-L6

### Required GitHub deployment secrets

Current SSH-based CD requires:

```text
EC2_HOST
EC2_USER
EC2_SSH_KEY
```

These must be configured as GitHub Actions repository secrets. Do not commit the private key.

---

## AWS Deployment

Production currently runs on **Amazon Linux 2023** on AWS EC2.

The EC2 instance has:

- Docker enabled as a system service
- Production data stored under `/app-data`
- The application container exposed on port `8000`
- AWS Systems Manager Agent installed and running
- The instance registered as an SSM managed node

### Current network model

During initial deployment, SSH access was temporarily opened for GitHub Actions. The long-term target is to migrate automated deployment to **AWS Systems Manager** and remove public SSH access.

Current application access is restricted to the configured client IP on port `8000` during the deployment/testing setup.

---

## AWS Systems Manager Hardening

SSM Agent is configured and the EC2 instance is online as a managed node.

The intended secure deployment architecture is:

```text
GitHub Actions
      │
      ▼
GitHub OIDC
      │
      ▼
AWS IAM Role
      │
      ▼
AWS Systems Manager
      │
      ▼
EC2
      │
      ▼
Docker
```

The migration from SSH-based GitHub Actions deployment to SSM-based deployment is the next security hardening step.

---

## Security Considerations

- Do not commit `.env` files.
- Do not commit Groq API keys.
- Do not commit EC2 private keys (`.pem`).
- Do not commit the large raw transaction/identity datasets.
- Keep production datasets outside the Docker image.
- Use GitHub Secrets for deployment credentials until the SSM/OIDC migration is complete.
- Restrict inbound EC2 ports to the smallest required source range.
- Remove public SSH access after the SSM deployment path is verified.
- Run the container as an unprivileged user; the production Dockerfile uses `appuser`. fileciteturn175file0L2-L5

---

## Deployment Workflow

The current release workflow is:

```text
Developer
   │
   │ git push main
   ▼
GitHub Actions
   │
   ├── pytest
   ├── production artifact validation
   ├── Docker smoke test
   └── Docker build/publish
   │
   ▼
GHCR
   │
   │ :v1
   ▼
AWS EC2
   │
   ├── docker pull
   ├── verify /app-data
   ├── replace container
   └── startup health check
   │
   ▼
FastAPI
   │
   ├── /api/v1/*
   ├── /
   └── /dashboard
```

---

## Known Production Considerations

### Large historical dataset

The historical transaction dataset is large enough to create significant memory pressure if loaded inefficiently. The application therefore uses a compact historical-data loading path and selected columns where applicable rather than blindly loading the entire raw schema into memory.

### CI vs production startup

CI intentionally does not load the full production dataset. The `CI_SMOKE_TEST=true` mode exists specifically to validate the application/container without requiring production data. fileciteturn172file0L2-L5

### Model version

The production deployment currently uses **V1**. Experimental model/fusion iterations are not part of the deployed production path.

---

## Roadmap

- [x] XGBoost fraud-risk model
- [x] Feature engineering and historical behavior features
- [x] Rule/behavioral risk engine
- [x] Cold-start detection
- [x] Decision engine
- [x] SHAP-based explainability support
- [x] LLM investigation agent
- [x] FastAPI API
- [x] Dashboard/static UI
- [x] Automated tests
- [x] Dockerized production application
- [x] GitHub Actions CI
- [x] GHCR image publishing
- [x] AWS EC2 deployment
- [x] AWS SSM managed-node setup
- [ ] Migrate GitHub Actions CD from SSH to SSM
- [ ] GitHub OIDC deployment role
- [ ] Remove public SSH access
- [ ] Add dedicated `/health` endpoint
- [ ] HTTPS/domain setup
- [ ] Git-SHA image tagging
- [ ] Automated rollback
- [ ] Production monitoring and centralized logs

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| API | FastAPI + Uvicorn |
| ML | XGBoost, scikit-learn, imbalanced-learn |
| Data | pandas, NumPy, PyArrow |
| Explainability | SHAP |
| LLM | Groq |
| Configuration | pydantic-settings, python-dotenv |
| Testing | pytest |
| Containerization | Docker |
| Registry | GitHub Container Registry |
| CI/CD | GitHub Actions |
| Cloud | AWS EC2 |
| Management | AWS Systems Manager |

---

## License

Add the project's intended license here if/when one is selected.

---

## Author

**Gaurav Sehgal**

GitHub: [GauravSehgal12](https://github.com/GauravSehgal12)
