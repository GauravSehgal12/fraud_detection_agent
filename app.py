from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.agent.tools import FraudInvestigationTools
from src.agent.llm import FraudLLM
from src.agent.fraud_agent import FraudInvestigationAgent

from src.api.route import router, set_agent

from src.services.app_container import AppContainer


container = AppContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "Starting Financial Fraud Risk API..."
    )

    # -----------------------------------------
    # 1. Load persisted model and data
    # -----------------------------------------

    container.load_all()

    features = container.features

    investigation_history = (
        container.investigation_history
    )

    risk_assessments = (
        container.risk_assessments
    )

    print(
        "XGBoost model loaded."
    )

    print(
        f"Features loaded: {len(features)}"
    )

    print(
        "Investigation history loaded: "
        f"{len(investigation_history)} rows"
    )

    print(
        "Risk assessments loaded: "
        f"{len(risk_assessments)}"
    )

    # -----------------------------------------
    # 2. Create investigation tools
    # -----------------------------------------

    tools = FraudInvestigationTools(
        transactions=investigation_history,
        risk_assessments=risk_assessments
    )

    print(
        "Investigation tools initialized."
    )

    # -----------------------------------------
    # 3. Create LLM
    # -----------------------------------------

    llm = FraudLLM()

    print(
        f"LLM initialized: {llm.model}"
    )

    # -----------------------------------------
    # 4. Create Fraud Agent
    # -----------------------------------------

    agent = FraudInvestigationAgent(
        tools=tools,
        llm=llm
    )

    print(
        "Fraud Investigation Agent initialized."
    )

    # -----------------------------------------
    # 5. Register agent with API
    # -----------------------------------------

    set_agent(agent)

    container.agent = agent

    print(
        "Financial Fraud Risk API is ready."
    )

    yield

    # -----------------------------------------
    # Shutdown
    # -----------------------------------------

    print(
        "Shutting down Financial Fraud Risk API..."
    )


app = FastAPI(
    title=(
        "Financial Fraud Risk & "
        "Investigation API"
    ),
    version="1.0.0",
    description=(
        "Production-oriented financial fraud "
        "risk detection and investigation service."
    ),
    lifespan=lifespan
)


app.include_router(
    router,
    prefix="/api/v1"
)


@app.get("/")
def root():

    return {
        "service": (
            "Financial Fraud Risk "
            "and Investigation Agent"
        ),
        "version": "1.0.0",
        "status": "running"
    }