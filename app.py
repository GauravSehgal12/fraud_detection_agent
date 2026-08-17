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

    # =====================================================
    # 1. LOAD MODEL AND DATA
    # =====================================================

    container.load_all()

    # -----------------------------------------------------
    # Get data from container
    # -----------------------------------------------------

    features = container.features

    transactions = (
        container.transactions
    )

    identity = (
        container.identity
    )

    investigation_history = (
        container.investigation_history
    )

    risk_assessments = (
        container.risk_assessments
    )

    # =====================================================
    # 2. STARTUP INFORMATION
    # =====================================================

    print(
        "\n========== APPLICATION DATA =========="
    )

    print(
        f"Features loaded: "
        f"{len(features)}"
    )

    print(
        "Raw transactions + identity loaded: "
        f"{transactions.shape}"
    )

    print(
        "Identity loaded: "
        f"{identity.shape}"
    )

    print(
        "Investigation history loaded: "
        f"{len(investigation_history)} rows"
    )

    print(
        "Risk assessments loaded: "
        f"{len(risk_assessments)}"
    )

    print(
        "======================================\n"
    )

    # =====================================================
    # 3. CREATE INVESTIGATION TOOLS
    # =====================================================

    tools = FraudInvestigationTools(
        transactions=transactions,
        identity=identity,
        risk_assessments=risk_assessments,
        model=container.model,
    )

    print(
        "Investigation tools initialized."
    )

    # =====================================================
    # 4. CREATE LLM
    # =====================================================

    llm = FraudLLM()

    print(
        f"LLM initialized: "
        f"{llm.model}"
    )

    # =====================================================
    # 5. CREATE FRAUD AGENT
    # =====================================================

    agent = FraudInvestigationAgent(
        tools=tools,
        llm=llm,
    )

    print(
        "Fraud Investigation Agent initialized."
    )

    # =====================================================
    # 6. REGISTER AGENT
    # =====================================================

    set_agent(agent)

    container.agent = agent

    print(
        "\nFinancial Fraud Risk API is ready."
    )

    yield

    # =====================================================
    # SHUTDOWN
    # =====================================================

    print(
        "Shutting down Financial Fraud Risk API..."
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

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

    lifespan=lifespan,
)


# =========================================================
# ROUTES
# =========================================================

app.include_router(
    router,
    prefix="/api/v1",
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "service": (
            "Financial Fraud Risk "
            "and Investigation Agent"
        ),

        "version": "1.0.0",

        "status": "running",
    }