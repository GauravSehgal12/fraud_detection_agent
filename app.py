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
    print("Starting Financial Fraud Risk API...")

    # =====================================================
    # 1. LOAD MODEL AND DATA & INITIALIZE SERVICES
    # =====================================================
    container.load_all()

    features = container.features
    transactions = container.transactions
    raw_transactions = container.raw_transactions
    identity = container.identity
    investigation_history = container.investigation_history
    risk_assessments = container.risk_assessments

    # =====================================================
    # 2. STARTUP INFORMATION
    # =====================================================
    print("\n========== APPLICATION DATA ==========")
    print(f"Model features: {len(features)}")
    print(f"Raw transactions: {raw_transactions.shape[0]} x {raw_transactions.shape[1]}")
    print(f"Identity: {identity.shape[0]} x {identity.shape[1]}")
    print(f"Merged: {transactions.shape[0]} x {transactions.shape[1]}")
    print(f"Investigation history loaded: {len(investigation_history)} rows")
    print(f"Risk assessments loaded: {len(risk_assessments)}")
    print("Cold-start detector initialized")
    print("Rule engine initialized")
    print("Decision engine initialized")
    print("======================================\n")

    # =====================================================
    # 3. CREATE INVESTIGATION TOOLS
    # =====================================================
    tools = FraudInvestigationTools(
        transactions=raw_transactions,
        identity=identity,
        risk_assessments=risk_assessments,
        model=container.model,
        cold_start_detector=container.cold_start_detector,
        rule_engine=container.rule_engine,
        decision_engine=container.decision_engine,
    )
    print("Investigation tools initialized.")

    # =====================================================
    # 4. CREATE LLM & AGENT
    # =====================================================
    llm = FraudLLM()
    print(f"LLM initialized: {llm.model}")

    agent = FraudInvestigationAgent(
        tools=tools,
        llm=llm,
    )
    print("Fraud Investigation Agent initialized")

    # =====================================================
    # 5. REGISTER AGENT
    # =====================================================
    set_agent(agent)
    container.agent = agent

    print("\nFinancial Fraud Risk API is ready.")

    yield

    # =====================================================
    # SHUTDOWN
    # =====================================================
    print("Shutting down Financial Fraud Risk API...")


app = FastAPI(
    title="Financial Fraud Risk & Investigation API",
    version="1.0.0",
    description="Production-oriented financial fraud risk detection and investigation service.",
    lifespan=lifespan,
)

app.include_router(
    router,
    prefix="/api/v1",
)


@app.get("/")
def root():
    return {
        "service": "Financial Fraud Risk and Investigation Agent",
        "version": "1.0.0",
        "status": "running",
    }