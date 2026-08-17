from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="Financial Fraud Risk & Investigation API",
    version="1.0.0",
    description=(
        "Production-oriented fraud risk "
        "and investigation service."
    )
)

app.include_router(
    router,
    prefix="/api/v1"
)


@app.get("/")
def root():

    return {
        "service": "Financial Fraud Risk Agent",
        "version": "1.0.0",
        "status": "running"
    }