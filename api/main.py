"""
api/main.py
-------------
FastAPI application entry point. Creates the app, wires up api/routes.py,
and ensures DB tables exist on startup.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Or:
    python -m api.main
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.database import create_tables
from api.routes import router
from config.settings import API_HOST, API_PORT, DEBUG
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting stock_ml_backtester API...")
    create_tables()
    yield
    logger.info("Shutting down stock_ml_backtester API...")


app = FastAPI(
    title="Stock ML Backtester API",
    description=(
        "Serves OHLCV data, LSTM/XGBoost/ensemble predictions, and "
        "backtest results for the stock_ml_backtester project."
    ),
    version="1.0.0",
    debug=DEBUG,
    lifespan=lifespan,
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=DEBUG)
