"""
api/routes.py
---------------
Thin FastAPI route handlers. All business logic lives in api/services.py —
routes just validate the request (via api/schemas.py), call a service,
and translate ServiceError into an HTTPException.

Usage:
    from api.routes import router
    app.include_router(router)
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api import services
from api.database import test_connection
from api.schemas import (
    HealthOut, StockDataOut,
    TrainRequest, TrainResponse,
    PredictionRequest, PredictionOut,
    BacktestRequest, BacktestResultOut,
)
from config.settings import STOCKS

router = APIRouter()


def _run(service_fn, *args, **kwargs):
    """Runs a service function, turning ServiceError into an HTTPException."""
    try:
        return service_fn(*args, **kwargs)
    except services.ServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok", database=test_connection())


@router.get("/stocks", response_model=list[str])
def list_stocks() -> list[str]:
    return STOCKS


@router.get("/stocks/{symbol}/data", response_model=list[StockDataOut])
def get_stock_data(
    symbol:     str,
    start_date: Optional[date] = Query(None),
    end_date:   Optional[date] = Query(None),
) -> list[StockDataOut]:
    return _run(services.get_stock_data_service, symbol, start_date, end_date)


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    return _run(services.train_models_service, request)


@router.post("/predict", response_model=PredictionOut)
def predict(request: PredictionRequest) -> PredictionOut:
    return _run(services.predict_service, request)


@router.get("/predictions/{symbol}", response_model=list[PredictionOut])
def get_predictions(
    symbol:     str,
    start_date: Optional[date] = Query(None),
    end_date:   Optional[date] = Query(None),
) -> list[PredictionOut]:
    return _run(services.get_predictions_service, symbol, start_date, end_date)


@router.post("/backtest", response_model=BacktestResultOut)
def backtest(request: BacktestRequest) -> BacktestResultOut:
    return _run(services.backtest_service, request)


@router.get("/backtests", response_model=list[BacktestResultOut])
def list_backtests(symbol: Optional[str] = Query(None)) -> list[BacktestResultOut]:
    return _run(services.list_backtests_service, symbol)


@router.get("/backtests/{backtest_id}", response_model=BacktestResultOut)
def get_backtest(backtest_id: str) -> BacktestResultOut:
    result = services.get_backtest_service(backtest_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Backtest not found: {backtest_id}")
    return result
