"""
tests/test_api.py
-------------------
Tests for the FastAPI layer (api/main.py, routes.py, services.py,
schemas.py) using FastAPI's TestClient against the local Postgres
instance, with synthetic OHLCV data seeded directly (no network —
training requests pass include_macro=False, include_sentiment=False
so no Yahoo Finance / RSS calls happen).

Trains a real (tiny, fast-ish) LSTM + XGBoost once per test session
via a module-scoped fixture, then reuses it across predict/backtest/
list tests — training is the slow part (~15-20s), everything after
just loads the saved models.

Run:
    pytest tests/test_api.py -v
"""

import os
import shutil
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.database import SessionLocal, StockData, Prediction, BacktestResult, Trade, create_tables
from api.main import app
from config.settings import MODELS_DIR
from data.storage import save_stock_data

TRAINED_SYMBOL   = "PYTAPI_TRAIN.NS"     # StockData.symbol is String(20) -- keep short
UNTRAINED_SYMBOL = "PYTAPI_UNTR.NS"


def _seed_synthetic_prices(symbol: str, n: int = 750, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    returns = rng.normal(0.0003, 0.015, n)
    close = 1500 * np.cumprod(1 + returns)
    df = pd.DataFrame({
        "open":  close * rng.uniform(0.995, 1.0, n),
        "high":  close * rng.uniform(1.0, 1.02, n),
        "low":   close * rng.uniform(0.98, 1.0, n),
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n),
    }, index=dates)
    df.index.name = "date"
    save_stock_data(symbol, df)
    return dates[0].date(), dates[-1].date()


def _cleanup_symbol(symbol: str) -> None:
    db = SessionLocal()
    try:
        db.query(Trade).filter(
            Trade.backtest_id.in_(
                db.query(BacktestResult.backtest_id).filter(BacktestResult.symbol == symbol)
            )
        ).delete(synchronize_session=False)
        db.query(BacktestResult).filter(BacktestResult.symbol == symbol).delete()
        db.query(Prediction).filter(Prediction.symbol == symbol).delete()
        db.query(StockData).filter(StockData.symbol == symbol).delete()
        db.commit()
    finally:
        db.close()

    safe = symbol.replace(".", "_")
    for pattern in [f"scaler_{safe}", f"lstm_{safe}", f"xgb_{safe}", f"ensemble_{safe}"]:
        for fname in os.listdir(MODELS_DIR):
            if fname.startswith(pattern):
                path = os.path.join(MODELS_DIR, fname)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)


@pytest.fixture(scope="module")
def client():
    create_tables()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def trained_symbol(client):
    """Seeds synthetic data, trains real models via the API, cleans up after."""
    start, end = _seed_synthetic_prices(TRAINED_SYMBOL)
    train_end = end - timedelta(days=90)   # leave room for a later backtest range

    resp = client.post("/train", json={
        "symbol": TRAINED_SYMBOL,
        "start_date": str(start),
        "end_date": str(train_end),
        "include_macro": False,
        "include_sentiment": False,
    })
    assert resp.status_code == 200, resp.text

    yield TRAINED_SYMBOL, start, end

    _cleanup_symbol(TRAINED_SYMBOL)


@pytest.fixture(scope="module")
def untrained_symbol(client):
    start, end = _seed_synthetic_prices(UNTRAINED_SYMBOL, seed=99)
    yield UNTRAINED_SYMBOL, start, end
    _cleanup_symbol(UNTRAINED_SYMBOL)


# ═════════════════════════════════════════════════════════════════════════════
# TestHealthAndStocks
# ═════════════════════════════════════════════════════════════════════════════

class TestHealthAndStocks:

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "database": True}

    def test_list_stocks(self, client):
        resp = client.get("/stocks")
        assert resp.status_code == 200
        assert "RELIANCE.NS" in resp.json()


# ═════════════════════════════════════════════════════════════════════════════
# TestStockDataEndpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestStockDataEndpoint:

    def test_get_data_for_known_symbol(self, client, untrained_symbol):
        symbol, start, end = untrained_symbol
        resp = client.get(f"/stocks/{symbol}/data")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) > 0
        assert rows[0]["symbol"] == symbol
        assert set(rows[0].keys()) == {"symbol", "date", "open", "high", "low", "close", "volume"}

    def test_get_data_date_filtered(self, client, untrained_symbol):
        symbol, start, end = untrained_symbol
        resp = client.get(f"/stocks/{symbol}/data?start_date={start}&end_date={start}")
        assert resp.status_code == 200
        rows = resp.json()
        assert all(r["date"] == str(start) for r in rows)

    def test_get_data_unknown_symbol_404(self, client):
        resp = client.get("/stocks/NOPE_NOT_REAL.NS/data")
        assert resp.status_code == 404


# ═════════════════════════════════════════════════════════════════════════════
# TestTrainPredictBacktestFlow
# ═════════════════════════════════════════════════════════════════════════════

class TestTrainPredictBacktestFlow:

    def test_train_response_shape(self, trained_symbol):
        # Training itself already happened in the fixture — just assert
        # on the side effects (files exist) since the fixture doesn't
        # hand back the response body.
        safe = TRAINED_SYMBOL.replace(".", "_")
        assert os.path.exists(os.path.join(MODELS_DIR, f"scaler_{safe}.pkl"))
        assert os.path.exists(os.path.join(MODELS_DIR, f"lstm_{safe}.keras"))
        assert os.path.exists(os.path.join(MODELS_DIR, f"xgb_{safe}.pkl"))

    def test_predict_after_training(self, client, trained_symbol):
        symbol, start, end = trained_symbol
        resp = client.post("/predict", json={"symbol": symbol, "end_date": str(end)})
        assert resp.status_code == 200, resp.text

        body = resp.json()
        assert body["symbol"] == symbol
        assert body["ensemble_signal"] in {"BUY", "HOLD", "SELL"}
        assert 0.0 <= body["confidence"] <= 1.0
        probs = [body["lstm_prob_down"], body["lstm_prob_hold"], body["lstm_prob_up"]]
        assert abs(sum(probs) - 1.0) < 1e-3

    def test_predictions_are_persisted(self, client, trained_symbol):
        symbol, start, end = trained_symbol
        # test_predict_after_training already ran and saved one
        resp = client.get(f"/predictions/{symbol}")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_backtest_after_training(self, client, trained_symbol):
        symbol, start, end = trained_symbol
        backtest_start = end - timedelta(days=80)

        resp = client.post("/backtest", json={
            "symbol": symbol,
            "start_date": str(backtest_start),
            "end_date": str(end),
        })
        assert resp.status_code == 200, resp.text

        body = resp.json()
        assert body["symbol"] == symbol
        assert body["backtest_id"].startswith("bt_")
        assert body["strategy"] == "ensemble"
        assert body["total_trades"] is not None
        assert body["total_trades"] == len(body["trades"])
        assert body["initial_capital"] > 0

    def test_get_backtest_by_id(self, client, trained_symbol):
        symbol, start, end = trained_symbol
        backtest_start = end - timedelta(days=80)
        create_resp = client.post("/backtest", json={
            "symbol": symbol, "start_date": str(backtest_start), "end_date": str(end),
        })
        backtest_id = create_resp.json()["backtest_id"]

        resp = client.get(f"/backtests/{backtest_id}")
        assert resp.status_code == 200
        assert resp.json()["backtest_id"] == backtest_id

    def test_get_backtest_unknown_id_404(self, client):
        resp = client.get("/backtests/bt_does_not_exist")
        assert resp.status_code == 404

    def test_list_backtests_filtered_by_symbol(self, client, trained_symbol):
        symbol, start, end = trained_symbol
        resp = client.get(f"/backtests?symbol={symbol}")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) >= 1
        assert all(r["symbol"] == symbol for r in results)


# ═════════════════════════════════════════════════════════════════════════════
# TestErrorsWithoutTraining
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorsWithoutTraining:

    def test_predict_without_training_404(self, client, untrained_symbol):
        symbol, start, end = untrained_symbol
        resp = client.post("/predict", json={"symbol": symbol})
        assert resp.status_code == 404
        assert "train" in resp.json()["detail"].lower()

    def test_backtest_without_training_404(self, client, untrained_symbol):
        symbol, start, end = untrained_symbol
        resp = client.post("/backtest", json={
            "symbol": symbol, "start_date": str(start), "end_date": str(end),
        })
        assert resp.status_code == 404

    def test_predict_completely_unknown_symbol_404(self, client):
        resp = client.post("/predict", json={"symbol": "TOTALLY_UNKNOWN.NS"})
        assert resp.status_code == 404
