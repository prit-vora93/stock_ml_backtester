"""
api/schemas.py
----------------
Pydantic request/response models for the FastAPI layer.

Response schemas that mirror a DB table (StockDataOut, PredictionOut)
use from_attributes=True so they can be built directly from SQLAlchemy
row objects: StockDataOut.model_validate(row).

Usage:
    from api.schemas import TrainRequest, PredictionOut, BacktestRequest
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status:   str
    database: bool


# ─────────────────────────────────────────────────────────────────────────────
# Stock data
# ─────────────────────────────────────────────────────────────────────────────

class StockDataOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    date:   date
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    symbol:            str
    start_date:        date
    end_date:           date
    include_macro:      bool = True
    include_sentiment:  bool = True


class TrainResponse(BaseModel):
    symbol:          str
    n_features:      int
    train_samples:    int
    val_samples:       int
    test_samples:      int
    xgb_metrics:       dict
    lstm_metrics:       dict
    trained_at:          datetime


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    symbol:   str
    end_date: Optional[date] = None   # defaults to today if omitted


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol:          str
    date:            date
    lstm_prob_down:  float
    lstm_prob_hold:  float
    lstm_prob_up:    float
    xgb_prob_down:   float
    xgb_prob_hold:   float
    xgb_prob_up:     float
    ensemble_signal: str
    confidence:      float


# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbol:           str
    start_date:       date
    end_date:         date
    initial_capital:  Optional[float] = None   # defaults to settings.INITIAL_CAPITAL


class TradeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol:      str
    entry_date:  date
    entry_price: float
    exit_date:   Optional[date]  = None
    exit_price:  Optional[float] = None
    quantity:    int
    pnl:         Optional[float] = None
    pnl_pct:     Optional[float] = None
    exit_reason: Optional[str]   = None


class BacktestResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    backtest_id:     str
    symbol:          str
    strategy:        str
    start_date:      date
    end_date:        date
    initial_capital: float
    final_capital:   Optional[float] = None
    total_return:    Optional[float] = None
    annual_return:   Optional[float] = None
    sharpe_ratio:    Optional[float] = None
    sortino_ratio:   Optional[float] = None
    max_drawdown:    Optional[float] = None
    win_rate:        Optional[float] = None
    profit_factor:   Optional[float] = None
    total_trades:    Optional[int]   = None
    winning_trades:  Optional[int]   = None
    losing_trades:   Optional[int]   = None
    trades:          list[TradeOut]  = Field(default_factory=list)
