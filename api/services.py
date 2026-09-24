"""
api/services.py
-----------------
DB-touching business logic connecting data/, models/, and backtesting/
to persistence. Route handlers in api/routes.py stay thin and just call
into these functions.

Four services:
    get_stock_data_service()   — read raw OHLCV
    train_models_service()     — preprocess + train LSTM/XGBoost + save
    predict_service()          — load saved models, run ONE live
                                  inference sequence, persist + return it
    backtest_service()         — load saved models, replay them over a
                                  historical range, run backtesting.engine,
                                  persist + return the result

Raises ServiceError (caught by api/routes.py and turned into a 4xx/5xx
HTTP response) for expected failure modes — e.g. "model not trained yet".
"""

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np

from api.database import SessionLocal, Prediction, BacktestResult, Trade
from api.schemas import (
    StockDataOut, TrainRequest, TrainResponse,
    PredictionRequest, PredictionOut,
    BacktestRequest, BacktestResultOut, TradeOut,
)
from backtesting.engine import run_backtest
from config.settings import INITIAL_CAPITAL
from data.preprocessor import (
    preprocess, SequenceConfig,
    save_scaler, load_scaler, load_scaler_metadata,
    build_dated_sequences,
)
from data.feature_engineer import build_full_features
from data.storage import get_stock_data
from models.ensemble import predict_ensemble
from models.lstm_predictor import build_lstm_model, train_lstm, predict_lstm, evaluate_lstm
from models.model_saver import save_lstm, load_lstm, save_xgboost, load_xgboost
from models.xgboost_classifier import train_xgboost, predict_xgboost, evaluate_xgboost
from utils.logger import logger


class ServiceError(Exception):
    """Raised for expected failure modes; api/routes.py maps this to an HTTP error."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message     = message
        self.status_code = status_code


# ─────────────────────────────────────────────────────────────────────────────
# Stock data
# ─────────────────────────────────────────────────────────────────────────────

def get_stock_data_service(
    symbol:     str,
    start_date: Optional[date] = None,
    end_date:   Optional[date] = None,
) -> list[StockDataOut]:
    df = get_stock_data(
        symbol,
        str(start_date) if start_date else None,
        str(end_date) if end_date else None,
    )
    if df is None:
        raise ServiceError(f"No data found for {symbol}. Has the fetcher run for this symbol?", 404)

    return [
        StockDataOut(
            symbol=symbol, date=idx.date(),
            open=row["open"], high=row["high"], low=row["low"],
            close=row["close"], volume=int(row["volume"]),
        )
        for idx, row in df.iterrows()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_models_service(request: TrainRequest) -> TrainResponse:
    data = preprocess(
        request.symbol, str(request.start_date), str(request.end_date),
        include_sentiment=request.include_sentiment,
        include_macro=request.include_macro,
    )
    if data is None:
        raise ServiceError(
            f"Preprocessing failed for {request.symbol} — check that the "
            f"fetcher has populated data for this symbol/range.", 422
        )

    xgb_model   = train_xgboost(data)
    xgb_metrics = evaluate_xgboost(xgb_model, data.X_test, data.y_test)

    lstm_model = build_lstm_model(data.seq_config.sequence_length, data.n_features)
    lstm_model, _ = train_lstm(lstm_model, data)
    lstm_metrics  = evaluate_lstm(lstm_model, data.X_test, data.y_test)

    save_scaler(data.scaler, request.symbol, data)
    save_lstm(lstm_model, request.symbol, metadata={
        "n_features": data.n_features,
        "sequence_length": data.seq_config.sequence_length,
    })
    save_xgboost(xgb_model, request.symbol, metadata={
        "n_features": data.n_features,
    })

    return TrainResponse(
        symbol=request.symbol,
        n_features=data.n_features,
        train_samples=len(data.y_train),
        val_samples=len(data.y_val),
        test_samples=len(data.y_test),
        xgb_metrics=xgb_metrics,
        lstm_metrics=lstm_metrics,
        trained_at=datetime.now(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared: load a symbol's saved model bundle
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_bundle(symbol: str) -> dict:
    """
    Loads the scaler + its metadata + both models for a symbol, or raises
    ServiceError naming exactly what's missing. Shared by predict_service()
    and backtest_service() since both need the same trained bundle.
    """
    meta = load_scaler_metadata(symbol)
    if meta is None or "feature_names" not in meta:
        raise ServiceError(
            f"No trained model found for {symbol}. "
            f"Call POST /train/{symbol} first.", 404
        )

    scaler     = load_scaler(symbol)
    lstm_model = load_lstm(symbol)
    xgb_model  = load_xgboost(symbol)

    missing = [
        name for name, obj in
        [("scaler", scaler), ("lstm", lstm_model), ("xgboost", xgb_model)]
        if obj is None
    ]
    if missing:
        raise ServiceError(
            f"{symbol}: model artifacts missing ({', '.join(missing)}) — "
            f"retrain with POST /train/{symbol}.", 404
        )

    seq_config = SequenceConfig(
        sequence_length=meta["sequence_length"],
        prediction_horizon=meta["prediction_horizon"],
        stride=1,
    )
    training_config = meta.get("training_config", {})

    return {
        "scaler":            scaler,
        "lstm_model":        lstm_model,
        "xgb_model":         xgb_model,
        "feature_names":     meta["feature_names"],
        "seq_config":        seq_config,
        "include_macro":     training_config.get("include_macro", True),
        "include_sentiment": training_config.get("include_sentiment", True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction (live, single sequence)
# ─────────────────────────────────────────────────────────────────────────────

def predict_service(request: PredictionRequest) -> PredictionOut:
    bundle = _load_model_bundle(request.symbol)
    end_date = request.end_date or date.today()

    # Buffer back far enough for technical-indicator warm-up (up to 252
    # trading days for 52-week high/low) plus a full sequence window.
    seq_len = bundle["seq_config"].sequence_length
    buffer_days = 400 + seq_len * 2
    start_date = end_date - timedelta(days=buffer_days)

    df = build_full_features(
        request.symbol, str(start_date), str(end_date),
        include_sentiment=bundle["include_sentiment"],
        include_macro=bundle["include_macro"],
    )
    if df is None or len(df) < seq_len:
        raise ServiceError(
            f"Not enough recent data for {request.symbol} to build an "
            f"inference sequence (need {seq_len} rows).", 422
        )

    feature_names = bundle["feature_names"]
    available = [f for f in feature_names if f in df.columns]
    if len(available) != len(feature_names):
        missing = set(feature_names) - set(available)
        raise ServiceError(
            f"{request.symbol}: {len(missing)} training features missing from "
            f"current data ({list(missing)[:5]}...) — model may need retraining.", 422
        )

    X_scaled = bundle["scaler"].transform(df[feature_names].values)
    X_seq, dates_seq = build_dated_sequences(X_scaled, df.index, bundle["seq_config"])
    if len(X_seq) == 0:
        raise ServiceError(f"Could not build any sequence for {request.symbol}.", 422)

    # Most recent sequence — the "as of today" prediction.
    X_latest    = X_seq[-1:]
    as_of_date  = dates_seq[-1].date() if hasattr(dates_seq[-1], "date") else dates_seq[-1]

    lstm_probs = predict_lstm(bundle["lstm_model"], X_latest)
    xgb_probs  = predict_xgboost(bundle["xgb_model"], X_latest)
    result     = predict_ensemble(lstm_probs, xgb_probs)

    prediction_out = PredictionOut(
        symbol=request.symbol, date=as_of_date,
        lstm_prob_down=float(lstm_probs[0, 0]), lstm_prob_hold=float(lstm_probs[0, 1]), lstm_prob_up=float(lstm_probs[0, 2]),
        xgb_prob_down=float(xgb_probs[0, 0]),   xgb_prob_hold=float(xgb_probs[0, 1]),   xgb_prob_up=float(xgb_probs[0, 2]),
        ensemble_signal=str(result["signal"][0]),
        confidence=float(result["confidence"][0]),
    )
    _upsert_prediction(prediction_out)
    return prediction_out


def _upsert_prediction(p: PredictionOut) -> None:
    db = SessionLocal()
    try:
        row = db.query(Prediction).filter(
            Prediction.symbol == p.symbol, Prediction.date == p.date
        ).first()
        if row is None:
            row = Prediction(symbol=p.symbol, date=p.date)
            db.add(row)

        row.lstm_prob_down = p.lstm_prob_down
        row.lstm_prob_hold = p.lstm_prob_hold
        row.lstm_prob_up   = p.lstm_prob_up
        row.xgb_prob_down  = p.xgb_prob_down
        row.xgb_prob_hold  = p.xgb_prob_hold
        row.xgb_prob_up    = p.xgb_prob_up
        row.ensemble_signal = p.ensemble_signal
        row.confidence       = p.confidence

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save prediction for {p.symbol}/{p.date}: {e}")
        raise
    finally:
        db.close()


def get_predictions_service(
    symbol:     str,
    start_date: Optional[date] = None,
    end_date:   Optional[date] = None,
) -> list[PredictionOut]:
    db = SessionLocal()
    try:
        query = db.query(Prediction).filter(Prediction.symbol == symbol)
        if start_date:
            query = query.filter(Prediction.date >= start_date)
        if end_date:
            query = query.filter(Prediction.date <= end_date)
        rows = query.order_by(Prediction.date.asc()).all()
        return [PredictionOut.model_validate(row) for row in rows]
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Backtest (historical replay)
# ─────────────────────────────────────────────────────────────────────────────

def backtest_service(request: BacktestRequest) -> BacktestResultOut:
    bundle = _load_model_bundle(request.symbol)
    seq_len = bundle["seq_config"].sequence_length

    buffer_days = 400 + seq_len * 2
    buffered_start = request.start_date - timedelta(days=buffer_days)

    df = build_full_features(
        request.symbol, str(buffered_start), str(request.end_date),
        include_sentiment=bundle["include_sentiment"],
        include_macro=bundle["include_macro"],
    )
    if df is None or len(df) < seq_len:
        raise ServiceError(
            f"Not enough data for {request.symbol} to backtest "
            f"{request.start_date}→{request.end_date}.", 422
        )

    feature_names = bundle["feature_names"]
    available = [f for f in feature_names if f in df.columns]
    if len(available) != len(feature_names):
        missing = set(feature_names) - set(available)
        raise ServiceError(
            f"{request.symbol}: {len(missing)} training features missing from "
            f"current data ({list(missing)[:5]}...) — model may need retraining.", 422
        )

    X_scaled = bundle["scaler"].transform(df[feature_names].values)
    X_seq, dates_seq = build_dated_sequences(X_scaled, df.index, bundle["seq_config"])
    if len(X_seq) == 0:
        raise ServiceError(f"Could not build any sequences for {request.symbol}.", 422)

    # Keep only sequences whose date falls inside the REQUESTED range —
    # everything before that was only fetched to warm up indicators/the
    # first sequence window, not to be backtested over.
    mask = [request.start_date <= d.date() <= request.end_date for d in dates_seq]
    X_seq   = X_seq[np.array(mask)]
    dates_seq = [d for d, keep in zip(dates_seq, mask) if keep]

    if len(X_seq) == 0:
        raise ServiceError(
            f"No sequences fall inside {request.start_date}→{request.end_date} "
            f"(range may be too short for sequence_length={seq_len}).", 422
        )

    lstm_probs = predict_lstm(bundle["lstm_model"], X_seq)
    xgb_probs  = predict_xgboost(bundle["xgb_model"], X_seq)
    ensemble_result = predict_ensemble(lstm_probs, xgb_probs)

    range_start = dates_seq[0].date()
    range_end   = dates_seq[-1].date()
    prices = get_stock_data(request.symbol, str(range_start), str(range_end))
    if prices is None:
        raise ServiceError(f"No OHLCV data for {request.symbol} to execute trades against.", 422)

    prices = prices.reindex(dates_seq)
    if prices[["open", "high", "low", "close"]].isna().any().any():
        raise ServiceError(
            f"{request.symbol}: price data doesn't fully cover the signal dates "
            f"(gaps between the feature and price sources) — cannot backtest reliably.", 422
        )

    initial_capital = request.initial_capital or INITIAL_CAPITAL
    summary = run_backtest(
        request.symbol, prices, list(ensemble_result["signal"]),
        initial_capital=initial_capital,
    )

    backtest_id = (
        f"bt_{datetime.now():%Y%m%d_%H%M%S}_"
        f"{request.symbol.replace('.', '_')}_ensemble"
    )
    _save_backtest_result(backtest_id, request.symbol, summary)

    return BacktestResultOut(
        backtest_id=backtest_id, symbol=request.symbol, strategy="ensemble",
        start_date=range_start, end_date=range_end,
        trades=[TradeOut(**t) for t in summary.trades],
        **summary.metrics,
    )


def _save_backtest_result(backtest_id: str, symbol: str, summary) -> None:
    db = SessionLocal()
    try:
        result_row = BacktestResult(
            backtest_id=backtest_id, symbol=symbol, strategy="ensemble",
            start_date=summary.start_date.date() if hasattr(summary.start_date, "date") else summary.start_date,
            end_date=summary.end_date.date() if hasattr(summary.end_date, "date") else summary.end_date,
            **summary.metrics,
        )
        db.add(result_row)

        for trade in summary.trades:
            db.add(Trade(backtest_id=backtest_id, **trade))

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save backtest result {backtest_id}: {e}")
        raise
    finally:
        db.close()


def get_backtest_service(backtest_id: str) -> Optional[BacktestResultOut]:
    db = SessionLocal()
    try:
        row = db.query(BacktestResult).filter(BacktestResult.backtest_id == backtest_id).first()
        if row is None:
            return None
        trades = db.query(Trade).filter(Trade.backtest_id == backtest_id).order_by(Trade.entry_date.asc()).all()
        out = BacktestResultOut.model_validate(row)
        out.trades = [TradeOut.model_validate(t) for t in trades]
        return out
    finally:
        db.close()


def list_backtests_service(symbol: Optional[str] = None) -> list[BacktestResultOut]:
    db = SessionLocal()
    try:
        query = db.query(BacktestResult)
        if symbol:
            query = query.filter(BacktestResult.symbol == symbol)
        rows = query.order_by(BacktestResult.created_at.desc()).all()
        return [BacktestResultOut.model_validate(row) for row in rows]
    finally:
        db.close()
