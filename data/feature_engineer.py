"""
data/feature_engineer.py
------------------------
Calculates ALL features for ML training by combining:
  1. Technical indicators  (price-based: RSI, MACD, Bollinger Bands etc.)
  2. Macro indicators      (VIX, USD/INR, crude oil, Nifty 50)
  3. News sentiment        (daily headline sentiment scores)
  4. Label                 (what model should predict: UP/HOLD/DOWN)

Final output: one DataFrame with 50+ columns per stock per day.

Usage:
    from data.feature_engineer import build_full_features

    features_df = build_full_features("RELIANCE.NS", "2020-01-01", "2024-01-01")
    print(features_df.shape)    # (800, 52)
    print(features_df.columns)  # all 52 feature columns + label
"""

import pandas as pd
import numpy as np
import ta

from data.storage       import get_stock_data
from data.macro_fetcher import fetch_macro_data, get_macro_feature_columns
from data.news_fetcher  import fetch_news_sentiment, get_sentiment_feature_columns
from config.settings    import (
    LABEL_THRESHOLD, LABEL_DOWN, LABEL_HOLD, LABEL_UP, STOCKS
)
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION: build_full_features
# Single entry point — call this to get a complete feature DataFrame
# ready for ML training.
# ─────────────────────────────────────────────────────────────────────────────

def build_full_features(
    symbol:     str,
    start_date: str,
    end_date:   str,
    include_sentiment: bool = True,
    include_macro:     bool = True,
) -> pd.DataFrame | None:
    """
    Builds complete feature DataFrame for one stock.

    Combines:
      - Raw OHLCV from PostgreSQL
      - 25+ technical indicators
      - 15 macro indicators (VIX, USD/INR, crude, Nifty)
      - 7 news sentiment features
      - Label column (target variable)

    Args:
        symbol:             e.g. "RELIANCE.NS"
        start_date:         "YYYY-MM-DD"
        end_date:           "YYYY-MM-DD"
        include_sentiment:  Include news sentiment features (default True)
        include_macro:      Include macro features (default True)

    Returns:
        Complete DataFrame ready for ML training, or None if failed.

    Example:
        df = build_full_features("RELIANCE.NS", "2020-01-01", "2024-01-01")
        print(df.shape)     # (800, 52)

        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols].values   # Input to LSTM
        y = df["label"].values         # Target labels
    """

    logger.info(f"Building full features for {symbol} | {start_date} → {end_date}")

    # ── Step 1: Get raw price data from DB ────────────────────────────────────
    raw_df = get_stock_data(symbol, start_date, end_date)
    if raw_df is None or raw_df.empty:
        logger.error(f"No price data found for {symbol}. Run fetcher first.")
        return None

    # ── Step 2: Add technical indicators ─────────────────────────────────────
    df = _add_technical_features(raw_df, symbol)
    if df is None:
        return None

    # ── Step 3: Add macro indicators ─────────────────────────────────────────
    if include_macro:
        df = _merge_macro_features(df, start_date, end_date)

    # ── Step 4: Add news sentiment ────────────────────────────────────────────
    if include_sentiment:
        df = _merge_sentiment_features(df, symbol, start_date, end_date)

    # ── Step 5: Add label (target variable) ──────────────────────────────────
    df = _add_label(df)

    # ── Step 5.5: Defragment DataFrame ───────────────────────────────────────
    # After many .join() and column assignment operations, pandas internally
    # stores the DataFrame in many small memory blocks (fragmented).
    # .copy() consolidates everything into one contiguous memory block.
    # This eliminates the PerformanceWarning and speeds up subsequent ops.
    df = df.copy()

    # ── Step 6: Drop rows with NaN ────────────────────────────────────────────
    # First ~200 rows have NaN due to long-window indicators (SMA_200 etc.)
    rows_before  = len(df)
    df.dropna(inplace=True)
    rows_dropped = rows_before - len(df)

    if len(df) < 100:
        logger.error(
            f"{symbol}: Only {len(df)} clean rows after NaN drop. "
            f"Not enough for training."
        )
        return None

    logger.success(
        f"{symbol}: Features complete | "
        f"{len(df)} rows | "
        f"{len(df.columns)} columns | "
        f"{rows_dropped} NaN rows dropped"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION: build_features_all_stocks
# Runs build_full_features for all stocks in STOCKS list
# ─────────────────────────────────────────────────────────────────────────────

def build_features_all_stocks(
    start_date: str,
    end_date:   str,
    include_sentiment: bool = True,
    include_macro:     bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Builds complete features for ALL stocks in settings.STOCKS.

    Returns:
        { "RELIANCE.NS": df, "INFY.NS": df, ... }
    """

    results = {}

    for symbol in STOCKS:
        logger.info(f"Processing {symbol}...")
        df = build_full_features(
            symbol, start_date, end_date,
            include_sentiment=include_sentiment,
            include_macro=include_macro,
        )
        if df is not None:
            results[symbol] = df

    logger.success(f"Built features for {len(results)}/{len(STOCKS)} stocks")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _add_technical_features
# All price-based technical indicators
# ─────────────────────────────────────────────────────────────────────────────

def _add_technical_features(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame | None:
    """Calculates and adds all technical indicators."""

    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        volume = df["volume"]

        # ── Trend: Moving Averages ────────────────────────────────────────────
        df["sma_20"]  = close.rolling(20).mean()
        df["sma_50"]  = close.rolling(50).mean()
        df["sma_200"] = close.rolling(200).mean()
        df["ema_12"]  = close.ewm(span=12, adjust=False).mean()
        df["ema_26"]  = close.ewm(span=26, adjust=False).mean()

        # Distance from moving averages (how stretched is price?)
        df["dist_sma20"]  = (close - df["sma_20"])  / df["sma_20"]
        df["dist_sma50"]  = (close - df["sma_50"])  / df["sma_50"]
        df["dist_sma200"] = (close - df["sma_200"]) / df["sma_200"]

        # ── Trend: MACD ───────────────────────────────────────────────────────
        df["macd"]        = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # ── Momentum: RSI ─────────────────────────────────────────────────────
        df["rsi_14"]  = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # ── Momentum: Stochastic ──────────────────────────────────────────────
        stoch         = ta.momentum.StochasticOscillator(
            high=high, low=low, close=close, window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # ── Momentum: Rate of Change ──────────────────────────────────────────
        df["roc_10"]  = close.pct_change(10) * 100
        df["roc_20"]  = close.pct_change(20) * 100

        # ── Volatility: Bollinger Bands ───────────────────────────────────────
        bb             = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"]   = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pos"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ── Volatility: ATR ───────────────────────────────────────────────────
        atr            = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        df["atr_14"]   = atr.average_true_range()
        df["atr_pct"]  = df["atr_14"] / close

        # ── Volume: OBV ───────────────────────────────────────────────────────
        df["obv"]      = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        df["vol_ratio"]= volume / volume.rolling(20).mean()
        df["obv_roc"]  = df["obv"].pct_change(5) * 100

        # ── Price Patterns ────────────────────────────────────────────────────
        df["price_change_1d"]  = close.pct_change(1)
        df["price_change_5d"]  = close.pct_change(5)
        df["price_change_20d"] = close.pct_change(20)
        df["high_low_ratio"]   = (high - low) / close
        df["body_ratio"]       = abs(close - open_) / (high - low + 1e-10)
        df["upper_shadow"]     = (high - close.clip(lower=open_)) / (high - low + 1e-10)
        df["lower_shadow"]     = (close.clip(upper=open_) - low)  / (high - low + 1e-10)

        # 52-week position
        high_252           = high.rolling(252).max()
        low_252            = low.rolling(252).min()
        df["position_52w"] = (close - low_252) / (high_252 - low_252 + 1e-10)

        # Time features
        df["day_of_week"]  = df.index.dayofweek
        df["month"]        = df.index.month

        return df

    except Exception as e:
        logger.error(f"Technical indicators failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _merge_macro_features
# Fetches macro data and merges it into the main DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _merge_macro_features(
    df:         pd.DataFrame,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    """
    Fetches macro data (VIX, USD/INR, crude, Nifty) and
    merges it into the main feature DataFrame.

    Uses left join so stock data rows are never lost.
    Missing macro values forward-filled then filled with 0.
    """

    try:
        macro_df = fetch_macro_data(start_date, end_date)

        if macro_df is None or macro_df.empty:
            logger.warning("Macro data unavailable — skipping macro features")
            return df

        # Keep only the derived feature columns (not raw prices)
        macro_cols    = get_macro_feature_columns()
        available     = [c for c in macro_cols if c in macro_df.columns]
        macro_df      = macro_df[available]

        # Make sure index types match before merging
        macro_df.index = pd.to_datetime(macro_df.index)

        # Left join: keep all stock data rows, match macro by date
        df = df.join(macro_df, how="left")

        # Forward fill then fill remaining NaN with 0
        macro_fill_cols = [c for c in available if c in df.columns]
        df[macro_fill_cols] = df[macro_fill_cols].ffill().fillna(0)

        logger.success(f"Merged {len(available)} macro features")

    except Exception as e:
        logger.warning(f"Macro merge failed: {e}. Continuing without macro features.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _merge_sentiment_features
# Fetches news sentiment and merges into main DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _merge_sentiment_features(
    df:         pd.DataFrame,
    symbol:     str,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    """
    Fetches news sentiment for a stock and merges into the main DataFrame.
    Days with no news get sentiment = 0.0 (neutral).
    """

    try:
        sentiment_df = fetch_news_sentiment(symbol, start_date, end_date)

        if sentiment_df is None or sentiment_df.empty:
            logger.warning(f"Sentiment unavailable for {symbol} — using neutral (0.0)")
            # Add neutral sentiment columns
            for col in get_sentiment_feature_columns():
                df[col] = 0.0
            return df

        # Keep only sentiment feature columns
        sent_cols     = get_sentiment_feature_columns()
        available     = [c for c in sent_cols if c in sentiment_df.columns]
        sentiment_df  = sentiment_df[available]

        # Make sure index types match
        sentiment_df.index = pd.to_datetime(sentiment_df.index)

        # Left join: keep all stock data rows, match sentiment by date
        df = df.join(sentiment_df, how="left")

        # Fill NaN sentiment with 0 (neutral)
        sent_fill_cols = [c for c in available if c in df.columns]
        df[sent_fill_cols] = df[sent_fill_cols].fillna(0)

        logger.success(f"Merged {len(available)} sentiment features for {symbol}")

    except Exception as e:
        logger.warning(f"Sentiment merge failed for {symbol}: {e}. Using neutral.")
        for col in get_sentiment_feature_columns():
            df[col] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _add_label
# Creates the target variable — what LSTM should predict
# ─────────────────────────────────────────────────────────────────────────────

def _add_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates label column based on next-day price movement.

    UP   (2): tomorrow's close is 1%+ higher → BUY signal
    DOWN (0): tomorrow's close is 1%+ lower  → SELL signal
    HOLD (1): movement within ±1%            → no action

    Also stores raw future_return for backtesting analysis.
    """

    close         = df["close"]
    next_close    = close.shift(-1)
    future_return = (next_close - close) / close

    df = df.copy()   # ← defragments DataFrame, eliminates PerformanceWarning

    df["label"] = np.where(
        future_return >  LABEL_THRESHOLD, LABEL_UP,
        np.where(
            future_return < -LABEL_THRESHOLD, LABEL_DOWN,
            LABEL_HOLD
        )
    )

    df["future_return"] = future_return

    # Log label distribution
    counts = df["label"].value_counts().sort_index()
    total  = len(df)
    logger.info(
        f"Labels → "
        f"DOWN:{counts.get(0,0)} ({counts.get(0,0)/total:.0%}) | "
        f"HOLD:{counts.get(1,0)} ({counts.get(1,0)/total:.0%}) | "
        f"UP:{counts.get(2,0)} ({counts.get(2,0)/total:.0%})"
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: get_feature_names
# Returns input feature columns (everything except label + future_return)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_names(df: pd.DataFrame) -> list[str]:
    """
    Returns list of feature column names for ML model input.
    Excludes label and future_return (those are targets, not inputs).

    Example:
        feature_cols = get_feature_names(df)
        X = df[feature_cols].values   # Shape: (rows, 50+)
        y = df["label"].values         # Shape: (rows,)
    """
    exclude = {"label", "future_return"}
    return [col for col in df.columns if col not in exclude]