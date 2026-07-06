"""
data/macro_fetcher.py
---------------------
Fetches macroeconomic data that affects Indian stock prices.

What it fetches:
  1. India VIX       → Market fear/volatility index (from NSE via yfinance)
  2. USD/INR rate    → Rupee strength vs dollar
  3. Crude oil price → Brent crude (affects inflation + import costs)
  4. Nifty 50 index  → Overall market direction (benchmark)

Why these matter:
  India VIX:
    - High VIX (>20) = market is scared = stocks usually fall
    - Low VIX (<15)  = market is calm   = stocks usually stable/rising
    - If VIX spikes suddenly → sell signal regardless of technical indicators

  USD/INR:
    - Rupee weakens (USD/INR rises) → BAD for import-heavy sectors (oil, electronics)
    - Rupee weakens → GOOD for IT sector (they earn in USD, report in INR)
    - Rupee strengthens → opposite effects

  Crude Oil:
    - India imports ~85% of its oil
    - Oil price rise → higher inflation → RBI raises rates → markets fall
    - Directly affects: ONGC, Reliance (refining), airlines, paint companies

  Nifty 50:
    - Individual stock often moves WITH the market
    - If Nifty is in strong uptrend, individual stocks get a tailwind
    - If Nifty is falling, most stocks fall regardless of their own technicals

Usage:
    from data.macro_fetcher import fetch_macro_data, get_macro_features

    macro_df = fetch_macro_data("2020-01-01", "2024-01-01")
    print(macro_df.columns)
    # ['vix', 'vix_change', 'usd_inr', 'usd_inr_change',
    #  'crude_oil', 'crude_change', 'nifty', 'nifty_return']
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# SYMBOLS
# Yahoo Finance symbols for macro indicators
# ─────────────────────────────────────────────────────────────────────────────

MACRO_SYMBOLS = {
    # ── Indian Market ─────────────────────────────────────────────────────────
    "vix":         "^INDIAVIX",  # India Volatility Index — market fear gauge
    "nifty":       "^NSEI",      # Nifty 50 — overall Indian market direction

    # ── Currency ──────────────────────────────────────────────────────────────
    "usd_inr":     "INR=X",      # USD/INR — rupee strength vs dollar
    "dxy":         "DX-Y.NYB",   # Dollar Index — global dollar strength
                                 # DXY up = dollar strong = bad for emerging markets

    # ── US Markets (strongest global correlation with India) ──────────────────
    "sp500":       "^GSPC",      # S&P 500 — US broad market
                                 # When S&P falls, India usually follows next day
    "nasdaq":      "^IXIC",      # Nasdaq — US tech index
                                 # Directly affects Indian IT sector (TCS, Infy, Wipro)

    # ── Asian Markets (same-day leading indicators) ───────────────────────────
    "nikkei":      "^N225",      # Nikkei 225 — Japan (opens before India)
    "hangseng":    "^HSI",       # Hang Seng — Hong Kong/China proxy

    # ── Commodities ───────────────────────────────────────────────────────────
    "crude_oil":   "BZ=F",       # Brent Crude — India imports 85% of oil
                                 # Oil up = inflation up = RBI raises rates = markets fall
    "natural_gas": "NG=F",       # Natural Gas — affects fertilizer, power, industrial
    "gold":        "GC=F",       # Gold — inverse relationship with stocks
                                 # Gold up = fear up = stocks may fall
}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: fetch_macro_data
# Downloads all macro indicators for a date range.
# Returns a single DataFrame with one row per trading day.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_macro_data(
    start_date: str,
    end_date:   str,
) -> pd.DataFrame | None:
    """
    Fetches all macro indicators and returns as a combined DataFrame.

    Args:
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"

    Returns:
        DataFrame with DatetimeIndex and columns:
            vix, vix_change, vix_regime,
            usd_inr, usd_inr_change, usd_inr_trend,
            crude_oil, crude_change,
            nifty_return, nifty_trend,
            market_stress  ← composite stress indicator

    Example:
        df = fetch_macro_data("2022-01-01", "2024-01-01")
        print(df.head(2))
        #               vix   vix_change  usd_inr  usd_inr_change  crude_oil
        # date
        # 2022-01-03   17.2      -0.02     74.3        0.001        79.5
        # 2022-01-04   16.8      -0.02     74.4        0.001        80.1
    """

    logger.info(f"Fetching macro data | {start_date} → {end_date}")

    raw_data = {}

    # ── Fetch each macro indicator ────────────────────────────────────────────
    for name, symbol in MACRO_SYMBOLS.items():
        try:
            df = yf.download(
                symbol,
                start      = start_date,
                end        = end_date,
                progress   = False,
                auto_adjust= True,
            )

            if df.empty:
                logger.warning(f"No data for {name} ({symbol})")
                continue

            # Handle MultiIndex columns (yfinance quirk)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # We only need the closing price
            raw_data[name] = df["Close"].copy()
            raw_data[name].index = raw_data[name].index.tz_localize(None)

            logger.success(f"  ✅ {name} ({symbol}): {len(raw_data[name])} rows")
            time.sleep(0.5)   # Small delay between requests

        except Exception as e:
            logger.error(f"  ❌ Failed to fetch {name} ({symbol}): {e}")

    if not raw_data:
        logger.error("Failed to fetch any macro data")
        return None

    # ── Combine into single DataFrame ─────────────────────────────────────────
    # Each indicator has its own index (trading days may differ)
    # We combine with outer join then forward-fill missing values
    # (e.g. crude oil trades on days Indian market is closed)

    macro_df = pd.DataFrame(raw_data)

    # Forward fill: if VIX has no data on a day, use previous day's value
    # This handles holidays where some markets are open and others aren't
    macro_df.ffill(inplace=True)
    macro_df.bfill(inplace=True)   # Back fill for any leading NaN
    macro_df.index.name = "date"

    # ── Rename columns clearly ────────────────────────────────────────────────
    rename_map = {
        "vix":       "vix",
        "usd_inr":   "usd_inr",
        "crude_oil": "crude_oil",
        "nifty":     "nifty",
    }
    macro_df.rename(columns=rename_map, inplace=True)

    # ── Calculate derived features ────────────────────────────────────────────
    macro_df = _add_macro_features(macro_df)

    logger.success(
        f"Macro data ready: {len(macro_df)} rows | "
        f"{len(macro_df.columns)} columns"
    )
    return macro_df


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _add_macro_features
# Calculates derived features from raw macro data.
# Raw values alone are less useful — changes and regimes matter more.
# ─────────────────────────────────────────────────────────────────────────────

def _add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived macro features on top of raw macro values.

    Why derivatives and not just raw values?
        Raw VIX = 18.5 → is that high or low? Depends on recent context.
        VIX change = +3.2 → VIX jumped 3.2 points TODAY → that's clearly alarming.
        Rate of change is more actionable than absolute value.
    """

    # ── VIX Features ──────────────────────────────────────────────────────────

    if "vix" in df.columns:
        # 1-day change in VIX
        # Sudden VIX spike (+5 in one day) = panic entering market
        df["vix_change"]    = df["vix"].pct_change(1)

        # VIX 5-day moving average (smoothed trend)
        df["vix_ma5"]       = df["vix"].rolling(5).mean()

        # VIX regime: categorical label based on absolute level
        # Low (<15): calm market, conditions good for buying
        # Medium (15-20): normal market uncertainty
        # High (20-25): elevated fear
        # Extreme (>25): panic/crisis mode
        df["vix_regime"] = pd.cut(
            df["vix"],
            bins   = [0, 15, 20, 25, 100],
            labels = [0, 1, 2, 3],          # 0=calm, 1=normal, 2=high, 3=extreme
        ).astype(float)

        # VIX spike flag: did VIX jump more than 10% in one day?
        # 1 = yes (fear entering market suddenly), 0 = no
        df["vix_spike"] = (df["vix_change"] > 0.10).astype(int)

    # ── USD/INR Features ──────────────────────────────────────────────────────

    if "usd_inr" in df.columns:
        # Daily change in USD/INR rate
        # Positive = rupee weakening (dollar getting stronger vs rupee)
        # Negative = rupee strengthening
        df["usd_inr_change"] = df["usd_inr"].pct_change(1)

        # 20-day trend: is rupee generally weakening or strengthening?
        # Positive = rupee on weakening trend
        # Negative = rupee on strengthening trend
        df["usd_inr_trend"] = df["usd_inr"].pct_change(20)

        # How far is current rate from 20-day average?
        # Tells model if rupee is unusually weak/strong right now
        usd_ma20            = df["usd_inr"].rolling(20).mean()
        df["usd_inr_dev"]   = (df["usd_inr"] - usd_ma20) / usd_ma20

    # ── Crude Oil Features ────────────────────────────────────────────────────

    if "crude_oil" in df.columns:
        # Daily oil price change
        df["crude_change"]  = df["crude_oil"].pct_change(1)

        # 5-day oil trend (short term direction)
        df["crude_trend"]   = df["crude_oil"].pct_change(5)

        # Is oil in a high-price regime? (above $80/barrel = painful for India)
        # 1 = expensive oil (inflationary pressure), 0 = manageable
        df["crude_high"]    = (df["crude_oil"] > 80).astype(int)

    # ── Nifty 50 Features ─────────────────────────────────────────────────────

    if "nifty" in df.columns:
        # Daily Nifty return
        # If market is up 2% today, most individual stocks also rise
        df["nifty_return"]  = df["nifty"].pct_change(1)

        # 20-day Nifty trend: is the broad market in uptrend or downtrend?
        df["nifty_trend"]   = df["nifty"].pct_change(20)

        # Nifty 50-day moving average distance
        # Is market above or below its 50-day average? (overall health)
        nifty_ma50          = df["nifty"].rolling(50).mean()
        df["nifty_vs_ma50"] = (df["nifty"] - nifty_ma50) / nifty_ma50

    # ── Market Stress Score (composite indicator) ─────────────────────────────
    # Combines VIX + Rupee weakness + Oil rise into one "stress" number
    # High stress score = bad conditions for stock market
    # Low stress score  = favorable conditions
    #
    # Formula:
    #   Stress = VIX_regime + usd_inr_trend_normalized + crude_high
    #   Range: roughly 0 (no stress) to 5 (maximum stress)

    stress_components = []

    if "vix_regime" in df.columns:
        stress_components.append(df["vix_regime"])          # 0 to 3

    if "usd_inr_trend" in df.columns:
        # Normalize rupee weakness to 0-1 range
        usd_stress = df["usd_inr_trend"].clip(0, 0.05) / 0.05
        stress_components.append(usd_stress)                # 0 to 1

    if "crude_high" in df.columns:
        stress_components.append(df["crude_high"])          # 0 or 1

    if stress_components:
        df["market_stress"] = sum(stress_components)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: get_macro_features
# Returns only the derived feature columns (not raw prices).
# These are the columns that get merged into the main feature DataFrame.
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_feature_columns() -> list[str]:
    """
    Returns list of macro feature column names that get added to main DataFrame.
    Used by feature_engineer.py to know which columns to merge in.
    """
    return [
        "vix", "vix_change", "vix_ma5", "vix_regime", "vix_spike",
        "usd_inr_change", "usd_inr_trend", "usd_inr_dev",
        "crude_change", "crude_trend", "crude_high",
        "nifty_return", "nifty_trend", "nifty_vs_ma50",
        "market_stress",
    ]