"""
data/fetcher.py
---------------
Downloads historical OHLCV stock data from Yahoo Finance using yfinance.

Responsibilities:
  - Fetch data for one stock or all stocks
  - Handle errors (network issues, invalid symbols, empty responses)
  - Clean raw yfinance output into consistent format
  - Validate data quality before returning

What this file does NOT do:
  - Save to database (that's storage.py)
  - Calculate indicators (that's feature_engineer.py)
  - Train models (that's models/)

Usage:
    from data.fetcher import fetch_stock_data, fetch_all_stocks

    df = fetch_stock_data("RELIANCE.NS", "2020-01-01", "2024-01-01")
    all_data = fetch_all_stocks()
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config.settings import STOCKS, DATA_YEARS
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# These numbers control fetcher behavior.
# Defined here (not in settings.py) because they're fetcher-specific details.
# ─────────────────────────────────────────────────────────────────────────────

# Minimum rows needed to be useful for ML training
# 200 rows = ~10 months of trading days
# Less than this = not enough history for LSTM to learn anything meaningful
MIN_ROWS_REQUIRED = 200

# How many times to retry a failed download before giving up
# Network hiccups happen — retrying 3 times handles most temporary failures
MAX_RETRIES = 3

# Seconds to wait between retry attempts
# Gives the network time to recover before trying again
RETRY_DELAY = 5

# Seconds to wait between fetching different stocks
# Yahoo Finance will block your IP if you make too many requests too fast
# 1 second gap = safe rate, won't get blocked
RATE_LIMIT_DELAY = 1


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: fetch_stock_data
# The main function. Downloads data for ONE stock for a given date range.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(
    symbol:     str,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame | None:
    """
    Downloads daily OHLCV data for one stock from Yahoo Finance.

    Args:
        symbol:     NSE stock symbol e.g. "RELIANCE.NS"
                    The ".NS" suffix is mandatory for Indian stocks
        start_date: Start date as string "YYYY-MM-DD"
        end_date:   End date as string "YYYY-MM-DD"

    Returns:
        pd.DataFrame with:
            - Index: DatetimeIndex (one row per trading day)
            - Columns: open, high, low, close, volume (all lowercase)
        None if fetch fails or data doesn't pass validation

    Example:
        df = fetch_stock_data("RELIANCE.NS", "2020-01-01", "2024-01-01")

        print(df.shape)
        # (996, 5)   ← ~996 trading days, 5 columns

        print(df.head(2))
        #               open     high      low    close    volume
        # date
        # 2020-01-02  1549.9  1556.7  1530.1  1549.0   5823400
        # 2020-01-03  1543.0  1543.0  1506.2  1510.5   8255600
    """

    logger.info(f"Fetching {symbol} | {start_date} → {end_date}")

    # ── Retry loop ────────────────────────────────────────────────────────────
    # We try up to MAX_RETRIES times.
    # If attempt 1 fails (network error), we wait and try again.
    # If all attempts fail, we return None (caller handles the failure).
    # This makes the fetcher robust against temporary network issues.

    for attempt in range(1, MAX_RETRIES + 1):
        try:

            # ── The actual download ───────────────────────────────────────────
            # yf.download() is the core yfinance function.
            #
            # progress=False:
            #   Disables the download progress bar in terminal.
            #   We use our own logger instead — cleaner output.
            #
            # auto_adjust=True:
            #   Adjusts historical prices for corporate actions like:
            #   - Stock splits (e.g. 10:1 split would look like 90% crash without this)
            #   - Dividends (price drops on ex-dividend date without this)
            #   With auto_adjust=True, all historical prices are comparable.
            #   ALWAYS use this for ML — you don't want fake "crashes" in training data.

            raw_df = yf.download(
                symbol,
                start      = start_date,
                end        = end_date,
                progress   = False,
                auto_adjust= True,
            )

            # ── Empty response check ──────────────────────────────────────────
            # yfinance returns an empty DataFrame (not an error) for:
            #   - Invalid symbols (e.g. "RELIANCE" without ".NS")
            #   - Dates before the stock was listed
            #   - Weekends/holidays only in the date range
            # We check for this explicitly.

            if raw_df is None or raw_df.empty:
                logger.warning(
                    f"No data returned for {symbol} "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                if attempt < MAX_RETRIES:
                    logger.info(f"Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue        # Go to next attempt
                else:
                    logger.error(f"All {MAX_RETRIES} attempts failed for {symbol}")
                    return None     # Give up, return None

            # ── Clean the raw data ────────────────────────────────────────────
            # yfinance's raw output has inconsistencies we need to fix.
            # _clean_dataframe() handles all of them (explained below).

            df = _clean_dataframe(raw_df, symbol)

            if df is None:
                # Cleaning failed — already logged inside _clean_dataframe
                return None

            # ── Validate data quality ─────────────────────────────────────────
            # Check the cleaned data is actually usable for ML.
            # _validate_data() checks row count, negatives, NaN%, etc.

            if not _validate_data(df, symbol):
                # Validation failed — already logged inside _validate_data
                return None

            # ── Success ───────────────────────────────────────────────────────
            logger.success(
                f"Fetched {symbol}: "
                f"{len(df)} rows | "
                f"{df.index[0].date()} → {df.index[-1].date()} | "
                f"Close range: ₹{df['close'].min():.0f}–₹{df['close'].max():.0f}"
            )
            return df

        except Exception as e:
            # Catches unexpected errors: SSL errors, timeout, server errors, etc.
            logger.error(f"Unexpected error fetching {symbol} (attempt {attempt}): {e}")

            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Giving up on {symbol} after {MAX_RETRIES} attempts")
                return None

    return None     # Should never reach here, but satisfies Python's return requirement


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: fetch_all_stocks
# Fetches data for ALL stocks defined in settings.py STOCKS list.
# Calls fetch_stock_data() once per stock with rate limiting between calls.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_stocks(
    start_date: str | None = None,
    end_date:   str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetches historical data for all stocks in settings.STOCKS.

    Args:
        start_date: Optional. Defaults to DATA_YEARS ago from today.
        end_date:   Optional. Defaults to today.

    Returns:
        Dictionary mapping symbol → DataFrame
        Only successful fetches are included (failed ones are excluded).

        Example:
        {
            "RELIANCE.NS": DataFrame(1250 rows),
            "INFY.NS":     DataFrame(1248 rows),
            "TCS.NS":      DataFrame(1247 rows),
            ...
        }

    Note:
        Takes ~30 seconds for 5 stocks due to rate limiting (1s between requests).
        This is intentional — don't remove the sleep.
    """

    # ── Calculate default date range if not provided ──────────────────────────
    # end_date defaults to today
    # start_date defaults to DATA_YEARS (5) years ago
    # This gives us ~1250 trading days per stock — enough for ML training

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    if start_date is None:
        start_dt   = datetime.today() - timedelta(days=DATA_YEARS * 365)
        start_date = start_dt.strftime("%Y-%m-%d")

    logger.info(f"Fetching {len(STOCKS)} stocks | {start_date} → {end_date}")
    logger.info(f"Stocks: {STOCKS}")
    logger.info(f"Estimated time: ~{len(STOCKS) * RATE_LIMIT_DELAY + 10}s")

    results = {}    # Will hold: { "RELIANCE.NS": df, "INFY.NS": df, ... }
    failed  = []    # Will hold symbols that failed

    for i, symbol in enumerate(STOCKS, start=1):

        logger.info(f"[{i}/{len(STOCKS)}] {symbol}...")

        df = fetch_stock_data(symbol, start_date, end_date)

        if df is not None:
            results[symbol] = df
            logger.success(f"  ✅ {symbol}: {len(df)} rows")
        else:
            failed.append(symbol)
            logger.error(f"  ❌ {symbol}: Failed")

        # ── Rate limiting ─────────────────────────────────────────────────────
        # Wait 1 second between each stock request.
        # Without this, Yahoo Finance sees rapid-fire requests and may:
        #   - Return empty data silently
        #   - Block your IP temporarily
        # We skip the wait after the last stock (no next request coming).

        if i < len(STOCKS):
            time.sleep(RATE_LIMIT_DELAY)

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.success(f"Done: {len(results)} succeeded, {len(failed)} failed")

    if failed:
        logger.warning(f"Failed symbols: {failed}")
        logger.info("Tip: Check symbol spelling and .NS suffix")

    for symbol, df in results.items():
        logger.info(
            f"  {symbol:<15} | "
            f"{len(df)} rows | "
            f"₹{df['close'].iloc[-1]:.0f} (latest close)"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: fetch_latest_data
# Fetches only the most recent few days for a stock.
# Used for daily updates — no need to re-fetch 5 years every day.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_latest_data(symbol: str, days: int = 5) -> pd.DataFrame | None:
    """
    Fetches only the most recent N trading days for a stock.
    Useful for daily database updates after initial bulk fetch.

    Args:
        symbol: Stock symbol e.g. "RELIANCE.NS"
        days:   Number of recent trading days to return (default 5 = 1 week)

    Returns:
        DataFrame with the last N trading days, or None if failed

    Example:
        latest = fetch_latest_data("RELIANCE.NS", days=3)
        #               open     high    low    close    volume
        # 2024-01-11  2456.0  2480.0  2440.0  2465.0  5200000
        # 2024-01-12  2467.0  2495.0  2460.0  2488.0  6100000
        # 2024-01-15  2488.0  2510.0  2477.0  2502.0  5900000
    """

    end_date = datetime.today().strftime("%Y-%m-%d")

    # Fetch days + 5 extra to account for weekends and holidays
    # (weekends have no trading data so we fetch more and trim)
    start_dt   = datetime.today() - timedelta(days=days + 5)
    start_date = start_dt.strftime("%Y-%m-%d")

    df = fetch_stock_data(symbol, start_date, end_date)

    if df is not None and len(df) >= 1:
        # Return only the last N actual trading days
        return df.tail(days)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPER: _clean_dataframe
# Fixes all the quirks in raw yfinance output.
# Called by fetch_stock_data() before validation.
# Name starts with _ to signal "this is internal, don't call from outside"
# ─────────────────────────────────────────────────────────────────────────────

def _clean_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """
    Cleans raw yfinance output into a consistent format.

    Problems this fixes:

    Problem 1 — MultiIndex columns:
        yfinance sometimes returns nested column headers like:
            ("Close", "RELIANCE.NS"), ("Open", "RELIANCE.NS")
        We need simple: Close, Open, High, Low, Volume
        Fix: df.columns = df.columns.get_level_values(0)

    Problem 2 — Extra columns:
        yfinance may include "Dividends", "Stock Splits" columns.
        We only need Open, High, Low, Close, Volume.
        Fix: keep only the 5 OHLCV columns

    Problem 3 — Uppercase column names:
        yfinance returns "Open", "Close" etc. (capital first letter)
        We want "open", "close" (all lowercase for consistency)
        Fix: df.columns = ["open", "high", "low", "close", "volume"]

    Problem 4 — Timezone-aware index:
        yfinance returns dates like: 2024-01-15 00:00:00+05:30
        PostgreSQL Date column doesn't need timezone
        Fix: df.index = df.index.tz_localize(None)

    Problem 5 — Float volume:
        yfinance sometimes returns volume as float (e.g. 5823400.0)
        Our DB column expects integer
        Fix: df["volume"] = df["volume"].fillna(0).astype(int)
    """

    try:

        # ── Fix Problem 1: MultiIndex columns ────────────────────────────────
        # Check if columns are multi-level (nested)
        # isinstance() checks if df.columns is a MultiIndex object
        if isinstance(df.columns, pd.MultiIndex):
            # get_level_values(0) takes only the first level
            # ("Close", "RELIANCE.NS") → "Close"
            df.columns = df.columns.get_level_values(0)

        # ── Fix Problem 2: Extra columns ──────────────────────────────────────
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]

        # Check all 5 columns actually exist
        missing = [c for c in ohlcv_cols if c not in df.columns]
        if missing:
            logger.error(
                f"{symbol}: Missing expected columns: {missing}. "
                f"Got: {list(df.columns)}"
            )
            return None

        # Keep only these 5 columns (drops Dividends, Stock Splits, etc.)
        # .copy() prevents SettingWithCopyWarning later
        df = df[ohlcv_cols].copy()

        # ── Fix Problem 3: Lowercase column names ─────────────────────────────
        df.columns = ["open", "high", "low", "close", "volume"]

        # ── Remove completely empty rows ──────────────────────────────────────
        # how="all" = only drop rows where EVERY column is NaN
        # (partial NaN rows are handled in validation)
        df.dropna(how="all", inplace=True)

        # ── Sort by date oldest first ─────────────────────────────────────────
        # ML models expect chronological order
        # LSTM specifically needs oldest → newest
        df.sort_index(inplace=True)

        # ── Fix Problem 4: Timezone-aware index ───────────────────────────────
        # Check if index has timezone info
        if hasattr(df.index, "tz") and df.index.tz is not None:
            # tz_localize(None) strips timezone, keeps the date/time values
            df.index = df.index.tz_localize(None)

        # ── Fix Problem 5: Float volume → integer ─────────────────────────────
        # fillna(0) replaces any NaN volumes with 0 first
        # then astype(int) converts float → int
        df["volume"] = df["volume"].fillna(0).astype(int)

        # ── Name the index ────────────────────────────────────────────────────
        df.index.name = "date"

        return df

    except Exception as e:
        logger.error(f"Failed to clean {symbol} DataFrame: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPER: _validate_data
# Checks cleaned data is actually usable before returning to caller.
# Returns True = data is good, False = data has problems.
# ─────────────────────────────────────────────────────────────────────────────

def _validate_data(df: pd.DataFrame, symbol: str) -> bool:
    """
    Runs quality checks on cleaned data.

    Check 1 — Enough rows:
        LSTM needs a minimum sequence length (60 days) plus training data.
        If we only have 100 rows, there's barely enough for 1 LSTM sequence.
        Minimum 200 rows = ~10 months of data = safe lower bound.

    Check 2 — No negative prices:
        Prices physically can't be negative.
        Negative values = data corruption or adjustment error.

    Check 3 — High >= Low:
        By definition, the day's high must be >= the day's low.
        Violations = data error (warn but don't reject — adjusted prices
        can occasionally have tiny violations due to rounding).

    Check 4 — NaN percentage:
        Some NaN is okay (first rows of moving averages will be NaN).
        More than 5% NaN in price columns = too much missing data.

    Check 5 — Zero close prices:
        Close price of 0 means no trading happened or data error.
        Worth warning about even if not rejecting.
    """

    price_cols = ["open", "high", "low", "close"]

    # ── Check 1: Minimum rows ─────────────────────────────────────────────────
    if len(df) < MIN_ROWS_REQUIRED:
        logger.warning(
            f"{symbol}: Only {len(df)} rows. "
            f"Need at least {MIN_ROWS_REQUIRED} for ML training. Skipping."
        )
        return False

    # ── Check 2: No negative prices ───────────────────────────────────────────
    for col in price_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            logger.error(
                f"{symbol}: Found {neg_count} negative values in '{col}'. "
                f"Data is corrupted."
            )
            return False

    # ── Check 3: High >= Low ──────────────────────────────────────────────────
    bad_candles = (df["high"] < df["low"]).sum()
    if bad_candles > 0:
        # Warning only — adjusted prices occasionally have this due to rounding
        logger.warning(
            f"{symbol}: {bad_candles} rows where high < low. "
            f"Possibly from price adjustment. Proceeding anyway."
        )

    # ── Check 4: NaN percentage in price columns ──────────────────────────────
    # .isna().mean() gives fraction of NaN values per column
    # .mean() again averages across all price columns
    nan_pct = df[price_cols].isna().mean().mean()

    if nan_pct > 0.05:
        logger.error(
            f"{symbol}: {nan_pct:.1%} of price values are NaN. "
            f"Too much missing data for reliable ML training."
        )
        return False

    elif nan_pct > 0:
        logger.warning(
            f"{symbol}: {nan_pct:.1%} NaN values present. "
            f"Acceptable — will be handled in preprocessing."
        )

    # ── Check 5: Zero close prices ────────────────────────────────────────────
    zero_closes = (df["close"] == 0).sum()
    if zero_closes > 0:
        logger.warning(f"{symbol}: {zero_closes} rows with close price = 0.")

    # ── All checks passed ─────────────────────────────────────────────────────
    logger.info(
        f"{symbol}: Validation passed | "
        f"{len(df)} rows | "
        f"NaN: {nan_pct:.1%} | "
        f"₹{df['close'].min():.0f} – ₹{df['close'].max():.0f}"
    )
    return True