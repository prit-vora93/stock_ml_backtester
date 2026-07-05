"""
data/storage.py
---------------
Saves and retrieves stock OHLCV data from PostgreSQL.

Responsibilities:
  - Save fetched DataFrames into the stock_data table
  - Read data back as DataFrames for ML training / backtesting
  - Skip duplicate rows (same symbol + date already in DB)
  - Provide utility functions (exists check, row count, latest date)

What this file does NOT do:
  - Fetch from internet (that's fetcher.py)
  - Calculate indicators (that's feature_engineer.py)
  - Train models (that's models/)

Usage:
    from data.storage import save_stock_data, get_stock_data

    # Save a fetched DataFrame to DB
    rows_saved = save_stock_data("RELIANCE.NS", df)

    # Read it back later (even after restart)
    df = get_stock_data("RELIANCE.NS", "2022-01-01", "2024-01-01")
"""

from datetime import date, datetime

import pandas as pd

from api.database import SessionLocal, StockData
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: save_stock_data
# Takes a cleaned DataFrame from fetcher.py and writes it to PostgreSQL.
# Skips rows that already exist (safe to call multiple times).
# ─────────────────────────────────────────────────────────────────────────────

def save_stock_data(symbol: str, df: pd.DataFrame) -> int:
    """
    Saves OHLCV DataFrame into the stock_data table in PostgreSQL.
    Rows that already exist (same symbol + date) are silently skipped.

    Args:
        symbol: Stock symbol e.g. "RELIANCE.NS"
        df:     Cleaned DataFrame from fetcher.py
                Must have columns: open, high, low, close, volume
                Must have DatetimeIndex

    Returns:
        Number of NEW rows actually inserted
        (existing rows are skipped and not counted)

    Example:
        # First time — saves everything
        saved = save_stock_data("RELIANCE.NS", df)
        print(saved)   # 1250

        # Second time — skips everything (already in DB)
        saved = save_stock_data("RELIANCE.NS", df)
        print(saved)   # 0
    """

    db         = SessionLocal()   # Open database session
    rows_saved = 0
    rows_skip  = 0

    try:
        logger.info(f"Saving {len(df)} rows for {symbol} to database...")

        # ── Loop through every row in the DataFrame ───────────────────────────
        # Each row = one trading day
        # We insert them one by one so we can check duplicates individually

        for row_date, row in df.iterrows():

            # ── Convert date to Python date object ────────────────────────────
            # df.iterrows() gives us a pandas Timestamp as the index
            # PostgreSQL Date column needs a plain Python date object
            # .date() extracts just the date part (drops time component)
            #
            # Example:
            #   pandas Timestamp: 2022-01-03 00:00:00
            #   Python date:      2022-01-03   ← this is what DB wants

            if hasattr(row_date, "date"):
                stock_date = row_date.date()
            else:
                stock_date = row_date

            # ── Duplicate check ───────────────────────────────────────────────
            # Before inserting, check if this symbol + date already exists.
            # Our DB has a UniqueConstraint on (symbol, date) which would
            # raise an error if we try to insert a duplicate.
            # Checking first is cleaner than catching that error.
            #
            # .first() returns the row if found, or None if not found.

            already_exists = db.query(StockData).filter(
                StockData.symbol == symbol,
                StockData.date   == stock_date,
            ).first()

            if already_exists:
                rows_skip += 1
                continue    # Skip this row, move to next date

            # ── Create new database row ───────────────────────────────────────
            # StockData() creates a Python object representing one DB row.
            # We explicitly cast types to avoid DB type mismatch errors:
            #   float() for prices (yfinance sometimes returns numpy float32)
            #   int()   for volume (yfinance sometimes returns numpy int64)

            new_row = StockData(
                symbol = symbol,
                date   = stock_date,
                open   = float(row["open"]),
                high   = float(row["high"]),
                low    = float(row["low"]),
                close  = float(row["close"]),
                volume = int(row["volume"]),
            )

            # db.add() stages the row (not saved yet)
            db.add(new_row)
            rows_saved += 1

        # ── Commit all staged rows in ONE transaction ─────────────────────────
        # db.commit() is what actually writes to PostgreSQL.
        # Doing ONE commit at the end (not after each row) is much faster.
        #
        # Why? Each commit is a disk write operation.
        # 1250 individual commits = 1250 disk writes = slow
        # 1 commit at end = 1 disk write = fast
        #
        # If anything fails before commit(), nothing is saved (all or nothing).
        # This is called a "transaction" — data integrity guaranteed.

        db.commit()

        logger.success(
            f"{symbol}: {rows_saved} new rows saved | "
            f"{rows_skip} duplicate rows skipped"
        )
        return rows_saved

    except Exception as e:
        # ── Rollback on error ─────────────────────────────────────────────────
        # If anything went wrong, undo ALL staged changes.
        # Prevents partial saves (e.g. only 600 of 1250 rows saved).
        db.rollback()
        logger.error(f"Failed to save {symbol}: {e}")
        return 0

    finally:
        # ── Always close the session ──────────────────────────────────────────
        # finally block runs whether or not an exception occurred.
        # Closing releases the DB connection back to the pool.
        # Not closing = connection leak = app eventually runs out of connections.
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: get_stock_data
# Reads saved OHLCV data from PostgreSQL and returns it as a DataFrame.
# Used by feature_engineer.py, ML models, and backtesting engine.
# ─────────────────────────────────────────────────────────────────────────────

def get_stock_data(
    symbol:     str,
    start_date: str | None = None,
    end_date:   str | None = None,
) -> pd.DataFrame | None:
    """
    Retrieves OHLCV data from PostgreSQL for a given stock and date range.

    Args:
        symbol:     Stock symbol e.g. "RELIANCE.NS"
        start_date: Optional start date "YYYY-MM-DD"
                    If None, returns from the earliest date available
        end_date:   Optional end date "YYYY-MM-DD"
                    If None, returns up to the latest date available

    Returns:
        pd.DataFrame with:
            - Index: DatetimeIndex named "date"
            - Columns: open, high, low, close, volume
            - Sorted oldest → newest
        None if no data found for this symbol/range

    Example:
        df = get_stock_data("RELIANCE.NS", "2022-01-01", "2024-01-01")

        print(df.shape)    # (496, 5)
        print(df.head(2))
        #               open     high      low    close    volume
        # date
        # 2022-01-03  2394.0  2406.5  2372.5  2389.0  5823400
        # 2022-01-04  2392.0  2414.0  2380.0  2401.5  6102200
    """

    db = SessionLocal()

    try:
        # ── Build the database query ──────────────────────────────────────────
        # Start with all rows for this symbol
        query = db.query(StockData).filter(
            StockData.symbol == symbol
        )

        # ── Apply date filters if provided ────────────────────────────────────
        # strptime() converts string "2022-01-01" → Python date object
        # We need date objects to compare with DB Date column

        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            query = query.filter(StockData.date >= start)

        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
            query = query.filter(StockData.date <= end)

        # ── Execute query sorted by date ──────────────────────────────────────
        # .asc() = ascending = oldest first
        # ML models and backtesting REQUIRE chronological order
        rows = query.order_by(StockData.date.asc()).all()

        # ── Handle empty result ───────────────────────────────────────────────
        if not rows:
            logger.warning(
                f"No data found for {symbol} "
                f"({start_date or 'beginning'} → {end_date or 'today'}). "
                f"Have you run the fetcher yet?"
            )
            return None

        # ── Convert DB rows → Python list of dicts → DataFrame ───────────────
        # Each 'row' is a StockData object with attributes: date, open, high...
        # We extract them into a list of dictionaries, then make a DataFrame.
        # This is the standard pattern for SQLAlchemy → pandas conversion.

        data = [
            {
                "date":   row.date,
                "open":   row.open,
                "high":   row.high,
                "low":    row.low,
                "close":  row.close,
                "volume": row.volume,
            }
            for row in rows      # List comprehension — one dict per DB row
        ]

        df = pd.DataFrame(data)

        # ── Set date as index ─────────────────────────────────────────────────
        # Convert date column to datetime type (needed for time series operations)
        # Then set it as the index (standard format for financial data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        logger.info(
            f"Retrieved {len(df)} rows for {symbol} | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
        return df

    except Exception as e:
        logger.error(f"Failed to retrieve data for {symbol}: {e}")
        return None

    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: save_all_stocks
# Convenience function — fetch + save all stocks in one call.
# Used in the daily update script.
# ─────────────────────────────────────────────────────────────────────────────

def save_all_stocks(stock_data_dict: dict) -> dict:
    """
    Saves multiple stocks to the database in one call.

    Args:
        stock_data_dict: Dictionary from fetch_all_stocks()
                         { "RELIANCE.NS": df, "INFY.NS": df, ... }

    Returns:
        Dictionary showing how many rows were saved per stock
        { "RELIANCE.NS": 1250, "INFY.NS": 1248, ... }

    Example:
        from data.fetcher import fetch_all_stocks
        from data.storage import save_all_stocks

        all_data   = fetch_all_stocks()
        save_result = save_all_stocks(all_data)
        print(save_result)
        # {"RELIANCE.NS": 1250, "INFY.NS": 1248, "TCS.NS": 1247, ...}
    """

    results = {}

    for symbol, df in stock_data_dict.items():
        rows_saved       = save_stock_data(symbol, df)
        results[symbol]  = rows_saved

    total = sum(results.values())
    logger.success(f"Saved total {total} new rows across {len(results)} stocks")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# Small helpers used by other parts of the system.
# ─────────────────────────────────────────────────────────────────────────────

def data_exists(symbol: str) -> bool:
    """
    Checks whether ANY data exists in the DB for a given symbol.

    Used before fetching to decide:
      - True  → data already in DB, maybe just update recent rows
      - False → no data at all, need full historical fetch

    Example:
        if not data_exists("RELIANCE.NS"):
            df = fetch_stock_data("RELIANCE.NS", "2020-01-01", "2024-01-01")
            save_stock_data("RELIANCE.NS", df)
        else:
            print("Already in DB, skipping full fetch")
    """
    db = SessionLocal()
    try:
        count = db.query(StockData).filter(
            StockData.symbol == symbol
        ).count()
        return count > 0
    finally:
        db.close()


def get_row_count(symbol: str) -> int:
    """
    Returns total number of rows in DB for a given symbol.

    Example:
        count = get_row_count("RELIANCE.NS")
        print(count)   # 1250
    """
    db = SessionLocal()
    try:
        return db.query(StockData).filter(
            StockData.symbol == symbol
        ).count()
    finally:
        db.close()


def get_latest_date(symbol: str) -> date | None:
    """
    Returns the most recent date stored in DB for a given symbol.

    Used for incremental updates:
      - Find latest date in DB (e.g. 2024-01-10)
      - Fetch only from 2024-01-11 onwards
      - No need to re-fetch years of existing data

    Example:
        latest = get_latest_date("RELIANCE.NS")
        print(latest)   # datetime.date(2024, 1, 10)

        # Now fetch only new data
        new_df = fetch_stock_data("RELIANCE.NS",
                                   str(latest), 
                                   datetime.today().strftime("%Y-%m-%d"))
    """
    db = SessionLocal()
    try:
        row = db.query(StockData).filter(
            StockData.symbol == symbol
        ).order_by(StockData.date.desc()).first()

        return row.date if row else None
    finally:
        db.close()


def get_all_symbols() -> list[str]:
    """
    Returns list of all unique stock symbols currently in the database.

    Example:
        symbols = get_all_symbols()
        print(symbols)
        # ['HDFCBANK.NS', 'INFY.NS', 'RELIANCE.NS', 'TCS.NS', 'WIPRO.NS']
    """
    db = SessionLocal()
    try:
        rows = db.query(StockData.symbol).distinct().all()
        return [row[0] for row in rows]
    finally:
        db.close()


def get_db_summary() -> None:
    """
    Prints a summary of all data currently in the database.
    Useful for quickly checking what's stored.

    Output example:
        ── Database Summary ──────────────────────────
          RELIANCE.NS   │ 1250 rows │ 2019-01-02 → 2024-01-15
          INFY.NS       │ 1248 rows │ 2019-01-02 → 2024-01-15
          TCS.NS        │ 1247 rows │ 2019-01-03 → 2024-01-15
          WIPRO.NS      │ 1251 rows │ 2019-01-02 → 2024-01-15
          HDFCBANK.NS   │ 1249 rows │ 2019-01-02 → 2024-01-15
        ── Total: 6245 rows across 5 stocks ──────────
    """
    db = SessionLocal()
    try:
        symbols = [row[0] for row in db.query(StockData.symbol).distinct().all()]

        if not symbols:
            logger.warning("Database is empty — no stock data found")
            return

        logger.info("── Database Summary " + "─" * 30)
        total_rows = 0

        for symbol in sorted(symbols):
            count    = db.query(StockData).filter(StockData.symbol == symbol).count()
            earliest = db.query(StockData).filter(
                StockData.symbol == symbol
            ).order_by(StockData.date.asc()).first()

            latest   = db.query(StockData).filter(
                StockData.symbol == symbol
            ).order_by(StockData.date.desc()).first()

            logger.info(
                f"  {symbol:<15} │ "
                f"{count:>4} rows │ "
                f"{earliest.date} → {latest.date}"
            )
            total_rows += count

        logger.info(f"── Total: {total_rows} rows across {len(symbols)} stocks " + "─" * 10)

    finally:
        db.close()