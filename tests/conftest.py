"""
tests/conftest.py
-----------------
Shared pytest fixtures available to ALL test files automatically.
No import needed — pytest discovers these automatically.

Fixtures defined here:
    sample_ohlcv_df     → small raw OHLCV DataFrame for testing
    sample_features_df  → feature-engineered DataFrame
    sample_prepared     → fully preprocessed PreparedData object
    db_session          → database session for DB tests
    valid_symbol        → stock symbol known to be in DB

Design:
    Fixtures use scope="module" where possible so expensive operations
    (like fetching from DB or building features) run ONCE per test file
    rather than once per test function. This keeps tests fast.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime

from api.database      import SessionLocal, StockData, create_tables
from data.storage      import get_stock_data
from data.feature_engineer import build_full_features
from data.preprocessor import preprocess, PreparedData


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS used across tests
# ─────────────────────────────────────────────────────────────────────────────

# Use a short date range for tests — fast, but enough data
TEST_SYMBOL     = "RELIANCE.NS"
TEST_START      = "2021-01-01"
TEST_END        = "2023-12-31"
MIN_ROWS        = 100     # Minimum acceptable rows
MIN_FEATURES    = 50      # Minimum acceptable feature columns
SEQUENCE_LENGTH = 60      # Must match settings.py


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: db_session
# Provides a database session. Rolls back after each test.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def db_session():
    """
    Yields a live database session for each test.
    Rolls back any changes after the test completes.

    Why rollback?
        Tests should not leave data behind in the database.
        Each test starts with a clean slate.

    scope="function" means a fresh session per test function.
    """
    create_tables()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: sample_ohlcv_df
# Raw OHLCV DataFrame from PostgreSQL.
# scope="module" = fetched once per test file, reused across tests.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_ohlcv_df():
    """
    Provides a real OHLCV DataFrame from the database.

    Requires RELIANCE.NS data to be in PostgreSQL.
    Run Day 2 fetcher before running tests.

    Returns:
        pd.DataFrame with columns: open, high, low, close, volume
        DatetimeIndex sorted oldest → newest
    """
    df = get_stock_data(TEST_SYMBOL, TEST_START, TEST_END)
    if df is None or df.empty:
        pytest.skip(
            f"No data for {TEST_SYMBOL} in DB. "
            f"Run python main.py (Day 2) first."
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: sample_features_df
# Feature-engineered DataFrame with all 108 columns.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_features_df():
    """
    Provides a feature-engineered DataFrame for RELIANCE.NS.

    Includes: technical indicators + macro + sentiment + label.
    Takes ~15 seconds (fetches macro data from Yahoo Finance).
    scope="module" ensures this runs ONCE per test file.

    Returns:
        pd.DataFrame with 100+ columns including 'label'
    """
    df = build_full_features(TEST_SYMBOL, TEST_START, TEST_END)
    if df is None or df.empty:
        pytest.skip(
            f"Feature engineering failed for {TEST_SYMBOL}. "
            f"Check data/feature_engineer.py."
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: sample_prepared
# Fully preprocessed PreparedData object ready for LSTM.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_prepared():
    """
    Provides a fully preprocessed PreparedData object.

    This is the most expensive fixture (~20 seconds).
    scope="module" ensures it runs ONCE and is reused.

    Returns:
        PreparedData with X_train, X_val, X_test, y_*, class_weights, scaler
    """
    data = preprocess(TEST_SYMBOL, TEST_START, TEST_END)
    if data is None:
        pytest.skip(
            f"Preprocessing failed for {TEST_SYMBOL}. "
            f"Check data/preprocessor.py."
        )
    return data


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: minimal_ohlcv_df
# Tiny manually-constructed DataFrame for unit tests.
# Does NOT require database or internet — fully self-contained.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def minimal_ohlcv_df():
    """
    Provides a small manually-constructed OHLCV DataFrame.

    Used for unit tests that don't need real data.
    Fast — no DB or network required.
    Contains 300 rows (enough for basic indicator calculation).

    Returns:
        pd.DataFrame with columns: open, high, low, close, volume
    """
    np.random.seed(42)
    n = 300

    # Simulate a realistic price series using random walk
    # Starting at ₹2000, moves ±1% per day
    price = 2000.0
    closes = []
    for _ in range(n):
        price *= (1 + np.random.uniform(-0.01, 0.01))
        closes.append(round(price, 2))

    closes = np.array(closes)

    df = pd.DataFrame({
        "open":   closes * np.random.uniform(0.99, 1.00, n),
        "high":   closes * np.random.uniform(1.00, 1.02, n),
        "low":    closes * np.random.uniform(0.98, 1.00, n),
        "close":  closes,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    })

    # Create a business-day DatetimeIndex (Mon-Fri only, like real stock data)
    dates = pd.date_range(start="2022-01-03", periods=n, freq="B")
    df.index = dates
    df.index.name = "date"

    return df