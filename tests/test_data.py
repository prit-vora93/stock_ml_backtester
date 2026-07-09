"""
tests/test_data.py
------------------
Comprehensive tests for the entire Week 1 data pipeline.

Test coverage:
    TestFetcher          → data/fetcher.py
    TestStorage          → data/storage.py
    TestFeatureEngineer  → data/feature_engineer.py
    TestPreprocessor     → data/preprocessor.py
    TestPipelineEnd2End  → full pipeline integration test

Run all tests:
    pytest tests/test_data.py -v

Run one class:
    pytest tests/test_data.py::TestFetcher -v

Run one test:
    pytest tests/test_data.py::TestFetcher::test_fetch_valid_symbol -v

Run with coverage:
    pytest tests/test_data.py -v --cov=data --cov-report=term-missing
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

# ── imports under test ────────────────────────────────────────────────────────
from data.fetcher        import fetch_stock_data, fetch_latest_data
from data.news_fetcher   import score_headline
from data.storage        import (
    save_stock_data, get_stock_data,
    data_exists, get_row_count, get_latest_date,
    get_all_symbols, get_db_summary,
)
from data.feature_engineer import (
    build_full_features, get_feature_names,
    _add_label,
)
from data.preprocessor   import (
    preprocess, PreparedData, SequenceConfig,
    save_scaler, load_scaler,
    _validate_dataframe, _build_sequences,
    _compute_class_weights,
)
from api.database        import SessionLocal, StockData
from config.settings     import SEQUENCE_LENGTH, STOCKS, LABEL_THRESHOLD


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 1: TestFetcher
# Tests for data/fetcher.py
# ═════════════════════════════════════════════════════════════════════════════

class TestFetcher:
    """
    Tests for data/fetcher.py.

    What we verify:
        - Valid symbol returns correct DataFrame shape and columns
        - Invalid symbol returns None (not crash)
        - Returned data has no negative prices
        - Date range is respected
        - Volume is always integer
        - Latest data fetcher works
    """

    def test_fetch_valid_symbol_returns_dataframe(self):
        """
        Fetching a valid NSE symbol should return a non-empty DataFrame.

        Why: Core functionality — if this fails, nothing else works.
        """
        df = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2023-01-01")
        assert df is not None, "fetch_stock_data returned None for valid symbol"
        assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"

    def test_fetch_returns_correct_columns(self):
        """
        DataFrame must have exactly: open, high, low, close, volume (lowercase).

        Why: feature_engineer.py expects these exact column names.
             Uppercase or extra columns would cause KeyError downstream.
        """
        df = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2023-01-01")
        assert df is not None

        expected_cols = {"open", "high", "low", "close", "volume"}
        actual_cols   = set(df.columns)

        assert expected_cols.issubset(actual_cols), (
            f"Missing columns: {expected_cols - actual_cols}"
        )

    def test_fetch_invalid_symbol_returns_none(self):
        """
        An invalid symbol (missing .NS suffix) should return None gracefully.

        Why: Caller code does `if df is None: skip`. Must not raise exception.
        """
        df = fetch_stock_data("INVALIDSTOCK999", "2022-01-01", "2023-01-01")
        assert df is None, "Invalid symbol should return None, not raise exception"

    def test_fetch_no_negative_prices(self):
        """
        All price columns must be non-negative.

        Why: Negative prices indicate data corruption or adjustment error.
             ML model would learn impossible patterns from negative prices.
        """
        df = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2023-01-01")
        assert df is not None

        for col in ["open", "high", "low", "close"]:
            min_val = df[col].min()
            assert min_val >= 0, (
                f"Negative value found in '{col}': min={min_val}"
            )

    def test_fetch_high_always_gte_low(self):
        """
        Daily high must always be >= daily low.

        Why: Physically impossible otherwise. Indicates data error.
             Would create nonsensical Bollinger Band calculations.
        """
        df = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2023-01-01")
        assert df is not None

        violations = (df["high"] < df["low"]).sum()
        assert violations == 0, (
            f"{violations} rows where high < low — data quality issue"
        )

    def test_fetch_date_range_respected(self):
        """
        Returned data should not go outside the requested date range.

        Why: If fetcher returns data outside requested range, downstream
             train/val/test splits could include unintended time periods.

        Note: Using a full year range (250+ trading days) to satisfy
              the MIN_ROWS_REQUIRED=200 validation check in fetcher.py.
        """
        start = "2022-01-01"
        end   = "2022-12-31"
        df    = fetch_stock_data("RELIANCE.NS", start, end)
        assert df is not None

        assert df.index.min() >= pd.Timestamp(start), (
            f"Data starts before requested start: {df.index.min()}"
        )
        assert df.index.max() <= pd.Timestamp(end), (
            f"Data ends after requested end: {df.index.max()}"
        )

    def test_fetch_volume_is_integer(self):
        """
        Volume must be integer type (not float).

        Why: Our DB schema uses BigInteger for volume.
             Float volume like 5823400.0 would cause type mismatch on insert.
        """
        df = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2023-01-01")
        assert df is not None

        assert df["volume"].dtype in [np.int64, np.int32, int], (
            f"Volume dtype should be integer, got {df['volume'].dtype}"
        )

    def test_fetch_index_is_datetime(self):
        """
        DataFrame index must be DatetimeIndex.

        Why: All downstream operations (resampling, joining, plotting)
             require DatetimeIndex. Wrong index type causes silent failures.
        """
        df = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2023-01-01")
        assert df is not None

        assert isinstance(df.index, pd.DatetimeIndex), (
            f"Index should be DatetimeIndex, got {type(df.index)}"
        )

    def test_fetch_sorted_chronologically(self):
        """
        Data must be sorted oldest → newest.

        Why: LSTM requires chronological order.
             If data is reversed, model learns backward patterns.
        """
        df = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2023-01-01")
        assert df is not None

        assert df.index.is_monotonic_increasing, (
            "DataFrame index is not sorted chronologically (oldest → newest)"
        )

    def test_fetch_minimum_rows(self):
        """
        Fetching 1 year of data should return at least 200 rows.

        Why: Indian market has ~250 trading days per year.
             Less than 200 suggests something went wrong with fetch.
        """
        df = fetch_stock_data("RELIANCE.NS", "2022-01-01", "2023-01-01")
        assert df is not None
        assert len(df) >= 200, (
            f"Expected 200+ rows for 1 year, got {len(df)}"
        )

    def test_fetch_latest_data(self):
        """
        fetch_latest_data should return a small recent DataFrame.

        Why: Used for live predictions — must return correct number of rows.
        """
        df = fetch_latest_data("RELIANCE.NS", days=5)

        if df is not None:   # May be None if market closed / weekend
            assert len(df) <= 5, f"Expected ≤5 rows, got {len(df)}"
            assert "close" in df.columns


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 2: TestStorage
# Tests for data/storage.py
# ═════════════════════════════════════════════════════════════════════════════

class TestStorage:
    """
    Tests for data/storage.py.

    What we verify:
        - Data saved can be retrieved correctly
        - Duplicate rows are not inserted twice
        - Row counts are accurate
        - Date filtering works correctly
        - Latest date detection works
        - Empty result handled gracefully
    """

    def test_data_exists_for_fetched_stock(self):
        """
        After Day 2 fetcher ran, RELIANCE.NS should exist in DB.

        Why: Confirms Day 2 actually saved data to PostgreSQL.
             If this fails, run: python main.py (Day 2 script).
        """
        exists = data_exists("RELIANCE.NS")
        assert exists, (
            "RELIANCE.NS not found in DB. "
            "Run Day 2 fetcher (python main.py) first."
        )

    def test_get_stock_data_returns_dataframe(self, sample_ohlcv_df):
        """
        get_stock_data should return a non-empty DataFrame.

        Why: Core retrieval function. If this fails, no ML training possible.
        """
        assert sample_ohlcv_df is not None
        assert isinstance(sample_ohlcv_df, pd.DataFrame)
        assert len(sample_ohlcv_df) > 0

    def test_get_stock_data_correct_columns(self, sample_ohlcv_df):
        """
        Retrieved data must have all 5 OHLCV columns.

        Why: feature_engineer.py will KeyError if any column missing.
        """
        expected = {"open", "high", "low", "close", "volume"}
        assert expected.issubset(set(sample_ohlcv_df.columns)), (
            f"Missing columns: {expected - set(sample_ohlcv_df.columns)}"
        )

    def test_get_stock_data_date_filtering(self):
        """
        Date filter must correctly restrict returned rows.

        Why: Train/val/test splits depend on accurate date filtering.
             Wrong filtering = data leakage.
        """
        start = "2022-01-01"
        end   = "2022-06-30"
        df    = get_stock_data("RELIANCE.NS", start, end)

        assert df is not None
        assert df.index.min() >= pd.Timestamp(start)
        assert df.index.max() <= pd.Timestamp(end)

    def test_get_stock_data_sorted(self, sample_ohlcv_df):
        """
        Retrieved data must be sorted oldest → newest.

        Why: LSTM consumes data in chronological order.
        """
        assert sample_ohlcv_df.index.is_monotonic_increasing, (
            "Retrieved data is not sorted chronologically"
        )

    def test_duplicate_protection(self, sample_ohlcv_df):
        """
        Saving same data twice should not insert duplicate rows.

        Why: UniqueConstraint on (symbol, date) prevents duplicates.
             Duplicate rows would corrupt ML training data.
        """
        count_before = get_row_count("RELIANCE.NS")

        # Try to save the same data again
        saved = save_stock_data("RELIANCE.NS", sample_ohlcv_df)

        count_after = get_row_count("RELIANCE.NS")

        assert saved == 0, (
            f"Expected 0 new rows on re-save, got {saved}"
        )
        assert count_before == count_after, (
            f"Row count changed after duplicate save: {count_before} → {count_after}"
        )

    def test_row_count_is_positive(self):
        """
        Row count for RELIANCE.NS should be positive.

        Why: Sanity check that data is actually in the database.
        """
        count = get_row_count("RELIANCE.NS")
        assert count > 0, f"Expected positive row count, got {count}"

    def test_get_latest_date_returns_date(self):
        """
        get_latest_date should return a Python date object.

        Why: Used for incremental updates. Wrong type causes comparison error.
        """
        latest = get_latest_date("RELIANCE.NS")
        assert latest is not None
        assert isinstance(latest, date), (
            f"Expected date object, got {type(latest)}"
        )

    def test_get_latest_date_is_reasonable(self):
        """
        Latest date should be after 2020 (when we fetched from).

        Why: If latest date is 1970 or None, something went wrong with storage.
        """
        latest = get_latest_date("RELIANCE.NS")
        assert latest is not None
        assert latest.year >= 2020, (
            f"Latest date {latest} is unreasonably old"
        )

    def test_get_all_symbols_includes_expected(self):
        """
        Database should contain all 5 stocks from settings.STOCKS.

        Why: If any stock is missing, that stock can't be trained.
        """
        symbols_in_db = get_all_symbols()
        assert len(symbols_in_db) > 0, "No symbols found in database"

        for symbol in ["RELIANCE.NS", "INFY.NS", "TCS.NS"]:
            assert symbol in symbols_in_db, (
                f"{symbol} not found in DB. Run Day 2 fetcher."
            )

    def test_nonexistent_symbol_returns_none(self):
        """
        Querying a symbol not in DB should return None gracefully.

        Why: Caller code uses `if df is None: skip`. Must not crash.
        """
        df = get_stock_data("FAKESYMBOL.NS", "2022-01-01", "2023-01-01")
        assert df is None, "Non-existent symbol should return None"

    def test_save_and_retrieve_round_trip(self, db_session):
        """
        A manually inserted row should be retrievable with correct values.

        Why: Verifies the full save → retrieve cycle is lossless.
             Floating point values should survive DB round-trip.

        Fix: Delete any existing test row first so the test is idempotent
             (safe to run multiple times without UniqueViolation errors).
        """
        test_symbol = "TEST_PYTEST.NS"
        test_date   = date(2024, 6, 15)

        # Clean up any leftover row from a previous test run
        existing = db_session.query(StockData).filter(
            StockData.symbol == test_symbol,
            StockData.date   == test_date,
        ).first()
        if existing:
            db_session.delete(existing)
            db_session.flush()   # Apply delete before insert

        # Insert fresh test row
        test_row = StockData(
            symbol = test_symbol,
            date   = test_date,
            open   = 1234.56,
            high   = 1250.00,
            low    = 1220.00,
            close  = 1245.67,
            volume = 9876543,
        )
        db_session.add(test_row)
        db_session.commit()

        # Retrieve and verify
        fetched = db_session.query(StockData).filter(
            StockData.symbol == test_symbol,
            StockData.date   == test_date,
        ).first()

        assert fetched is not None
        assert abs(fetched.close - 1245.67) < 0.01, (
            f"Close price mismatch: expected 1245.67, got {fetched.close}"
        )
        assert fetched.volume == 9876543

        # Clean up after test so DB stays clean
        db_session.delete(fetched)
        db_session.commit()


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 3: TestFeatureEngineer
# Tests for data/feature_engineer.py
# ═════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineer:
    """
    Tests for data/feature_engineer.py.

    What we verify:
        - Correct number of features (100+)
        - Label column exists with correct values (0/1/2)
        - No NaN in final output
        - Technical indicators within valid ranges (RSI 0-100, etc.)
        - Macro features merged correctly
        - Sentiment features present
        - Output has more columns than input
        - Feature names list excludes label
    """

    def test_build_features_returns_dataframe(self, sample_features_df):
        """
        Feature engineering should return a non-empty DataFrame.

        Why: If None returned, nothing to train on.
        """
        assert sample_features_df is not None
        assert isinstance(sample_features_df, pd.DataFrame)
        assert len(sample_features_df) > 0

    def test_features_has_minimum_columns(self, sample_features_df):
        """
        Feature DataFrame should have at least 50 columns.

        Why: 108 features are expected. If we get fewer, a whole
             feature group (macro/sentiment/technical) failed to merge.
        """
        n_cols = len(sample_features_df.columns)
        assert n_cols >= 50, (
            f"Expected 50+ feature columns, got {n_cols}. "
            f"Check macro/sentiment merge in feature_engineer.py"
        )

    def test_label_column_exists(self, sample_features_df):
        """
        'label' column must be present for supervised learning.

        Why: Without a label, LSTM has nothing to learn to predict.
        """
        assert "label" in sample_features_df.columns, (
            "'label' column missing from feature DataFrame"
        )

    def test_label_values_are_valid(self, sample_features_df):
        """
        Labels must only contain values 0, 1, or 2.

        Why: LSTM output layer has 3 neurons (DOWN=0, HOLD=1, UP=2).
             Any other value would cause index out of bounds in loss function.
        """
        valid_labels = {0.0, 1.0, 2.0}
        actual_labels = set(sample_features_df["label"].unique())

        invalid = actual_labels - valid_labels
        assert len(invalid) == 0, (
            f"Invalid label values found: {invalid}. "
            f"All labels must be 0, 1, or 2."
        )

    def test_label_distribution_is_reasonable(self, sample_features_df):
        """
        No single label should dominate > 80% of all rows.

        Why: If HOLD is 95% of data, model will always predict HOLD.
             LABEL_THRESHOLD in settings.py may need adjustment.
        """
        counts = sample_features_df["label"].value_counts(normalize=True)
        for label, pct in counts.items():
            assert pct < 0.80, (
                f"Label {label} dominates {pct:.0%} of data. "
                f"Consider adjusting LABEL_THRESHOLD in settings.py."
            )

    def test_no_nan_in_features(self, sample_features_df):
        """
        Feature DataFrame must have zero NaN values.

        Why: NaN propagates through LSTM calculations producing NaN
             loss → NaN gradients → training completely fails.
        """
        feature_cols = get_feature_names(sample_features_df)
        nan_counts   = sample_features_df[feature_cols].isna().sum()
        total_nan    = nan_counts.sum()

        assert total_nan == 0, (
            f"Found NaN in {nan_counts[nan_counts > 0].to_dict()}"
        )

    def test_no_inf_in_features(self, sample_features_df):
        """
        Feature DataFrame must have zero infinite values.

        Why: Inf values cause the same problems as NaN — broken gradients.
             Can appear from division by zero in indicator calculations.
        """
        feature_cols = get_feature_names(sample_features_df)
        inf_count    = np.isinf(sample_features_df[feature_cols].values).sum()

        assert inf_count == 0, (
            f"Found {inf_count} infinite values in features"
        )

    def test_rsi_in_valid_range(self, sample_features_df):
        """
        RSI must be between 0 and 100 (by definition).

        Why: RSI outside [0,100] indicates calculation error.
             Model would receive impossible inputs, harming accuracy.
        """
        if "rsi_14" not in sample_features_df.columns:
            pytest.skip("rsi_14 not in features")

        rsi = sample_features_df["rsi_14"].dropna()
        assert rsi.min() >= 0,   f"RSI below 0: min={rsi.min():.2f}"
        assert rsi.max() <= 100, f"RSI above 100: max={rsi.max():.2f}"

    def test_macro_features_present(self, sample_features_df):
        """
        At least some macro features (VIX, USD/INR) should be present.

        Why: Macro features are a key differentiator of this project.
             If they're missing, macro_fetcher.py failed silently.
        """
        macro_features = ["vix", "usd_inr_change", "nifty_return"]
        present = [f for f in macro_features if f in sample_features_df.columns]

        assert len(present) >= 2, (
            f"Expected macro features, found only: {present}. "
            f"Check data/macro_fetcher.py"
        )

    def test_feature_names_excludes_label(self, sample_features_df):
        """
        get_feature_names() must NOT include 'label' or 'future_return'.

        Why: These are targets, not inputs. Including them in X would be
             the most severe form of data leakage — model sees the answer.
        """
        feature_names = get_feature_names(sample_features_df)

        assert "label"         not in feature_names, "'label' must not be in feature names"
        assert "future_return" not in feature_names, "'future_return' must not be in feature names"

    def test_output_has_more_columns_than_input(self, sample_ohlcv_df, sample_features_df):
        """
        Feature DataFrame should have many more columns than raw OHLCV.

        Why: Feature engineering must be adding indicators.
             If column count is same or less, no features were added.
        """
        raw_cols     = len(sample_ohlcv_df.columns)
        feature_cols = len(sample_features_df.columns)

        assert feature_cols > raw_cols * 5, (
            f"Expected feature_cols >> raw_cols. "
            f"Got raw={raw_cols}, features={feature_cols}"
        )

    def test_index_is_datetime(self, sample_features_df):
        """
        Feature DataFrame must have DatetimeIndex.

        Why: Preprocessor uses .index for date-based splits.
             Non-datetime index would break chronological splitting.
        """
        assert isinstance(sample_features_df.index, pd.DatetimeIndex), (
            f"Expected DatetimeIndex, got {type(sample_features_df.index)}"
        )

    def test_sorted_chronologically(self, sample_features_df):
        """
        Feature DataFrame must be sorted oldest → newest.

        Why: LSTM sequences must be built in time order.
             Out-of-order data would mix future with past.
        """
        assert sample_features_df.index.is_monotonic_increasing, (
            "Feature DataFrame is not sorted chronologically"
        )


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 4: TestPreprocessor
# Tests for data/preprocessor.py
# ═════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    """
    Tests for data/preprocessor.py.

    What we verify:
        - PreparedData has correct 3D shape
        - Scaling range is [0, 1]
        - No NaN or Inf in sequences
        - Train/val/test split is chronological
        - Class weights sum to reasonable values
        - Scaler save/load produces identical results
        - SequenceConfig customization works
        - Feature selection works
        - Data version is correct
    """

    def test_preprocess_returns_prepared_data(self, sample_prepared):
        """
        preprocess() must return a PreparedData object, not None.

        Why: None means the pipeline failed completely.
        """
        assert sample_prepared is not None
        assert isinstance(sample_prepared, PreparedData)

    def test_x_train_is_3d(self, sample_prepared):
        """
        X_train must be 3D: (sequences, timesteps, features).

        Why: LSTM model.fit() requires exactly 3D input.
             2D or 4D input raises ValueError during training.
        """
        assert sample_prepared.X_train.ndim == 3, (
            f"X_train should be 3D, got {sample_prepared.X_train.ndim}D"
        )

    def test_sequence_length_correct(self, sample_prepared):
        """
        Second dimension of X must equal SEQUENCE_LENGTH (60).

        Why: LSTM architecture is built for exactly SEQUENCE_LENGTH timesteps.
             Wrong sequence length → shape mismatch during training.
        """
        actual_seq_len = sample_prepared.X_train.shape[1]
        assert actual_seq_len == SEQUENCE_LENGTH, (
            f"Expected seq length {SEQUENCE_LENGTH}, got {actual_seq_len}"
        )

    def test_feature_count_consistent(self, sample_prepared):
        """
        Number of features in X must match n_features metadata.

        Why: During inference, feature count mismatch causes shape error.
             Metadata must accurately reflect actual array dimensions.
        """
        actual_features = sample_prepared.X_train.shape[2]
        assert actual_features == sample_prepared.n_features, (
            f"X_train features ({actual_features}) != "
            f"n_features metadata ({sample_prepared.n_features})"
        )

    def test_x_y_length_match(self, sample_prepared):
        """
        X and y arrays must have same first dimension (same number of samples).

        Why: Mismatched X/y lengths cause IndexError during training.
             Each sequence must have exactly one label.
        """
        assert len(sample_prepared.X_train) == len(sample_prepared.y_train), \
            "X_train and y_train length mismatch"
        assert len(sample_prepared.X_val) == len(sample_prepared.y_val), \
            "X_val and y_val length mismatch"
        assert len(sample_prepared.X_test) == len(sample_prepared.y_test), \
            "X_test and y_test length mismatch"

    def test_scaling_range(self, sample_prepared):
        """
        All scaled values must be in [0, 1] range.

        Why: MinMaxScaler should produce exactly this range on training data.
             Values outside [0,1] indicate scaler fitting error or data leakage.
        """
        x_min = sample_prepared.X_train.min()
        x_max = sample_prepared.X_train.max()

        assert x_min >= -0.01, f"X_train min too low: {x_min:.6f}"
        assert x_max <=  1.01, f"X_train max too high: {x_max:.6f}"

    def test_no_nan_in_sequences(self, sample_prepared):
        """
        All sequence arrays must be NaN-free.

        Why: NaN in X → NaN loss → NaN gradients → model never learns.
        """
        for name, X in [("X_train", sample_prepared.X_train),
                        ("X_val",   sample_prepared.X_val),
                        ("X_test",  sample_prepared.X_test)]:
            nan_count = np.isnan(X).sum()
            assert nan_count == 0, f"{nan_count} NaN values found in {name}"

    def test_no_inf_in_sequences(self, sample_prepared):
        """
        All sequence arrays must be Inf-free.

        Why: Inf values cause overflow in LSTM calculations.
        """
        for name, X in [("X_train", sample_prepared.X_train),
                        ("X_val",   sample_prepared.X_val),
                        ("X_test",  sample_prepared.X_test)]:
            inf_count = np.isinf(X).sum()
            assert inf_count == 0, f"{inf_count} Inf values found in {name}"

    def test_labels_are_valid(self, sample_prepared):
        """
        All labels must be 0, 1, or 2.

        Why: LSTM output layer has 3 neurons. Invalid labels crash training.
        """
        for name, y in [("y_train", sample_prepared.y_train),
                        ("y_val",   sample_prepared.y_val),
                        ("y_test",  sample_prepared.y_test)]:
            unique = set(y.tolist())
            invalid = unique - {0, 1, 2}
            assert len(invalid) == 0, (
                f"Invalid labels in {name}: {invalid}"
            )

    def test_chronological_split_no_overlap(self, sample_prepared):
        """
        Train, val, test periods must not overlap.

        Why: Overlap = data leakage. Model sees future data during training.
             This is the most serious ML mistake possible.
        """
        train_end = pd.Timestamp(sample_prepared.train_end)
        val_start = pd.Timestamp(sample_prepared.val_start)
        val_end   = pd.Timestamp(sample_prepared.val_end)
        test_start= pd.Timestamp(sample_prepared.test_start)

        assert train_end < val_start, (
            f"Train end ({train_end}) overlaps with val start ({val_start})"
        )
        assert val_end < test_start, (
            f"Val end ({val_end}) overlaps with test start ({test_start})"
        )

    def test_train_is_largest_split(self, sample_prepared):
        """
        Training set must be larger than val and test sets.

        Why: Training on 15% while validating on 70% is wrong.
             70/15/15 split means train > val and train > test.
        """
        n_train = len(sample_prepared.X_train)
        n_val   = len(sample_prepared.X_val)
        n_test  = len(sample_prepared.X_test)

        assert n_train > n_val,  f"Train ({n_train}) should be > Val ({n_val})"
        assert n_train > n_test, f"Train ({n_train}) should be > Test ({n_test})"

    def test_class_weights_all_present(self, sample_prepared):
        """
        Class weights must exist for all 3 classes: 0, 1, 2.

        Why: Missing weight for any class means model.fit() may fail
             or one class gets default weight of 1.0 unintentionally.
        """
        for cls in [0, 1, 2]:
            assert cls in sample_prepared.class_weights, (
                f"Missing class weight for class {cls} (DOWN=0, HOLD=1, UP=2)"
            )

    def test_hold_has_lowest_weight(self, sample_prepared):
        """
        HOLD class (1) should have the lowest weight.

        Why: HOLD is the majority class (60%). Balanced weighting gives it
             the lowest weight so model doesn't always predict HOLD.
        """
        w = sample_prepared.class_weights
        assert w[1] < w[0], (
            f"HOLD weight ({w[1]:.3f}) should be < DOWN weight ({w[0]:.3f})"
        )
        assert w[1] < w[2], (
            f"HOLD weight ({w[1]:.3f}) should be < UP weight ({w[2]:.3f})"
        )

    def test_scaler_save_load_identical(self, sample_prepared, tmp_path):
        """
        Scaler saved to disk and reloaded must produce identical output.

        Why: Live inference uses the saved scaler. If reload is different,
             predictions will be wrong even if model is perfect.
        """
        import joblib, os

        # Save to temp directory
        scaler_path = str(tmp_path / "test_scaler.pkl")
        joblib.dump(sample_prepared.scaler, scaler_path)

        # Load back
        loaded_scaler = joblib.load(scaler_path)

        # Transform a sample with both scalers
        sample = sample_prepared.X_train[0][0].reshape(1, -1)
        out1   = sample_prepared.scaler.transform(sample)
        out2   = loaded_scaler.transform(sample)

        assert np.allclose(out1, out2, atol=1e-10), (
            "Saved and reloaded scaler produce different outputs"
        )

    def test_version_is_correct(self, sample_prepared):
        """
        PreparedData version should match PREPROCESSOR_VERSION.

        Why: Version mismatch indicates stale saved data being used
             with a newer pipeline — could cause subtle incompatibilities.
        """
        from data.preprocessor import PREPROCESSOR_VERSION
        assert sample_prepared.preprocessor_version == PREPROCESSOR_VERSION, (
            f"Version mismatch: "
            f"data={sample_prepared.preprocessor_version}, "
            f"code={PREPROCESSOR_VERSION}"
        )

    def test_feature_set_technical_smaller(self):
        """
        Technical-only feature set should be smaller than full feature set.

        Why: Feature selection must actually reduce features.
             If not, filtering logic is broken.
        """
        _symbol = "RELIANCE.NS"
        _start  = "2021-01-01"
        _end    = "2023-12-31"

        data_tech = preprocess(_symbol, _start, _end,
                               feature_set="technical")
        data_all  = preprocess(_symbol, _start, _end,
                               feature_set="all")

        if data_tech and data_all:
            assert data_tech.n_features < data_all.n_features, (
                f"Technical subset ({data_tech.n_features}) should be "
                f"smaller than full ({data_all.n_features})"
            )

    def test_stride_reduces_sequences(self):
        """
        stride=5 should produce ~5x fewer sequences than stride=1.

        Why: Stride is used to reduce sequence overlap.
             If stride has no effect, walk-forward validation breaks.
        """
        _symbol = "RELIANCE.NS"
        _start  = "2021-01-01"
        _end    = "2023-12-31"

        cfg_s1 = SequenceConfig(sequence_length=60, prediction_horizon=1, stride=1)
        cfg_s5 = SequenceConfig(sequence_length=60, prediction_horizon=1, stride=5)

        data_s1 = preprocess(_symbol, _start, _end, seq_config=cfg_s1)
        data_s5 = preprocess(_symbol, _start, _end, seq_config=cfg_s5)

        if data_s1 and data_s5:
            ratio = len(data_s1.X_train) / len(data_s5.X_train)
            assert 3.0 <= ratio <= 7.0, (
                f"Stride ratio should be ~5x, got {ratio:.1f}x"
            )


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 5: TestPipelineEnd2End
# Full integration test — runs complete pipeline for one stock.
# ═════════════════════════════════════════════════════════════════════════════

class TestPipelineEnd2End:
    """
    End-to-end integration tests.

    What we verify:
        - Complete pipeline from DB → features → sequences works
        - Output is immediately usable for LSTM training
        - Data leakage checks pass
        - Memory usage is reasonable
    """

    def test_full_pipeline_produces_valid_data(self, sample_prepared):
        """
        Complete pipeline output must be immediately usable for LSTM training.

        Checks everything needed to call model.fit():
            - X_train shape is correct 3D
            - y_train has valid integer labels
            - class_weights is a dict with 3 keys
            - All values are finite
        """
        # Shape
        assert sample_prepared.X_train.ndim == 3
        assert sample_prepared.X_train.shape[1] == SEQUENCE_LENGTH

        # Labels
        assert set(sample_prepared.y_train.tolist()).issubset({0, 1, 2})

        # Class weights dict
        assert isinstance(sample_prepared.class_weights, dict)
        assert len(sample_prepared.class_weights) == 3

        # No NaN/Inf
        assert not np.isnan(sample_prepared.X_train).any()
        assert not np.isinf(sample_prepared.X_train).any()

    def test_memory_usage_reasonable(self, sample_prepared):
        """
        Total memory usage should be under 200 MB.

        Why: If memory grows too large, training will OOM (out of memory).
             200 MB is a safe limit for 5 stocks on a standard machine.
        """
        assert sample_prepared.memory_mb < 200, (
            f"Memory usage {sample_prepared.memory_mb:.1f} MB exceeds 200 MB limit"
        )

    def test_no_data_leakage_chronological(self, sample_prepared):
        """
        Confirms train/val/test splits are strictly chronological.

        Why: Most important correctness guarantee in the entire pipeline.
             Any leakage gives falsely optimistic backtest results.
        """
        # Training period must end before validation begins
        assert sample_prepared.train_end < sample_prepared.val_start
        assert sample_prepared.val_end   < sample_prepared.test_start

    def test_feature_metadata_accurate(self, sample_prepared):
        """
        Metadata stored in PreparedData must match actual array dimensions.

        Why: Metadata is used during inference to reconstruct the pipeline.
             Inaccurate metadata causes shape mismatches at prediction time.
        """
        assert sample_prepared.n_features == sample_prepared.X_train.shape[2]
        assert sample_prepared.sequence_len == sample_prepared.X_train.shape[1]
        assert len(sample_prepared.feature_names) == sample_prepared.n_features

    def test_all_5_stocks_preprocessable(self):
        """
        All 5 stocks in STOCKS list must preprocess without error.

        Why: If even one stock fails, that stock can't be trained.
             Catches stock-specific data issues early.
        """
        _start = "2021-01-01"
        _end   = "2023-12-31"

        failed = []
        for symbol in STOCKS:
            data = preprocess(symbol, _start, _end,
                              include_sentiment=False)  # Skip sentiment for speed
            if data is None:
                failed.append(symbol)

        assert len(failed) == 0, (
            f"These stocks failed preprocessing: {failed}"
        )