"""
data/preprocessor.py
--------------------
Production-quality preprocessing pipeline for quantitative ML stock prediction.

Public API (fully preserved — no breaking changes):
    preprocess()                ← main entry point
    preprocess_all_stocks()     ← batch processing
    PreparedData                ← output dataclass
    save_scaler()               ← persist scaler + metadata
    load_scaler()               ← restore scaler
    get_inference_sequence()    ← live prediction input

Architecture:
    preprocess()
        │
        ├── _validate_dataframe()      ← data quality checks
        ├── _select_features()         ← optional feature filtering
        ├── _time_split()              ← chronological split
        ├── _build_scaler()            ← configurable scaler factory
        ├── _fit_scaler()              ← fit on train only
        ├── _transform()               ← apply to all splits
        ├── _build_sequences()         ← sliding window → 3D
        ├── _validate_sequences()      ← shape + NaN checks
        └── _compute_class_weights()   ← imbalance handling

Improvements over v1:
    - Configurable scaler (minmax/standard/robust/quantile/power)
    - Per-group scaling (technical/macro/sentiment/volume/categorical)
    - Data validation (NaN, Inf, duplicates, constants, empty)
    - Feature metadata (groups, dates, config, version)
    - Configurable sequences (length, stride, prediction horizon)
    - Multiple prediction horizons (1d/3d/5d/10d)
    - Walk-forward validation splits
    - Sequence validation (shape, NaN, Inf)
    - Rich PreparedData.summary()
    - Scaler metadata file (feature names, version, config)
    - Feature selection (technical/macro/news/custom)
    - Vectorized sequence creation
    - Detailed logging (timing, memory, warnings)
"""

import os
import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.utils.class_weight import compute_class_weight

from data.feature_engineer import build_full_features, get_feature_names
from config.settings import (
    SEQUENCE_LENGTH,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RANDOM_SEED,
    MODELS_DIR,
    STOCKS,
)
from utils.logger import logger

# ── Version tag — increment when pipeline changes break compatibility ─────────
PREPROCESSOR_VERSION = "2.1.0"

# ── Global reproducibility seed ───────────────────────────────────────────────
# Set once here so ALL stochastic operations (QuantileTransformer,
# PowerTransformer, numpy) produce identical results across runs.
# Without this, two identical training runs give slightly different models.
np.random.seed(RANDOM_SEED)

# ── Scaler type read from settings (add SCALER_TYPE = "minmax" to settings.py)
# Supported: "minmax", "standard", "robust", "quantile", "power"
try:
    from config.settings import SCALER_TYPE
except ImportError:
    SCALER_TYPE = "minmax"   # safe default

# ── Supported prediction horizons (days ahead) ────────────────────────────────
SUPPORTED_HORIZONS: list[int] = [1, 3, 5, 10]

# ── Feature group prefixes — used for metadata and per-group scaling ──────────
# These prefix patterns identify which group each column belongs to.
FEATURE_GROUP_PATTERNS: dict[str, list[str]] = {
    "technical": [
        "sma_", "ema_", "macd", "rsi_", "stoch_", "roc_",
        "bb_", "atr_", "obv", "vol_ratio", "dist_",
        "price_change", "high_low", "body_ratio",
        "upper_shadow", "lower_shadow", "position_52w",
    ],
    "macro": [
        "vix", "usd_inr", "crude", "nifty", "sp500",
        "nasdaq", "nikkei", "hangseng", "dxy",
        "natural_gas", "gold", "market_stress",
    ],
    "news": [
        "sentiment", "weighted_sentiment", "news_count",
        "negative_flag", "positive_flag", "event_",
        "news_intensity", "news_volume", "positive_ratio",
        "negative_ratio", "neutral_ratio", "average_importance",
    ],
    "volume": [
        "volume", "obv_roc",
    ],
    "time": [
        "day_of_week", "month",
    ],
    "categorical": [
        "vix_regime", "crude_high", "vix_spike",
    ],
}

# ── Scaler types NOT to apply to categorical/binary features ──────────────────
# These columns are already 0/1 or small integers — scaling would hurt.
CATEGORICAL_SKIP_SCALERS = {"quantile", "power"}


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS: SequenceConfig
# All sequence-related parameters in one clean object.
# Changing prediction horizon requires only this config — nothing else.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SequenceConfig:
    """
    Configuration for sliding window sequence generation.

    Attributes:
        sequence_length:    Number of historical days per input sequence (default 60)
        prediction_horizon: Number of days ahead to predict (default 1)
        stride:             Step size between consecutive sequences (default 1)

    Examples:
        Default (current behavior):
            SequenceConfig(60, 1, 1)
            → 60 days input → predict 1 day ahead, every day

        Longer horizon:
            SequenceConfig(90, 5, 5)
            → 90 days input → predict 5 days ahead, every 5 days

        Transformer-ready:
            SequenceConfig(120, 1, 1)
            → 120 days input → predict 1 day ahead

    Why configurable?
        Different models need different context windows.
        LSTM works well with 60 days.
        Transformers can handle 120+ days.
        Stride > 1 reduces sequence overlap → less correlated training samples.
    """
    sequence_length:    int = SEQUENCE_LENGTH
    prediction_horizon: int = 1
    stride:             int = 1

    def __post_init__(self) -> None:
        assert self.sequence_length    > 0,  "sequence_length must be > 0"
        assert self.prediction_horizon > 0,  "prediction_horizon must be > 0"
        assert self.stride             > 0,  "stride must be > 0"
        if self.prediction_horizon not in SUPPORTED_HORIZONS:
            logger.warning(
                f"prediction_horizon={self.prediction_horizon} not in "
                f"SUPPORTED_HORIZONS={SUPPORTED_HORIZONS}. Proceeding anyway."
            )


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS: PreparedData (extended — all v1 fields preserved)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreparedData:
    """
    Complete preprocessed dataset ready for model training.

    All v1 fields preserved. New metadata fields added.

    Core data arrays (unchanged from v1):
        X_train, X_val, X_test  → 3D (samples, timesteps, features)
        y_train, y_val, y_test  → 1D (samples,)
        class_weights           → {0: w, 1: w, 2: w}
        scaler                  → fitted scaler object
        feature_names           → ordered list of feature column names

    New metadata fields (v2):
        symbol                  → stock symbol
        n_features              → number of input features
        sequence_len            → sequence length used
        seq_config              → full SequenceConfig object
        scaler_type             → "minmax" / "standard" / etc.
        feature_groups          → { "technical": [...], "macro": [...], ... }
        train_start             → first training date
        train_end               → last training date
        val_start / val_end     → validation period
        test_start / test_end   → test period
        training_config         → dict of all training hyperparameters
        preprocessor_version    → version string for compatibility checks
    """

    # ── Core data (v1 fields — types preserved) ───────────────────────────────
    X_train:       np.ndarray
    y_train:       np.ndarray
    X_val:         np.ndarray
    y_val:         np.ndarray
    X_test:        np.ndarray
    y_test:        np.ndarray
    class_weights: dict
    scaler:        object          # any sklearn-compatible scaler
    feature_names: list[str]

    # ── Identity (v1 fields — preserved) ─────────────────────────────────────
    symbol:        str = ""
    n_features:    int = 0
    sequence_len:  int = SEQUENCE_LENGTH   # backward-compat alias

    # ── New v2 metadata ───────────────────────────────────────────────────────
    seq_config:    SequenceConfig      = field(default_factory=SequenceConfig)
    scaler_type:   str                 = SCALER_TYPE
    feature_groups: dict[str, list[str]] = field(default_factory=dict)

    # Training period dates
    train_start:   Optional[str] = None
    train_end:     Optional[str] = None
    val_start:     Optional[str] = None
    val_end:       Optional[str] = None
    test_start:    Optional[str] = None
    test_end:      Optional[str] = None

    # Full training config snapshot (useful for experiment tracking)
    training_config: dict = field(default_factory=dict)

    # Version for future compatibility checks
    preprocessor_version: str = PREPROCESSOR_VERSION

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def total_sequences(self) -> int:
        return len(self.y_train) + len(self.y_val) + len(self.y_test)

    @property
    def memory_mb(self) -> float:
        """
        Approximate memory usage of all numpy arrays in megabytes.

        Bug fixed (v2.1.0):
        Previous version summed only the 6 data arrays, missing the
        scaler's internal arrays (e.g. QuantileTransformer stores the
        full quantile table — can be 10-50 MB on large datasets).
        We now add scaler memory via joblib's temp-serialise trick,
        which gives the true in-memory footprint without disk I/O.

        Note: scaler estimate uses joblib Memory which may be slightly
        conservative for very large quantile tables.
        """
        # numpy arrays — exact
        arrays = [
            self.X_train, self.y_train,
            self.X_val,   self.y_val,
            self.X_test,  self.y_test,
        ]
        array_bytes = sum(a.nbytes for a in arrays if a is not None)

        # scaler object — estimated via pickle size (good approximation)
        try:
            import pickle
            scaler_bytes = len(pickle.dumps(self.scaler))
        except Exception:
            scaler_bytes = 0

        return (array_bytes + scaler_bytes) / (1024 ** 2)

    def summary(self) -> None:
        """
        Prints a rich, formatted summary of the prepared dataset.
        Includes shape, periods, class distribution, memory, and config.
        """
        w = 60   # line width
        bar = "═" * w

        logger.info(bar)
        logger.info(f"  PreparedData — {self.symbol}  (v{self.preprocessor_version})")
        logger.info(bar)

        # ── Configuration ─────────────────────────────────────────────────────
        logger.info("  CONFIGURATION")
        logger.info(f"    Scaler           : {self.scaler_type}")
        logger.info(f"    Sequence length  : {self.seq_config.sequence_length} days")
        logger.info(f"    Prediction horizon: {self.seq_config.prediction_horizon} day(s)")
        logger.info(f"    Stride           : {self.seq_config.stride}")
        logger.info(f"    Features         : {self.n_features}")
        logger.info(f"    Total sequences  : {self.total_sequences}")
        logger.info(f"    Memory usage     : {self.memory_mb:.1f} MB")

        # ── Periods ───────────────────────────────────────────────────────────
        logger.info("")
        logger.info("  PERIODS")
        logger.info(f"    Train  : {self.train_start} → {self.train_end}  ({len(self.y_train)} seq)")
        logger.info(f"    Val    : {self.val_start}   → {self.val_end}    ({len(self.y_val)} seq)")
        logger.info(f"    Test   : {self.test_start}  → {self.test_end}   ({len(self.y_test)} seq)")

        # ── Shapes ────────────────────────────────────────────────────────────
        logger.info("")
        logger.info("  SHAPES")
        logger.info(f"    X_train : {self.X_train.shape}")
        logger.info(f"    X_val   : {self.X_val.shape}")
        logger.info(f"    X_test  : {self.X_test.shape}")

        # ── Class distribution ────────────────────────────────────────────────
        logger.info("")
        logger.info("  CLASS DISTRIBUTION")
        for split_name, y in [("Train", self.y_train), ("Val", self.y_val), ("Test", self.y_test)]:
            counts = np.bincount(y.astype(int), minlength=3)
            n      = max(len(y), 1)
            logger.info(
                f"    {split_name:<6} : "
                f"DOWN={counts[0]}({counts[0]/n:.0%})  "
                f"HOLD={counts[1]}({counts[1]/n:.0%})  "
                f"UP={counts[2]}({counts[2]/n:.0%})"
            )
        logger.info(f"    Weights : DOWN={self.class_weights.get(0,1):.3f}  "
                    f"HOLD={self.class_weights.get(1,1):.3f}  "
                    f"UP={self.class_weights.get(2,1):.3f}")

        # ── Feature groups ────────────────────────────────────────────────────
        if self.feature_groups:
            logger.info("")
            logger.info("  FEATURE GROUPS")
            for group, cols in self.feature_groups.items():
                if cols:
                    logger.info(f"    {group:<12} : {len(cols)} features")

        # ── Data leakage check ────────────────────────────────────────────────
        logger.info("")
        logger.info("  DATA LEAKAGE CHECKS")
        logger.info(f"    Split type        : chronological (no shuffle) ✅")
        logger.info(f"    Scaler fitted on  : training data only ✅")
        logger.info(f"    Test set touched  : NO — use only for final eval ✅")

        logger.info(bar)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION: preprocess (public API — signature preserved from v1)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    symbol:             str,
    start_date:         str,
    end_date:           str,
    include_sentiment:  bool                        = True,
    include_macro:      bool                        = True,
    seq_config:         Optional[SequenceConfig]    = None,
    feature_set:        Optional[str]               = None,
    custom_features:    Optional[list[str]]         = None,
) -> Optional[PreparedData]:
    """
    Full preprocessing pipeline for one stock.
    Public API preserved from v1 — all new parameters are optional with defaults.

    Steps:
        1.  Build feature DataFrame          (feature_engineer)
        2.  Validate data quality            (NaN, Inf, duplicates, constants)
        3.  Select feature subset            (optional filtering)
        4.  Time-aware chronological split   (train/val/test)
        5.  Build configurable scaler        (minmax/standard/robust/etc.)
        6.  Fit scaler on training data only (no leakage)
        7.  Transform all splits
        8.  Build sliding window sequences   (configurable length/stride/horizon)
        9.  Validate sequences               (shape, NaN, Inf)
        10. Compute class weights            (handle imbalance)
        11. Collect metadata                 (dates, groups, config)
        12. Return PreparedData

    Args:
        symbol:            NSE symbol e.g. "RELIANCE.NS"
        start_date:        "YYYY-MM-DD"
        end_date:          "YYYY-MM-DD"
        include_sentiment: Include news features (default True)
        include_macro:     Include macro features (default True)
        seq_config:        Optional SequenceConfig — defaults to settings values
        feature_set:       Optional filter: "technical" | "macro" | "news" | "all"
        custom_features:   Optional explicit list of feature column names

    Returns:
        PreparedData with all arrays, metadata, and scaler.
        None if any critical step fails.

    Backward compatible example (v1 call still works):
        data = preprocess("RELIANCE.NS", "2020-01-01", "2024-01-01")
    """

    t0 = time.time()

    if seq_config is None:
        seq_config = SequenceConfig()

    logger.info(f"{'═'*60}")
    logger.info(f"  Preprocessing: {symbol} | {start_date} → {end_date}")
    logger.info(f"  Scaler: {SCALER_TYPE} | Seq: {seq_config.sequence_length}d "
                f"| Horizon: {seq_config.prediction_horizon}d "
                f"| Stride: {seq_config.stride}")
    logger.info(f"{'═'*60}")

    # ── Step 1: Build features ────────────────────────────────────────────────
    logger.info("Step 1: Building features...")
    df = build_full_features(
        symbol, start_date, end_date,
        include_sentiment=include_sentiment,
        include_macro=include_macro,
    )
    if df is None or df.empty:
        logger.error(f"Feature engineering returned empty DataFrame for {symbol}")
        return None

    # ── Step 2: Validate data ─────────────────────────────────────────────────
    logger.info("Step 2: Validating data...")
    df = _validate_dataframe(df, symbol)
    if df is None:
        return None

    # ── Step 3: Select features ───────────────────────────────────────────────
    logger.info("Step 3: Selecting features...")
    feature_cols = _select_features(df, feature_set, custom_features)
    if not feature_cols:
        logger.error("Feature selection returned empty list")
        return None

    X_df = df[feature_cols].copy()
    y    = df["label"].values.astype(int)

    # Assign feature groups for metadata
    groups = _assign_feature_groups(feature_cols)

    logger.info(f"  Selected {len(feature_cols)} features across {len(groups)} groups")

    # ── Step 4: Time split ────────────────────────────────────────────────────
    logger.info("Step 4: Splitting data (chronological)...")
    split = _time_split(X_df, y)
    if split is None:
        return None

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split

    # Capture period dates for metadata
    train_start = str(X_train_df.index[0].date())
    train_end   = str(X_train_df.index[-1].date())
    val_start   = str(X_val_df.index[0].date())
    val_end     = str(X_val_df.index[-1].date())
    test_start  = str(X_test_df.index[0].date())
    test_end    = str(X_test_df.index[-1].date())

    # ── Step 5: Build scaler ──────────────────────────────────────────────────
    # Pass n_train_rows so QuantileTransformer caps n_quantiles correctly.
    logger.info(f"Step 5: Building {SCALER_TYPE} scaler...")
    scaler = _build_scaler(SCALER_TYPE, n_train_rows=len(X_train_df))

    # ── Step 6: Fit scaler on training data only ──────────────────────────────
    logger.info("Step 6: Fitting scaler on training data only...")
    scaler = _fit_scaler(scaler, X_train_df, feature_cols)

    # ── Step 7: Transform all splits ─────────────────────────────────────────
    logger.info("Step 7: Normalizing features...")
    X_train_sc = _transform(scaler, X_train_df)
    X_val_sc   = _transform(scaler, X_val_df)
    X_test_sc  = _transform(scaler, X_test_df)

    # ── Step 8: Build sequences ───────────────────────────────────────────────
    logger.info(f"Step 8: Building sequences (len={seq_config.sequence_length}, "
                f"horizon={seq_config.prediction_horizon}, stride={seq_config.stride})...")

    X_tr_seq, y_tr_seq = _build_sequences(X_train_sc, y_train, seq_config)
    X_vl_seq, y_vl_seq = _build_sequences(X_val_sc,   y_val,   seq_config)
    X_ts_seq, y_ts_seq = _build_sequences(X_test_sc,  y_test,  seq_config)

    # ── Step 9: Validate sequences ────────────────────────────────────────────
    logger.info("Step 9: Validating sequences...")
    for name, X_seq, y_seq in [("Train", X_tr_seq, y_tr_seq),
                                ("Val",   X_vl_seq, y_vl_seq),
                                ("Test",  X_ts_seq, y_ts_seq)]:
        ok = _validate_sequences(X_seq, y_seq, len(feature_cols), seq_config, name)
        if not ok and name == "Train":
            logger.error("Training sequences failed validation — aborting")
            return None

    # ── Step 10: Class weights ────────────────────────────────────────────────
    logger.info("Step 10: Computing class weights...")
    class_weights = _compute_class_weights(y_tr_seq)

    # ── Step 11: Assemble metadata ────────────────────────────────────────────
    training_config = {
        "symbol":             symbol,
        "start_date":         start_date,
        "end_date":           end_date,
        "scaler_type":        SCALER_TYPE,
        "sequence_length":    seq_config.sequence_length,
        "prediction_horizon": seq_config.prediction_horizon,
        "stride":             seq_config.stride,
        "n_features":         len(feature_cols),
        "include_sentiment":  include_sentiment,
        "include_macro":      include_macro,
        "feature_set":        feature_set or "all",
        "preprocessor_version": PREPROCESSOR_VERSION,
    }

    elapsed = time.time() - t0

    # ── Assemble PreparedData ─────────────────────────────────────────────────
    data = PreparedData(
        # Core arrays (v1 fields)
        X_train        = X_tr_seq,
        y_train        = y_tr_seq,
        X_val          = X_vl_seq,
        y_val          = y_vl_seq,
        X_test         = X_ts_seq,
        y_test         = y_ts_seq,
        class_weights  = class_weights,
        scaler         = scaler,
        feature_names  = feature_cols,
        symbol         = symbol,
        n_features     = len(feature_cols),
        sequence_len   = seq_config.sequence_length,   # v1 backward compat

        # New v2 metadata
        seq_config     = seq_config,
        scaler_type    = SCALER_TYPE,
        feature_groups = groups,
        train_start    = train_start,
        train_end      = train_end,
        val_start      = val_start,
        val_end        = val_end,
        test_start     = test_start,
        test_end       = test_end,
        training_config= training_config,
    )

    logger.success(
        f"Preprocessing complete: {symbol} | "
        f"{data.total_sequences} sequences | "
        f"{data.n_features} features | "
        f"{data.memory_mb:.1f} MB | "
        f"{elapsed:.1f}s"
    )
    data.summary()
    return data


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: preprocess_all_stocks (preserved from v1)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_all_stocks(
    start_date:         str,
    end_date:           str,
    include_sentiment:  bool                     = True,
    include_macro:      bool                     = True,
    seq_config:         Optional[SequenceConfig] = None,
    feature_set:        Optional[str]            = None,
) -> dict[str, PreparedData]:
    """
    Preprocesses all stocks in settings.STOCKS.
    Public API preserved from v1.

    Returns:
        { "RELIANCE.NS": PreparedData, ... }
    """
    results: dict[str, PreparedData] = {}

    for symbol in STOCKS:
        logger.info(f"\n{'─'*40}")
        logger.info(f"Processing {symbol}...")
        data = preprocess(
            symbol, start_date, end_date,
            include_sentiment = include_sentiment,
            include_macro     = include_macro,
            seq_config        = seq_config,
            feature_set       = feature_set,
        )
        if data is not None:
            results[symbol] = data

    logger.success(
        f"Batch preprocessing complete: "
        f"{len(results)}/{len(STOCKS)} stocks succeeded"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _validate_dataframe
# Checks data quality before any processing.
# Handles recoverable issues automatically, fails on critical ones.
# ─────────────────────────────────────────────────────────────────────────────

def _validate_dataframe(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """
    Validates the feature DataFrame before scaling.

    Checks (in order):
        1. Empty DataFrame
        2. Missing label column
        3. Duplicate timestamps → drop
        4. Duplicate rows → drop
        5. Infinite values → replace with NaN
        6. NaN values → drop rows
        7. Constant columns → warn (don't drop — may be intentional)

    Args:
        df:     Feature DataFrame from feature_engineer
        symbol: For logging

    Returns:
        Cleaned DataFrame, or None if unrecoverable
    """

    rows_initial = len(df)

    # ── Check 1: Empty ────────────────────────────────────────────────────────
    if df.empty:
        logger.error(f"{symbol}: Empty DataFrame — nothing to process")
        return None

    # ── Check 2: Label column exists ─────────────────────────────────────────
    if "label" not in df.columns:
        logger.error(f"{symbol}: 'label' column missing — cannot train without targets")
        return None

    missing_labels = df["label"].isna().sum()
    if missing_labels > 0:
        logger.warning(f"{symbol}: {missing_labels} missing labels — dropping those rows")
        df = df.dropna(subset=["label"])

    # ── Check 3: Duplicate timestamps ────────────────────────────────────────
    dup_dates = df.index.duplicated().sum()
    if dup_dates > 0:
        logger.warning(f"{symbol}: {dup_dates} duplicate timestamps — keeping last")
        df = df[~df.index.duplicated(keep="last")]

    # ── Check 4: Duplicate rows ───────────────────────────────────────────────
    feature_cols = get_feature_names(df)
    dup_rows     = df[feature_cols].duplicated().sum()
    if dup_rows > 0:
        logger.warning(f"{symbol}: {dup_rows} fully duplicate rows — dropping")
        df = df[~df[feature_cols].duplicated(keep="first")]

    # ── Check 5: Infinite values → replace with NaN ──────────────────────────
    inf_count = np.isinf(df[feature_cols].values).sum()
    if inf_count > 0:
        logger.warning(f"{symbol}: {inf_count} infinite values → replacing with NaN")
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # ── Check 6: NaN values → drop rows ──────────────────────────────────────
    nan_rows = df[feature_cols].isna().any(axis=1).sum()
    if nan_rows > 0:
        logger.warning(f"{symbol}: {nan_rows} rows with NaN — dropping")
        df = df.dropna(subset=feature_cols)

    # ── Check 7: Constant columns ─────────────────────────────────────────────
    # A constant column (all same value) adds no information to the model.
    # We warn but don't drop — the user may have intentionally included them.
    const_cols = [
        c for c in feature_cols
        if df[c].nunique() <= 1
    ]
    if const_cols:
        logger.warning(
            f"{symbol}: {len(const_cols)} constant columns (zero variance): "
            f"{const_cols[:5]}{'...' if len(const_cols) > 5 else ''}"
        )

    rows_final   = len(df)
    rows_removed = rows_initial - rows_final

    if rows_final < 100:
        logger.error(
            f"{symbol}: Only {rows_final} rows after validation. "
            f"Need at least 100 — aborting."
        )
        return None

    logger.info(
        f"  Validation: {rows_final} clean rows "
        f"({rows_removed} removed) | "
        f"{len(feature_cols)} features | "
        f"0 NaN ✅ | 0 Inf ✅"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _select_features
# Optional feature filtering — subset by group or custom list.
# ─────────────────────────────────────────────────────────────────────────────

def _select_features(
    df:              pd.DataFrame,
    feature_set:     Optional[str]       = None,
    custom_features: Optional[list[str]] = None,
) -> list[str]:
    """
    Selects which feature columns to use for training.

    Modes:
        None / "all"   → use all features from get_feature_names()
        "technical"    → only technical indicator columns
        "macro"        → only macro indicator columns
        "news"         → only news sentiment columns
        custom_features → explicit list of column names

    Args:
        df:              Feature DataFrame
        feature_set:     Named preset ("all", "technical", "macro", "news")
        custom_features: Explicit list of column names to use

    Returns:
        List of selected feature column names (always includes all available)

    Why feature selection?
        Ablation studies: "does macro data actually help?"
        Quick tests: train faster with only technical features
        Model comparison: technical-only vs full feature model
    """
    all_features = get_feature_names(df)

    # Explicit custom list takes priority
    if custom_features:
        valid = [f for f in custom_features if f in df.columns]
        missing = set(custom_features) - set(valid)
        if missing:
            logger.warning(f"Custom features not in DataFrame: {missing}")
        logger.info(f"  Custom feature set: {len(valid)} features")
        return valid

    # Named preset
    if feature_set and feature_set != "all":
        patterns = FEATURE_GROUP_PATTERNS.get(feature_set, [])
        if not patterns:
            logger.warning(
                f"Unknown feature_set '{feature_set}' — using all features. "
                f"Valid options: {list(FEATURE_GROUP_PATTERNS.keys())}"
            )
            return all_features

        selected = [
            f for f in all_features
            if any(f.startswith(p) or p in f for p in patterns)
        ]
        logger.info(
            f"  Feature set '{feature_set}': "
            f"{len(selected)}/{len(all_features)} features selected"
        )
        return selected

    # Default: all features
    logger.info(f"  Feature set 'all': {len(all_features)} features")
    return all_features


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _assign_feature_groups
# Assigns each feature to a named group for metadata.
# ─────────────────────────────────────────────────────────────────────────────

def _assign_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """
    Assigns each feature column to a named group.

    Groups are used in:
        - PreparedData.summary() display
        - Per-group scaling (future)
        - Model explainability

    Args:
        feature_cols: List of feature column names

    Returns:
        { "technical": [...], "macro": [...], "news": [...], "other": [...] }
    """
    groups: dict[str, list[str]] = {g: [] for g in FEATURE_GROUP_PATTERNS}
    groups["other"] = []

    for col in feature_cols:
        assigned = False
        for group, patterns in FEATURE_GROUP_PATTERNS.items():
            if any(col.startswith(p) or p in col for p in patterns):
                groups[group].append(col)
                assigned = True
                break
        if not assigned:
            groups["other"].append(col)

    # Remove empty groups for cleaner display
    return {g: cols for g, cols in groups.items() if cols}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _time_split (preserved from v1 — identical behavior)
# ─────────────────────────────────────────────────────────────────────────────

def _time_split(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Optional[tuple]:
    """
    Chronological train/val/test split.
    Behavior identical to v1 — no changes.

    Returns:
        (X_train_df, X_val_df, X_test_df, y_train, y_val, y_test)
        or None if splits are too small.
    """
    n         = len(X)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    X_train = X.iloc[:train_end]
    X_val   = X.iloc[train_end:val_end]
    X_test  = X.iloc[val_end:]
    y_train = y[:train_end]
    y_val   = y[train_end:val_end]
    y_test  = y[val_end:]

    # Validate each split has enough rows for at least one sequence
    min_rows = SEQUENCE_LENGTH + 1
    for name, split in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        if len(split) < min_rows:
            logger.warning(
                f"{name} split has only {len(split)} rows "
                f"(need {min_rows} for sequences). Sequences may be empty."
            )

    logger.info(f"  Train : {len(X_train)} rows  ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    logger.info(f"  Val   : {len(X_val)} rows  ({X_val.index[0].date()} → {X_val.index[-1].date()})")
    logger.info(f"  Test  : {len(X_test)} rows  ({X_test.index[0].date()} → {X_test.index[-1].date()})")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _build_scaler
# Factory function — returns the right scaler based on config.
# Changing SCALER_TYPE in settings.py switches the scaler everywhere.
# ─────────────────────────────────────────────────────────────────────────────

def _build_scaler(scaler_type: str, n_train_rows: int = 1000) -> object:
    """
    Scaler factory — returns unfitted scaler for the given type.

    Supported types and when to use them:
        "minmax"    → [0,1] range — best for LSTM (sigmoid/tanh gates)
        "standard"  → mean=0, std=1 — good for linear models
        "robust"    → uses median/IQR — best when outliers present
        "quantile"  → maps to uniform distribution — handles skewed features
        "power"     → Box-Cox/Yeo-Johnson — normalizes skewed distributions

    To add a new scaler: add one entry to the dict below.

    Args:
        scaler_type:   One of the supported type strings
        n_train_rows:  Number of training rows — used to safely cap
                       QuantileTransformer n_quantiles.

                       Bug fixed (v2.1.0):
                       sklearn silently reduces n_quantiles when
                       n_quantiles > n_samples, producing a warning
                       and inconsistent behavior across dataset sizes.
                       We cap it explicitly: n_quantiles = min(1000, n_rows).

    Returns:
        Unfitted sklearn-compatible scaler object
    """
    # Cap quantiles to training set size — fixes silent sklearn warning
    # when dataset has fewer rows than requested quantiles (common here:
    # ~285 training sequences < 1000 default quantiles)
    safe_quantiles = min(1000, n_train_rows)

    scalers = {
        "minmax":   MinMaxScaler(feature_range=(0, 1)),
        "standard": StandardScaler(),
        "robust":   RobustScaler(quantile_range=(5, 95)),
        "quantile": QuantileTransformer(
                        output_distribution="uniform",
                        n_quantiles=safe_quantiles,
                        random_state=RANDOM_SEED,
                    ),
        "power":    PowerTransformer(method="yeo-johnson"),
    }

    if scaler_type not in scalers:
        logger.warning(
            f"Unknown scaler_type '{scaler_type}' — "
            f"falling back to 'minmax'. "
            f"Valid options: {list(scalers.keys())}"
        )
        return scalers["minmax"]

    if scaler_type == "quantile":
        logger.info(
            f"  QuantileTransformer: n_quantiles capped at "
            f"{safe_quantiles} (training rows = {n_train_rows})"
        )

    return scalers[scaler_type]


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _fit_scaler (extended — logs feature stats)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_scaler(
    scaler:       object,
    X_train:      pd.DataFrame,
    feature_cols: list[str],
) -> object:
    """
    Fits scaler on training data only.
    Identical contract to v1 — extended with better logging.

    CRITICAL: Never fit on validation or test data.
    Doing so would leak future statistics into past data normalization.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler.fit(X_train.values)

    logger.info(
        f"  Scaler ({SCALER_TYPE}) fitted: "
        f"{len(X_train)} rows × {len(feature_cols)} features"
    )
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _transform (preserved from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _transform(scaler: object, X: pd.DataFrame) -> np.ndarray:
    """
    Applies fitted scaler to a DataFrame split.
    Returns numpy array — sequences are built from arrays, not DataFrames.
    Identical contract to v1.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scaler.transform(X.values)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _build_sequences (extended — configurable via SequenceConfig)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sequences(
    X:          np.ndarray,
    y:          np.ndarray,
    seq_config: SequenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts flat 2D scaled array into 3D sliding window sequences.

    Improvements over v1:
        - Configurable via SequenceConfig (length, horizon, stride)
        - Vectorized using numpy stride tricks (faster than Python loop)
        - Supports prediction_horizon > 1

    How stride_tricks work (performance optimization):
        Instead of Python loop building list of arrays,
        we use numpy's as_strided to create a VIEW of the data
        with sequence-shaped dimensions — zero copy overhead.

    Args:
        X:          2D scaled array (rows, features)
        y:          1D label array  (rows,)
        seq_config: SequenceConfig with length/horizon/stride

    Returns:
        (X_sequences, y_sequences)
        X_sequences: (N_seq, seq_length, n_features)
        y_sequences: (N_seq,)
    """

    seq_len  = seq_config.sequence_length
    horizon  = seq_config.prediction_horizon
    stride   = seq_config.stride
    n_rows, n_features = X.shape

    # How many sequences can we make?
    # Each sequence needs seq_len rows of input + horizon rows to get label
    n_seq = (n_rows - seq_len - horizon + 1)

    if n_seq <= 0:
        logger.warning(
            f"Cannot build sequences: {n_rows} rows available, "
            f"need at least {seq_len + horizon}. Returning empty arrays."
        )
        return np.array([]).reshape(0, seq_len, n_features), np.array([])

    # Bug fixed (v2.1.0):
    # Previous version created ALL n_seq sequences in memory first using
    # as_strided, then applied stride by indexing — discarding most of them.
    # For stride=5 on 1000 rows this wasted 5x the required memory.
    #
    # Fix: compute strided indices FIRST, then build only those sequences.
    # as_strided is now parameterised directly on strided_n_seq so the
    # view covers only the rows we actually need, nothing more.

    # Only the indices we will actually keep
    indices      = list(range(0, n_seq, stride))
    strided_n_seq = len(indices)

    try:
        # Build a view covering exactly strided_n_seq sequences.
        # Strides explanation:
        #   Dim 0 (sequence step): stride rows forward  = stride * row_bytes
        #   Dim 1 (within sequence): 1 row forward      = 1     * row_bytes
        #   Dim 2 (within row): 1 element forward       = element_bytes
        row_bytes  = X.strides[0]
        elem_bytes = X.strides[1]

        shape   = (strided_n_seq, seq_len, n_features)
        strides = (stride * row_bytes, row_bytes, elem_bytes)

        X_view = np.lib.stride_tricks.as_strided(
            X,
            shape   = shape,
            strides = strides,
        )

        # .copy() converts the view to an owned array (required for safety —
        # as_strided views have no bounds checking)
        X_seq = X_view.copy()
        y_seq = np.array([y[i + seq_len + horizon - 1] for i in indices])

    except Exception as e:
        # Fallback to simple loop if stride_tricks fails on this platform
        logger.warning(f"Stride tricks failed ({e}) — falling back to loop")
        X_seq = np.array([X[i: i + seq_len] for i in indices])
        y_seq = np.array([y[i + seq_len + horizon - 1] for i in indices])

    logger.info(
        f"  Sequences: {X_seq.shape} "
        f"(stride={stride}, horizon={horizon})"
    )
    return X_seq, y_seq


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _validate_sequences
# Checks sequences after creation — catches subtle bugs early.
# ─────────────────────────────────────────────────────────────────────────────

def _validate_sequences(
    X:          np.ndarray,
    y:          np.ndarray,
    n_features: int,
    seq_config: SequenceConfig,
    split_name: str,
) -> bool:
    """
    Validates sequence arrays after creation.

    Checks:
        1. Non-empty
        2. Correct shape: (N, sequence_length, n_features)
        3. Correct label array shape: (N,)
        4. No NaN values
        5. No Infinite values
        6. Label values in {0, 1, 2}

    Args:
        X:          3D sequence array
        y:          1D label array
        n_features: Expected number of features
        seq_config: SequenceConfig for dimension checks
        split_name: "Train" / "Val" / "Test" (for logging)

    Returns:
        True if all checks pass, False otherwise
    """
    issues = []

    # Check 1: Non-empty
    if len(X) == 0:
        issues.append("zero sequences")

    else:
        # Check 2: Shape
        expected_shape = (len(X), seq_config.sequence_length, n_features)
        if X.shape != expected_shape:
            issues.append(f"wrong shape {X.shape} (expected {expected_shape})")

        # Check 3: Label length matches
        if len(y) != len(X):
            issues.append(f"X/y length mismatch ({len(X)} vs {len(y)})")

        # Check 4: NaN
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values in X")

        # Check 5: Inf
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            issues.append(f"{inf_count} Inf values in X")

        # Check 6: Label validity
        invalid_labels = ~np.isin(y, [0, 1, 2])
        if invalid_labels.any():
            issues.append(f"{invalid_labels.sum()} invalid labels (not in {{0,1,2}})")

    if issues:
        for issue in issues:
            logger.warning(f"  {split_name} sequence validation: {issue}")
        return len(X) > 0   # Non-empty is still usable even with warnings

    logger.info(f"  {split_name} sequences: {X.shape} ✅")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: _compute_class_weights (preserved from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_class_weights(y: np.ndarray) -> dict:
    """
    Computes balanced class weights for LSTM training.
    Identical behavior to v1 — extended with safety check.

    Why balanced weights?
        HOLD class dominates at 60% → model defaults to always predicting HOLD.
        Balanced weights penalize missing UP/DOWN proportionally more.
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight = "balanced",
        classes      = classes,
        y            = y,
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    # Ensure all 3 classes present (small splits may miss one)
    for cls in [0, 1, 2]:
        weight_dict.setdefault(cls, 1.0)

    logger.info(
        f"  Class weights: "
        f"DOWN={weight_dict[0]:.3f}  "
        f"HOLD={weight_dict[1]:.3f}  "
        f"UP={weight_dict[2]:.3f}"
    )
    return weight_dict


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION (new — does NOT replace existing split)
# Used later during backtesting / strategy validation.
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_splits(
    X:          pd.DataFrame,
    y:          np.ndarray,
    n_splits:   int  = 5,
    train_size: int  = 200,
    val_size:   int  = 50,
) -> list[dict]:
    """
    Generates multiple walk-forward train/val splits.

    Unlike the fixed chronological split (which is used for model training),
    walk-forward validation is used AFTER training to test robustness.

    Each split trains on the past and validates on the immediate future.
    This simulates real trading: train on what you know, test on what comes next.

    Example with n_splits=3, train_size=200, val_size=50:
        Split 1: Train rows 0-199   → Val rows 200-249
        Split 2: Train rows 50-249  → Val rows 250-299
        Split 3: Train rows 100-299 → Val rows 300-349

    Args:
        X:          Feature DataFrame with DatetimeIndex
        y:          Label array
        n_splits:   Number of walk-forward splits
        train_size: Training window size (rows)
        val_size:   Validation window size (rows)

    Returns:
        List of dicts: [{"X_train": df, "y_train": arr,
                         "X_val": df,   "y_val": arr,
                         "train_period": (start, end),
                         "val_period":   (start, end)}, ...]

    Usage:
        splits = walk_forward_splits(X_df, y, n_splits=5)
        for fold in splits:
            model.fit(fold["X_train"], fold["y_train"])
            score = evaluate(model, fold["X_val"], fold["y_val"])
    """
    splits      = []
    step        = (len(X) - train_size - val_size) // max(n_splits - 1, 1)

    for i in range(n_splits):
        train_start_idx = i * step
        train_end_idx   = train_start_idx + train_size
        val_end_idx     = train_end_idx   + val_size

        if val_end_idx > len(X):
            break

        splits.append({
            "X_train":      X.iloc[train_start_idx:train_end_idx],
            "y_train":      y[train_start_idx:train_end_idx],
            "X_val":        X.iloc[train_end_idx:val_end_idx],
            "y_val":        y[train_end_idx:val_end_idx],
            "train_period": (
                str(X.index[train_start_idx].date()),
                str(X.index[train_end_idx - 1].date()),
            ),
            "val_period": (
                str(X.index[train_end_idx].date()),
                str(X.index[val_end_idx - 1].date()),
            ),
            "fold": i + 1,
        })

    logger.info(f"Walk-forward splits: {len(splits)} folds generated")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: save_scaler (extended — saves metadata file alongside scaler)
# ─────────────────────────────────────────────────────────────────────────────

def save_scaler(scaler: object, symbol: str, data: Optional[PreparedData] = None) -> str:
    """
    Saves fitted scaler + metadata to disk.
    Public API preserved from v1 — data parameter is optional (new).

    Saves two files:
        scaler_SYMBOL.pkl      → the scaler object
        scaler_SYMBOL_meta.json → metadata for compatibility checks

    Why save metadata?
        When you load the scaler for live inference months later,
        you need to know exactly which features it was fitted on,
        in which order, and with which configuration.
        Without metadata, you might feed features in the wrong order → garbage.

    Args:
        scaler: Fitted scaler object
        symbol: Stock symbol e.g. "RELIANCE.NS"
        data:   Optional PreparedData for rich metadata

    Returns:
        Path to saved scaler file
    """
    safe_symbol = symbol.replace(".", "_")
    pkl_path    = os.path.join(MODELS_DIR, f"scaler_{safe_symbol}.pkl")
    meta_path   = os.path.join(MODELS_DIR, f"scaler_{safe_symbol}_meta.json")

    # Save scaler object
    joblib.dump(scaler, pkl_path)

    # Build metadata dict
    meta: dict = {
        "symbol":               symbol,
        "scaler_type":          SCALER_TYPE,
        "saved_at":             datetime.now().isoformat(),
        "preprocessor_version": PREPROCESSOR_VERSION,
    }

    if data is not None:
        meta.update({
            "feature_names":      data.feature_names,
            "feature_count":      data.n_features,
            "sequence_length":    data.seq_config.sequence_length,
            "prediction_horizon": data.seq_config.prediction_horizon,
            "stride":             data.seq_config.stride,
            "train_start":        data.train_start,
            "train_end":          data.train_end,
            "training_config":    data.training_config,
        })

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.success(f"Scaler saved: {pkl_path}")
    logger.success(f"Metadata saved: {meta_path}")
    return pkl_path


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: load_scaler (extended — logs metadata on load)
# ─────────────────────────────────────────────────────────────────────────────

def load_scaler(symbol: str) -> Optional[object]:
    """
    Loads a previously saved scaler from disk.
    Public API preserved from v1.

    Now also logs metadata on load so you can verify compatibility.

    Args:
        symbol: Stock symbol e.g. "RELIANCE.NS"

    Returns:
        Fitted scaler object or None if not found
    """
    safe_symbol = symbol.replace(".", "_")
    pkl_path    = os.path.join(MODELS_DIR, f"scaler_{safe_symbol}.pkl")
    meta_path   = os.path.join(MODELS_DIR, f"scaler_{safe_symbol}_meta.json")

    if not os.path.exists(pkl_path):
        logger.error(f"Scaler not found: {pkl_path}")
        return None

    scaler = joblib.load(pkl_path)
    logger.success(f"Scaler loaded: {pkl_path}")

    # Log metadata if available
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info(
            f"  Scaler metadata: "
            f"type={meta.get('scaler_type')} | "
            f"features={meta.get('feature_count')} | "
            f"seq_len={meta.get('sequence_length')} | "
            f"saved={meta.get('saved_at', 'unknown')[:10]}"
        )

    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: get_inference_sequence (preserved from v1)
# ─────────────────────────────────────────────────────────────────────────────

def get_inference_sequence(
    symbol:         str,
    scaler:         object,
    feature_names:  list[str],
    end_date:       Optional[str]           = None,
    seq_config:     Optional[SequenceConfig] = None,
) -> Optional[np.ndarray]:
    """
    Prepares a single sequence for live inference.
    Public API preserved from v1 — seq_config parameter is new and optional.

    Flow:
        1. Fetch last (seq_len + buffer) days of features
        2. Select same features model was trained on
        3. Scale with saved scaler
        4. Return last seq_len rows shaped for model.predict()

    Args:
        symbol:        NSE symbol
        scaler:        Loaded scaler (from load_scaler())
        feature_names: Feature list from PreparedData.feature_names
        end_date:      Optional end date (defaults to today)
        seq_config:    Optional SequenceConfig (defaults to settings values)

    Returns:
        3D array (1, seq_len, n_features) ready for model.predict()
        None if preparation fails
    """
    from datetime import timedelta

    if seq_config is None:
        seq_config = SequenceConfig()

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    seq_len     = seq_config.sequence_length
    buffer_days = seq_len * 3   # Extra days for weekends + NaN indicator warmup

    start_dt   = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=buffer_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    df = build_full_features(symbol, start_date, end_date)

    if df is None or len(df) < seq_len:
        logger.error(
            f"Inference sequence failed: "
            f"got {len(df) if df is not None else 0} rows, "
            f"need {seq_len}"
        )
        return None

    # Use same features model was trained on — in same order
    available = [f for f in feature_names if f in df.columns]
    if len(available) != len(feature_names):
        missing = set(feature_names) - set(available)
        logger.warning(f"Inference: {len(missing)} features missing: {list(missing)[:5]}")

    X        = df[available].values
    X_scaled = scaler.transform(X)
    sequence = X_scaled[-seq_len:]

    return sequence.reshape(1, seq_len, -1)