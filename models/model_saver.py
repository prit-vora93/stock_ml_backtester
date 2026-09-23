"""
models/model_saver.py
----------------------
Saves and loads trained model artifacts to/from MODELS_DIR, alongside
metadata for compatibility checks.

Mirrors the save_scaler()/load_scaler() pattern in data/preprocessor.py:
each model gets its own artifact file plus a `_meta.json` sidecar
recording what it was trained on (symbol, feature count, sequence
length, training config, timestamp) so a model loaded months later
can be checked for compatibility before being fed live data.

Usage:
    from models.model_saver import (
        save_lstm, load_lstm,
        save_xgboost, load_xgboost,
        save_ensemble_config, load_ensemble_config,
    )
"""

import json
import os
from datetime import datetime
from typing import Optional

from config.settings import MODELS_DIR
from utils.logger import logger


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "_")


def _write_meta(path: str, symbol: str, model_type: str, metadata: Optional[dict]) -> None:
    meta: dict = {
        "symbol":     symbol,
        "model_type": model_type,
        "saved_at":   datetime.now().isoformat(),
    }
    if metadata:
        meta.update(metadata)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _read_meta(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM (Keras)
# ─────────────────────────────────────────────────────────────────────────────

def save_lstm(model, symbol: str, metadata: Optional[dict] = None) -> str:
    """
    Saves a trained Keras LSTM model + metadata for a symbol.

    Args:
        model:    Trained keras.Model (from lstm_predictor.build_lstm_model)
        symbol:   Stock symbol e.g. "RELIANCE.NS"
        metadata: Optional dict (feature_names, sequence_length, training_config, ...)

    Returns:
        Path to the saved model file.
    """
    safe      = _safe_symbol(symbol)
    path      = os.path.join(MODELS_DIR, f"lstm_{safe}.keras")
    meta_path = os.path.join(MODELS_DIR, f"lstm_{safe}_meta.json")

    model.save(path)
    _write_meta(meta_path, symbol, "lstm", metadata)

    logger.success(f"LSTM model saved: {path}")
    return path


def load_lstm(symbol: str):
    """
    Loads a previously saved Keras LSTM model for a symbol.

    Returns:
        keras.Model or None if not found.
    """
    from tensorflow import keras

    safe      = _safe_symbol(symbol)
    path      = os.path.join(MODELS_DIR, f"lstm_{safe}.keras")
    meta_path = os.path.join(MODELS_DIR, f"lstm_{safe}_meta.json")

    if not os.path.exists(path):
        logger.error(f"LSTM model not found: {path}")
        return None

    model = keras.models.load_model(path)
    logger.success(f"LSTM model loaded: {path}")

    meta = _read_meta(meta_path)
    if meta:
        logger.info(
            f"  LSTM metadata: features={meta.get('n_features')} | "
            f"seq_len={meta.get('sequence_length')} | "
            f"saved={meta.get('saved_at', 'unknown')[:10]}"
        )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def save_xgboost(model, symbol: str, metadata: Optional[dict] = None) -> str:
    """
    Saves a trained XGBClassifier + metadata for a symbol.

    Args:
        model:    Trained xgboost.XGBClassifier
        symbol:   Stock symbol e.g. "RELIANCE.NS"
        metadata: Optional dict (feature_names, training_config, ...)

    Returns:
        Path to the saved model file.
    """
    safe      = _safe_symbol(symbol)
    path      = os.path.join(MODELS_DIR, f"xgb_{safe}.json")
    meta_path = os.path.join(MODELS_DIR, f"xgb_{safe}_meta.json")

    model.save_model(path)
    _write_meta(meta_path, symbol, "xgboost", metadata)

    logger.success(f"XGBoost model saved: {path}")
    return path


def load_xgboost(symbol: str):
    """
    Loads a previously saved XGBoost model for a symbol.

    Returns:
        xgboost.XGBClassifier or None if not found.
    """
    import xgboost as xgb

    safe      = _safe_symbol(symbol)
    path      = os.path.join(MODELS_DIR, f"xgb_{safe}.json")
    meta_path = os.path.join(MODELS_DIR, f"xgb_{safe}_meta.json")

    if not os.path.exists(path):
        logger.error(f"XGBoost model not found: {path}")
        return None

    model = xgb.XGBClassifier()
    model.load_model(path)
    logger.success(f"XGBoost model loaded: {path}")

    meta = _read_meta(meta_path)
    if meta:
        logger.info(
            f"  XGBoost metadata: features={meta.get('n_features')} | "
            f"saved={meta.get('saved_at', 'unknown')[:10]}"
        )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble config
# ─────────────────────────────────────────────────────────────────────────────
# The ensemble itself has no trainable parameters — just the weights and
# confidence threshold it was built with. Saving these lets a later
# inference run reconstruct the exact same combination logic.
# ─────────────────────────────────────────────────────────────────────────────

def save_ensemble_config(symbol: str, config: dict) -> str:
    """
    Saves ensemble configuration (weights, confidence threshold, etc.)
    for a symbol.

    Args:
        symbol: Stock symbol e.g. "RELIANCE.NS"
        config: dict, typically {"lstm_weight": .., "xgb_weight": ..,
                                  "min_confidence": .., "saved_at": ..}

    Returns:
        Path to the saved config file.
    """
    safe = _safe_symbol(symbol)
    path = os.path.join(MODELS_DIR, f"ensemble_{safe}_config.json")

    payload = dict(config)
    payload.setdefault("symbol", symbol)
    payload.setdefault("saved_at", datetime.now().isoformat())

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.success(f"Ensemble config saved: {path}")
    return path


def load_ensemble_config(symbol: str) -> Optional[dict]:
    """
    Loads a previously saved ensemble config for a symbol.

    Returns:
        dict or None if not found (caller should fall back to
        settings.LSTM_WEIGHT / XGB_WEIGHT / MIN_CONFIDENCE defaults).
    """
    safe = _safe_symbol(symbol)
    path = os.path.join(MODELS_DIR, f"ensemble_{safe}_config.json")

    config = _read_meta(path)
    if config is None:
        logger.warning(f"Ensemble config not found: {path} — caller should use defaults")
        return None

    logger.success(f"Ensemble config loaded: {path}")
    return config
