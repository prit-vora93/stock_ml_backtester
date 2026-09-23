"""
models/ensemble.py
--------------------
Combines LSTM + XGBoost probability outputs into a single trading signal.

Combination:
    ensemble_probs = LSTM_WEIGHT * lstm_probs + XGB_WEIGHT * xgb_probs

Signal:
    argmax(ensemble_probs) -> DOWN/HOLD/UP
    if confidence (max prob) < MIN_CONFIDENCE -> forced to HOLD
      (not confident enough to act, regardless of which class "won")

    DOWN -> "SELL", HOLD -> "HOLD", UP -> "BUY"
    (matches api.database.Prediction.ensemble_signal's string values)

This module is deliberately DB-agnostic — it only combines arrays of
probabilities and returns plain dicts/arrays. predictions_to_db_rows()
shapes its output to match api.database.Prediction's columns, but the
actual DB session/insert is left to whatever caller owns persistence
(backtesting engine / API layer), keeping this module a pure function
over numpy arrays that's trivial to unit test.

Usage:
    from models.ensemble import EnsembleConfig, predict_ensemble
    from models.lstm_predictor import predict_lstm
    from models.xgboost_classifier import predict_xgboost

    lstm_probs = predict_lstm(lstm_model, data.X_test)
    xgb_probs  = predict_xgboost(xgb_model, data.X_test)
    result     = predict_ensemble(lstm_probs, xgb_probs)
    # result["signal"], result["confidence"], result["predicted_class"], result["probs"]
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config.settings import (
    LSTM_WEIGHT,
    XGB_WEIGHT,
    MIN_CONFIDENCE,
    LABEL_DOWN,
    LABEL_HOLD,
    LABEL_UP,
)
from utils.logger import logger

LABEL_NAMES = ["DOWN", "HOLD", "UP"]
SIGNAL_MAP  = {LABEL_DOWN: "SELL", LABEL_HOLD: "HOLD", LABEL_UP: "BUY"}


@dataclass
class EnsembleConfig:
    """
    Ensemble combination parameters. Defaults come from settings.py so
    behavior matches project-wide config unless explicitly overridden
    (e.g. by models/optimizer.py during tuning).
    """
    lstm_weight:    float = LSTM_WEIGHT
    xgb_weight:     float = XGB_WEIGHT
    min_confidence: float = MIN_CONFIDENCE

    def __post_init__(self) -> None:
        total = self.lstm_weight + self.xgb_weight
        if not np.isclose(total, 1.0, atol=1e-6):
            logger.warning(
                f"Ensemble weights sum to {total:.3f}, not 1.0 — normalizing "
                f"({self.lstm_weight:.3f}/{self.xgb_weight:.3f})"
            )
            self.lstm_weight /= total
            self.xgb_weight  /= total


def combine_probabilities(
    lstm_probs: np.ndarray,
    xgb_probs:  np.ndarray,
    config:     Optional[EnsembleConfig] = None,
) -> np.ndarray:
    """
    Weighted average of LSTM and XGBoost probabilities.

    Args:
        lstm_probs: (N, 3) [DOWN, HOLD, UP] — from predict_lstm()
        xgb_probs:  (N, 3) [DOWN, HOLD, UP] — from predict_xgboost(),
                    same row order/alignment as lstm_probs (i.e. row i in
                    both must be predictions for the same sequence)
        config:     EnsembleConfig (defaults to settings.py weights)

    Returns:
        (N, 3) weighted-average probabilities.
    """
    if config is None:
        config = EnsembleConfig()

    if lstm_probs.shape != xgb_probs.shape:
        raise ValueError(
            f"lstm_probs shape {lstm_probs.shape} != xgb_probs shape {xgb_probs.shape} "
            f"— both models must predict on the same aligned rows"
        )

    return config.lstm_weight * lstm_probs + config.xgb_weight * xgb_probs


def probs_to_signals(ensemble_probs: np.ndarray, config: Optional[EnsembleConfig] = None) -> dict:
    """
    Converts ensemble probabilities into discrete classes/signals.

    Args:
        ensemble_probs: (N, 3) [DOWN, HOLD, UP]
        config:         EnsembleConfig (defaults to settings.py)

    Returns:
        {
            "predicted_class": (N,) int array in {0,1,2} — raw argmax,
                                BEFORE confidence gating (useful for
                                accuracy metrics against true labels),
            "confidence":       (N,) float array — max probability,
            "signal":           (N,) array of "BUY"/"HOLD"/"SELL" strings —
                                AFTER confidence gating: predictions below
                                min_confidence are forced to "HOLD"
                                regardless of which class had the highest
                                probability (not confident enough to act).
        }
    """
    if config is None:
        config = EnsembleConfig()

    predicted_class = ensemble_probs.argmax(axis=1)
    confidence      = ensemble_probs.max(axis=1)

    signal = np.array([SIGNAL_MAP[int(c)] for c in predicted_class], dtype=object)
    low_confidence = confidence < config.min_confidence
    signal[low_confidence] = "HOLD"

    return {
        "predicted_class": predicted_class,
        "confidence":      confidence,
        "signal":          signal,
    }


def predict_ensemble(
    lstm_probs: np.ndarray,
    xgb_probs:  np.ndarray,
    config:     Optional[EnsembleConfig] = None,
) -> dict:
    """
    Full pipeline: combine LSTM + XGBoost probabilities into signals.

    Returns:
        {"probs": (N,3), "predicted_class": (N,), "confidence": (N,), "signal": (N,)}
    """
    if config is None:
        config = EnsembleConfig()

    ensemble_probs = combine_probabilities(lstm_probs, xgb_probs, config)
    result = probs_to_signals(ensemble_probs, config)
    result["probs"] = ensemble_probs
    return result


def evaluate_ensemble(
    lstm_probs: np.ndarray,
    xgb_probs:  np.ndarray,
    y_true:     np.ndarray,
    config:     Optional[EnsembleConfig] = None,
) -> dict:
    """
    Evaluates ensemble accuracy against true labels.

    Note: uses `predicted_class` (raw argmax), not the confidence-gated
    `signal` — folding low-confidence predictions into HOLD makes
    "accuracy" ill-defined (a correct DOWN call at low confidence would
    count as a wrong HOLD prediction). `pct_actionable` below reports
    how often the ensemble was confident enough to actually act.

    Returns:
        {"accuracy": float, "f1_macro": float, "report": str,
         "avg_confidence": float, "pct_actionable": float}
    """
    result = predict_ensemble(lstm_probs, xgb_probs, config)
    y_pred = result["predicted_class"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "report":   classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0
        ),
        "avg_confidence": float(result["confidence"].mean()),
        "pct_actionable":  float((result["signal"] != "HOLD").mean()),
    }
    logger.info(
        f"Ensemble eval: accuracy={metrics['accuracy']:.3f}  "
        f"f1_macro={metrics['f1_macro']:.3f}  "
        f"avg_confidence={metrics['avg_confidence']:.3f}  "
        f"actionable={metrics['pct_actionable']:.1%}"
    )
    return metrics


def predictions_to_db_rows(
    lstm_probs:      np.ndarray,
    xgb_probs:       np.ndarray,
    ensemble_result: dict,
) -> list[dict]:
    """
    Shapes ensemble output into dicts matching api.database.Prediction's
    probability/signal columns, one per row — ready to be passed as
    Prediction(**row, symbol=..., date=...) by whatever caller owns the
    DB session (kept out of this module so it stays a pure function over
    numpy arrays).

    Args:
        lstm_probs:      (N, 3) [DOWN, HOLD, UP]
        xgb_probs:       (N, 3) [DOWN, HOLD, UP]
        ensemble_result: output of predict_ensemble() for the same rows

    Returns:
        List of N dicts with keys: lstm_prob_down/hold/up,
        xgb_prob_down/hold/up, ensemble_signal, confidence.
    """
    n = len(lstm_probs)
    confidence = ensemble_result["confidence"]
    signal     = ensemble_result["signal"]

    return [
        {
            "lstm_prob_down": float(lstm_probs[i, 0]),
            "lstm_prob_hold": float(lstm_probs[i, 1]),
            "lstm_prob_up":   float(lstm_probs[i, 2]),
            "xgb_prob_down":  float(xgb_probs[i, 0]),
            "xgb_prob_hold":  float(xgb_probs[i, 1]),
            "xgb_prob_up":    float(xgb_probs[i, 2]),
            "ensemble_signal": str(signal[i]),
            "confidence":      float(confidence[i]),
        }
        for i in range(n)
    ]
