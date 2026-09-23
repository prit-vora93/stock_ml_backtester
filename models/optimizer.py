"""
models/optimizer.py
---------------------
Lightweight hyperparameter search for the XGBoost classifier, the LSTM
classifier, and the ensemble combination weights.

Design notes:
    - XGBoost search is a full grid search — training is cheap (seconds),
      so exhaustive search over a small grid is practical.
    - LSTM search is capped at a small number of trials (default 4) since
      each trial is a full training run — this is a "good enough" search,
      not exhaustive. Callers who want more should pass a bigger
      param_grid and accept the runtime cost.
    - Ensemble weight search needs no retraining at all — it just
      recombines already-computed LSTM/XGBoost probabilities, so it's a
      full grid search over weight/confidence combinations and is fast.

All three search functions select against the VALIDATION split
(data.X_val/y_val), never the test split — the test split stays untouched
until final evaluation, consistent with the rest of this pipeline's
no-leakage discipline.

Usage:
    from models.optimizer import optimize_xgboost, optimize_lstm, optimize_ensemble_weights

    xgb_result = optimize_xgboost(data)
    best_xgb_model = xgb_result["best_model"]
"""

import itertools
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score

from config.settings import RANDOM_SEED
from models.ensemble import EnsembleConfig, predict_ensemble
from models.lstm_predictor import build_lstm_model, train_lstm, predict_lstm
from models.xgboost_classifier import build_xgboost_model, train_xgboost, predict_xgboost, _last_timestep
from utils.logger import logger

DEFAULT_XGB_GRID: dict[str, list] = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 4, 6],
    "learning_rate": [0.01, 0.05, 0.1],
}

DEFAULT_LSTM_GRID: dict[str, list] = {
    "lstm_units": [(64, 32), (32, 16)],
    "dropout":    [0.2, 0.3, 0.4],
}

DEFAULT_WEIGHT_GRID = [round(w, 2) for w in np.arange(0.0, 1.01, 0.1)]
DEFAULT_CONFIDENCE_GRID = [round(c, 2) for c in np.arange(0.4, 0.81, 0.05)]


def optimize_xgboost(
    data,
    param_grid: Optional[dict[str, list]] = None,
    scoring:    str = "f1_macro",
) -> dict:
    """
    Grid search over XGBoost hyperparameters, scored on data.X_val/y_val.

    Args:
        data:       PreparedData
        param_grid: dict of param_name -> list of values to try
                    (defaults to DEFAULT_XGB_GRID)
        scoring:    "f1_macro" or "accuracy"

    Returns:
        {
            "best_params": dict,
            "best_score":  float,
            "best_model":  fitted XGBClassifier,
            "all_results": list of {"params": dict, "score": float},
        }
    """
    grid = param_grid or DEFAULT_XGB_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    logger.info(f"Optimizing XGBoost: {len(combos)} combinations, scoring={scoring}")

    X_val = _last_timestep(data.X_val)
    all_results = []
    best_score  = -np.inf
    best_params = None
    best_model  = None

    for combo in combos:
        params = dict(zip(keys, combo))
        model  = build_xgboost_model(**params)
        model  = train_xgboost(data, model=model)

        y_pred = model.predict(X_val)
        score  = (
            f1_score(data.y_val, y_pred, average="macro", zero_division=0)
            if scoring == "f1_macro"
            else (y_pred == data.y_val).mean()
        )

        all_results.append({"params": params, "score": float(score)})
        if score > best_score:
            best_score, best_params, best_model = score, params, model

    logger.success(f"XGBoost optimization done: best_score={best_score:.3f}  best_params={best_params}")

    return {
        "best_params": best_params,
        "best_score":  best_score,
        "best_model":  best_model,
        "all_results": all_results,
    }


def optimize_lstm(
    data,
    param_grid: Optional[dict[str, list]] = None,
    max_trials: int = 4,
    epochs:     int = 20,
    patience:   int = 5,
    scoring:    str = "f1_macro",
) -> dict:
    """
    Small random search over LSTM hyperparameters, scored on
    data.X_val/y_val. Capped at `max_trials` since each trial is a full
    training run — this is NOT exhaustive grid search.

    Args:
        data:       PreparedData
        param_grid: dict of param_name -> list of values to try
                    (defaults to DEFAULT_LSTM_GRID)
        max_trials: Max number of (random) combinations to try
        epochs:     Epochs per trial (kept low for search speed —
                    use settings.LSTM_EPOCHS for the final model)
        patience:   EarlyStopping patience per trial
        scoring:    "f1_macro" or "accuracy"

    Returns:
        {
            "best_params": dict,
            "best_score":  float,
            "best_model":  fitted keras.Model,
            "all_results": list of {"params": dict, "score": float},
        }
    """
    grid = param_grid or DEFAULT_LSTM_GRID
    keys = list(grid.keys())
    all_combos = list(itertools.product(*[grid[k] for k in keys]))

    rng = np.random.default_rng(RANDOM_SEED)
    if len(all_combos) > max_trials:
        chosen_idx = rng.choice(len(all_combos), size=max_trials, replace=False)
        combos = [all_combos[i] for i in chosen_idx]
    else:
        combos = all_combos

    logger.info(f"Optimizing LSTM: {len(combos)}/{len(all_combos)} combinations, scoring={scoring}")

    seq_len    = data.X_train.shape[1]
    n_features = data.X_train.shape[2]

    all_results = []
    best_score  = -np.inf
    best_params = None
    best_model  = None

    for combo in combos:
        params = dict(zip(keys, combo))
        model  = build_lstm_model(seq_len, n_features, **params)
        model, _ = train_lstm(data=data, model=model, epochs=epochs, patience=patience)

        probs  = predict_lstm(model, data.X_val)
        y_pred = probs.argmax(axis=1)
        score  = (
            f1_score(data.y_val, y_pred, average="macro", zero_division=0)
            if scoring == "f1_macro"
            else (y_pred == data.y_val).mean()
        )

        all_results.append({"params": params, "score": float(score)})
        if score > best_score:
            best_score, best_params, best_model = score, params, model

    logger.success(f"LSTM optimization done: best_score={best_score:.3f}  best_params={best_params}")

    return {
        "best_params": best_params,
        "best_score":  best_score,
        "best_model":  best_model,
        "all_results": all_results,
    }


def optimize_ensemble_weights(
    lstm_probs:      np.ndarray,
    xgb_probs:       np.ndarray,
    y_true:          np.ndarray,
    weight_grid:     Optional[list[float]] = None,
    confidence_grid: Optional[list[float]] = None,
    scoring:         str = "f1_macro",
) -> dict:
    """
    Full grid search over ensemble (lstm_weight, min_confidence) combos.
    No retraining needed — just recombines already-computed probabilities,
    so this is fast even with a fine-grained grid.

    Scoring uses `predicted_class` (raw argmax), not confidence-gated
    `signal` — see models.ensemble.evaluate_ensemble for why.

    Args:
        lstm_probs:      (N, 3) from predict_lstm() on the VALIDATION split
        xgb_probs:       (N, 3) from predict_xgboost() on the same split
        y_true:          (N,) true labels for that split
        weight_grid:     lstm_weight values to try (xgb_weight = 1 - lstm_weight)
                         (defaults to DEFAULT_WEIGHT_GRID: 0.0-1.0 step 0.1)
        confidence_grid: min_confidence values to try
                         (defaults to DEFAULT_CONFIDENCE_GRID: 0.4-0.8 step 0.05)
        scoring:         "f1_macro" or "accuracy"

    Returns:
        {
            "best_config": EnsembleConfig,
            "best_score":  float,
            "all_results": list of {"lstm_weight": float, "min_confidence": float, "score": float},
        }
    """
    weights      = weight_grid or DEFAULT_WEIGHT_GRID
    confidences  = confidence_grid or DEFAULT_CONFIDENCE_GRID

    logger.info(
        f"Optimizing ensemble weights: {len(weights)} weights x "
        f"{len(confidences)} confidence thresholds"
    )

    all_results = []
    best_score  = -np.inf
    best_config = None

    for lstm_weight in weights:
        config = EnsembleConfig(
            lstm_weight=lstm_weight, xgb_weight=1.0 - lstm_weight, min_confidence=0.0
        )
        combined = config.lstm_weight * lstm_probs + config.xgb_weight * xgb_probs
        y_pred   = combined.argmax(axis=1)
        score    = (
            f1_score(y_true, y_pred, average="macro", zero_division=0)
            if scoring == "f1_macro"
            else (y_pred == y_true).mean()
        )

        # min_confidence doesn't change predicted_class/accuracy (it only
        # gates the "signal" output), so we score it once per weight pair
        # and record every confidence value for completeness/inspection.
        for min_confidence in confidences:
            all_results.append({
                "lstm_weight":    lstm_weight,
                "min_confidence": min_confidence,
                "score":          float(score),
            })

        if score > best_score:
            best_score  = score
            best_config = EnsembleConfig(
                lstm_weight=lstm_weight, xgb_weight=1.0 - lstm_weight,
                min_confidence=confidences[len(confidences) // 2],
            )

    logger.success(
        f"Ensemble weight optimization done: best_score={best_score:.3f}  "
        f"lstm_weight={best_config.lstm_weight:.2f}  xgb_weight={best_config.xgb_weight:.2f}"
    )

    return {
        "best_config": best_config,
        "best_score":  best_score,
        "all_results": all_results,
    }
