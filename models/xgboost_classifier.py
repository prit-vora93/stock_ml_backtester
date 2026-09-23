"""
models/xgboost_classifier.py
------------------------------
Trains and evaluates an XGBoost multi-class classifier (DOWN/HOLD/UP)
on the tabular snapshot of each sequence's most recent day.

Why the "last timestep" instead of the full sequence?
    PreparedData.X_train is 3D: (samples, sequence_length, features) —
    built for the LSTM. XGBoost (like all gradient-boosted tree models)
    takes 2D tabular input. Flattening the full sequence
    (sequence_length * n_features columns, e.g. 60*41=2460) would
    massively outnumber the few hundred training samples typical here
    and overfit badly. Using only the most recent day's feature vector
    (X[:, -1, :]) keeps XGBoost doing what it's good at — today's
    snapshot — while the LSTM handles the temporal pattern; the
    ensemble in models/ensemble.py combines both views.

Usage:
    from models.xgboost_classifier import train_xgboost, predict_xgboost, evaluate_xgboost
    from data.preprocessor import preprocess

    data  = preprocess("RELIANCE.NS", "2020-01-01", "2024-01-01")
    model = train_xgboost(data)
    probs = predict_xgboost(model, data.X_test)   # (N, 3) [DOWN, HOLD, UP]
"""

from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config.settings import (
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    RANDOM_SEED,
)
from utils.logger import logger

LABEL_NAMES = ["DOWN", "HOLD", "UP"]

# Stop training if validation mlogloss hasn't improved for this many rounds.
DEFAULT_EARLY_STOPPING_ROUNDS = 20


def _last_timestep(X: np.ndarray) -> np.ndarray:
    """
    Extracts the most recent day's feature vector from a 3D sequence array.
    (N, seq_len, n_features) -> (N, n_features). Already-2D input passes through.
    """
    if X.ndim == 3:
        return X[:, -1, :]
    return X


def build_xgboost_model(
    n_estimators:          int   = XGB_N_ESTIMATORS,
    max_depth:             int   = XGB_MAX_DEPTH,
    learning_rate:         float = XGB_LEARNING_RATE,
    early_stopping_rounds: Optional[int] = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> xgb.XGBClassifier:
    """
    Builds an unfitted XGBoost multi-class classifier using settings.py
    defaults unless overridden — the same knobs models/optimizer.py tunes.
    """
    return xgb.XGBClassifier(
        n_estimators          = n_estimators,
        max_depth              = max_depth,
        learning_rate          = learning_rate,
        objective               = "multi:softprob",
        num_class               = 3,
        eval_metric              = "mlogloss",
        early_stopping_rounds   = early_stopping_rounds,
        random_state             = RANDOM_SEED,
        n_jobs                   = -1,
    )


def train_xgboost(
    data,
    model:             Optional[xgb.XGBClassifier] = None,
    use_class_weights: bool                         = True,
) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier on data.X_train/y_train (last timestep
    only), early-stopping against data.X_val/y_val.

    Args:
        data:              PreparedData (from data.preprocessor.preprocess)
        model:             Optional pre-built XGBClassifier
                           (else built from settings.py defaults)
        use_class_weights: Apply data.class_weights as per-sample weights,
                           same imbalance handling the LSTM path uses.

    Returns:
        Fitted XGBClassifier.
    """
    if model is None:
        model = build_xgboost_model()

    X_train = _last_timestep(data.X_train)
    X_val   = _last_timestep(data.X_val)

    sample_weight = None
    if use_class_weights:
        sample_weight = np.array(
            [data.class_weights.get(int(c), 1.0) for c in data.y_train]
        )

    logger.info(
        f"Training XGBoost: {X_train.shape[0]} samples x "
        f"{X_train.shape[1]} features"
    )

    model.fit(
        X_train, data.y_train,
        sample_weight = sample_weight,
        eval_set       = [(X_val, data.y_val)],
        verbose         = False,
    )

    train_acc = accuracy_score(data.y_train, model.predict(X_train))
    val_acc   = accuracy_score(data.y_val,   model.predict(X_val))
    logger.success(
        f"XGBoost trained: train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
        f"best_iteration={getattr(model, 'best_iteration', model.n_estimators)}"
    )

    return model


def predict_xgboost(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """
    Returns class probabilities for [DOWN, HOLD, UP].

    Args:
        model: Fitted XGBClassifier
        X:     3D sequence array (N, seq_len, n_features) or 2D (N, n_features)

    Returns:
        (N, 3) array of probabilities, columns ordered [DOWN, HOLD, UP].
    """
    return model.predict_proba(_last_timestep(X))


def evaluate_xgboost(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluates the model on a held-out split.

    Returns:
        {"accuracy": float, "f1_macro": float, "report": str}
    """
    X2d    = _last_timestep(X)
    y_pred = model.predict(X2d)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "report":   classification_report(
            y, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0
        ),
    }
    logger.info(
        f"XGBoost eval: accuracy={metrics['accuracy']:.3f}  "
        f"f1_macro={metrics['f1_macro']:.3f}"
    )
    return metrics


def get_feature_importance(
    model:         xgb.XGBClassifier,
    feature_names: list[str],
    top_n:         int = 20,
) -> list[tuple[str, float]]:
    """
    Returns the top_n most important features as (name, importance),
    sorted descending. Useful for sanity-checking the model isn't
    leaning entirely on one noisy column.
    """
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)
    return pairs[:top_n]
