"""
models/lstm_predictor.py
--------------------------
Trains and evaluates a Keras LSTM classifier (DOWN/HOLD/UP) on the full
3D sequences produced by data/preprocessor.py — this is the model that
actually uses the temporal structure (X.shape == (N, seq_len, n_features)),
unlike models/xgboost_classifier.py which only sees the last timestep.

Architecture:
    Input (seq_len, n_features)
      -> LSTM(64, return_sequences=True) -> Dropout
      -> LSTM(32)                        -> Dropout
      -> Dense(16, relu)
      -> Dense(3, softmax)               [DOWN, HOLD, UP]

Usage:
    from models.lstm_predictor import build_lstm_model, train_lstm, predict_lstm, evaluate_lstm
    from data.preprocessor import preprocess

    data  = preprocess("RELIANCE.NS", "2020-01-01", "2024-01-01")
    model = build_lstm_model(data.seq_config.sequence_length, data.n_features)
    model, history = train_lstm(model, data)
    probs = predict_lstm(model, data.X_test)   # (N, 3) [DOWN, HOLD, UP]
"""

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config.settings import (
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_PATIENCE,
    RANDOM_SEED,
)
from utils.logger import logger

LABEL_NAMES = ["DOWN", "HOLD", "UP"]


def build_lstm_model(
    sequence_length: int,
    n_features:      int,
    lstm_units:      tuple[int, int] = (64, 32),
    dropout:         float           = 0.3,
    learning_rate:   float           = 1e-3,
):
    """
    Builds an unfitted Keras LSTM classifier for 3-class prediction.

    Args:
        sequence_length: Must match data.seq_config.sequence_length
        n_features:      Must match data.n_features
        lstm_units:      (first_layer_units, second_layer_units)
        dropout:         Dropout rate after each LSTM layer
        learning_rate:   Adam optimizer learning rate

    Returns:
        Compiled, unfitted keras.Model.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(RANDOM_SEED)

    model = keras.Sequential([
        keras.Input(shape=(sequence_length, n_features)),
        layers.LSTM(lstm_units[0], return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(lstm_units[1]),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss       = "sparse_categorical_crossentropy",
        metrics     = ["accuracy"],
    )
    return model


def train_lstm(
    model,
    data,
    epochs:            int   = LSTM_EPOCHS,
    batch_size:        int   = LSTM_BATCH_SIZE,
    patience:          int   = LSTM_PATIENCE,
    use_class_weights: bool  = True,
    verbose:           int   = 0,
):
    """
    Trains `model` on data.X_train/y_train, early-stopping on val_loss
    against data.X_val/y_val.

    Args:
        model:              Unfitted/compiled keras.Model (from build_lstm_model)
        data:               PreparedData (from data.preprocessor.preprocess)
        epochs:             Max epochs (settings.LSTM_EPOCHS)
        batch_size:         settings.LSTM_BATCH_SIZE
        patience:           EarlyStopping patience (settings.LSTM_PATIENCE)
        use_class_weights:  Apply data.class_weights (HOLD is usually the
                            majority class — without this the model tends
                            to collapse to always predicting HOLD)
        verbose:            Keras verbosity (0=silent, 1=progress bar)

    Returns:
        (fitted_model, keras.callbacks.History)
    """
    from tensorflow import keras

    class_weight = None
    if use_class_weights:
        class_weight = {int(k): float(v) for k, v in data.class_weights.items()}

    has_val = len(data.X_val) > 0
    if not has_val:
        logger.warning(
            "No validation data — training without early stopping "
            f"for the full {epochs} epochs. Split ratios may need adjusting."
        )

    callbacks = []
    if has_val:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ))

    logger.info(
        f"Training LSTM: {data.X_train.shape[0]} samples, "
        f"up to {epochs} epochs (patience={patience})"
    )

    history = model.fit(
        data.X_train, data.y_train,
        validation_data = (data.X_val, data.y_val) if has_val else None,
        epochs           = epochs,
        batch_size        = batch_size,
        class_weight       = class_weight,
        callbacks           = callbacks,
        verbose              = verbose,
    )

    epochs_trained = len(history.history["loss"])
    train_acc      = history.history["accuracy"][-1]
    val_acc        = history.history.get("val_accuracy", [None])[-1]

    logger.success(
        f"LSTM trained: {epochs_trained} epochs | "
        f"train_acc={train_acc:.3f}" +
        (f" | val_acc={val_acc:.3f}" if val_acc is not None else "")
    )

    return model, history


def predict_lstm(model, X: np.ndarray) -> np.ndarray:
    """
    Returns class probabilities for [DOWN, HOLD, UP].

    Args:
        model: Fitted keras.Model
        X:     3D sequence array (N, seq_len, n_features)

    Returns:
        (N, 3) array of probabilities, columns ordered [DOWN, HOLD, UP].
    """
    return model.predict(X, verbose=0)


def evaluate_lstm(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluates the model on a held-out split.

    Returns:
        {"accuracy": float, "f1_macro": float, "report": str}
    """
    probs  = predict_lstm(model, X)
    y_pred = probs.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "report":   classification_report(
            y, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0
        ),
    }
    logger.info(
        f"LSTM eval: accuracy={metrics['accuracy']:.3f}  "
        f"f1_macro={metrics['f1_macro']:.3f}"
    )
    return metrics
