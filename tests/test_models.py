"""
tests/test_models.py
---------------------
Tests for the models/ package: xgboost_classifier, lstm_predictor,
ensemble, model_saver, optimizer.

All tests use small, synthetic, in-memory PreparedData objects (no DB,
no network) with a deliberately learnable signal (the last timestep's
first feature crossing a threshold), so training is fast and accuracy
assertions are meaningful rather than arbitrary.

Fixtures are module-scoped where the underlying model is expensive to
train (LSTM), so each is only trained once and reused across tests that
just need "a trained model to call predict/evaluate/save on".

Run:
    pytest tests/test_models.py -v
"""

import numpy as np
import pytest

from data.preprocessor import PreparedData

from models.xgboost_classifier import (
    build_xgboost_model, train_xgboost, predict_xgboost,
    evaluate_xgboost, get_feature_importance,
)
from models.lstm_predictor import (
    build_lstm_model, train_lstm, predict_lstm, evaluate_lstm,
)
from models.ensemble import (
    EnsembleConfig, combine_probabilities, probs_to_signals,
    predict_ensemble, evaluate_ensemble, predictions_to_db_rows,
)
from models import model_saver
from models.optimizer import (
    optimize_xgboost, optimize_ensemble_weights, optimize_lstm,
)


N_TRAIN, N_VAL, N_TEST = 150, 40, 40
SEQ_LEN, N_FEATURES    = 10, 5


def _make_split(n, seed):
    """A sequence array whose label is fully determined by the last
    timestep's first feature — an easy, learnable synthetic signal."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, SEQ_LEN, N_FEATURES)).astype("float32")
    y = np.where(X[:, -1, 0] > 0.66, 2, np.where(X[:, -1, 0] < 0.33, 0, 1)).astype(int)
    return X, y


@pytest.fixture(scope="module")
def synthetic_data() -> PreparedData:
    X_train, y_train = _make_split(N_TRAIN, seed=1)
    X_val,   y_val   = _make_split(N_VAL,   seed=2)
    X_test,  y_test  = _make_split(N_TEST,  seed=3)

    return PreparedData(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        class_weights={0: 1.0, 1: 1.0, 2: 1.0},
        scaler=None,
        feature_names=[f"f{i}" for i in range(N_FEATURES)],
        symbol="SYN.NS", n_features=N_FEATURES, sequence_len=SEQ_LEN,
    )


@pytest.fixture(scope="module")
def trained_xgb(synthetic_data):
    return train_xgboost(synthetic_data)


@pytest.fixture(scope="module")
def trained_lstm(synthetic_data):
    model = build_lstm_model(SEQ_LEN, N_FEATURES, lstm_units=(16, 8))
    model, history = train_lstm(model, synthetic_data, epochs=15, patience=5)
    return model, history


# ═════════════════════════════════════════════════════════════════════════════
# TestXGBoostClassifier
# ═════════════════════════════════════════════════════════════════════════════

class TestXGBoostClassifier:

    def test_train_returns_fitted_model(self, trained_xgb):
        assert trained_xgb is not None
        assert hasattr(trained_xgb, "predict_proba")

    def test_predict_shape_and_probabilities_sum_to_one(self, synthetic_data, trained_xgb):
        probs = predict_xgboost(trained_xgb, synthetic_data.X_test)
        assert probs.shape == (N_TEST, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_accepts_2d_input_directly(self, synthetic_data, trained_xgb):
        """XGBoost should also accept an already-2D (last-timestep) array,
        not just the full 3D sequence — _last_timestep must pass 2D through."""
        X_2d = synthetic_data.X_test[:, -1, :]
        probs = predict_xgboost(trained_xgb, X_2d)
        assert probs.shape == (N_TEST, 3)

    def test_learns_the_synthetic_signal(self, synthetic_data, trained_xgb):
        metrics = evaluate_xgboost(trained_xgb, synthetic_data.X_test, synthetic_data.y_test)
        assert metrics["accuracy"] > 0.9, (
            f"XGBoost should easily learn this deterministic signal, got {metrics['accuracy']}"
        )

    def test_feature_importance_identifies_driving_feature(self, synthetic_data, trained_xgb):
        top = get_feature_importance(trained_xgb, synthetic_data.feature_names, top_n=3)
        assert top[0][0] == "f0", "f0 drives the label and should be the top feature"

    def test_build_xgboost_model_uses_settings_defaults(self):
        from config.settings import XGB_N_ESTIMATORS, XGB_MAX_DEPTH
        model = build_xgboost_model()
        assert model.n_estimators == XGB_N_ESTIMATORS
        assert model.max_depth == XGB_MAX_DEPTH


# ═════════════════════════════════════════════════════════════════════════════
# TestLSTMPredictor
# ═════════════════════════════════════════════════════════════════════════════

class TestLSTMPredictor:

    def test_build_model_output_shape(self):
        model = build_lstm_model(SEQ_LEN, N_FEATURES)
        dummy = np.random.rand(4, SEQ_LEN, N_FEATURES).astype("float32")
        out = model.predict(dummy, verbose=0)
        assert out.shape == (4, 3)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4), "softmax output must sum to 1"

    def test_train_lstm_returns_history_with_expected_keys(self, trained_lstm):
        _, history = trained_lstm
        assert "loss" in history.history
        assert "accuracy" in history.history
        assert "val_loss" in history.history

    def test_predict_shape_and_probabilities_sum_to_one(self, synthetic_data, trained_lstm):
        model, _ = trained_lstm
        probs = predict_lstm(model, synthetic_data.X_test)
        assert probs.shape == (N_TEST, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    def test_evaluate_returns_metrics(self, synthetic_data, trained_lstm):
        model, _ = trained_lstm
        metrics = evaluate_lstm(model, synthetic_data.X_test, synthetic_data.y_test)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "DOWN" in metrics["report"]

    def test_train_handles_empty_validation_split(self):
        """No validation rows shouldn't crash — just trains without early stopping."""
        X_train, y_train = _make_split(30, seed=10)
        empty_X = np.empty((0, SEQ_LEN, N_FEATURES), dtype="float32")
        empty_y = np.array([], dtype=int)

        data = PreparedData(
            X_train=X_train, y_train=y_train,
            X_val=empty_X, y_val=empty_y,
            X_test=empty_X, y_test=empty_y,
            class_weights={0: 1.0, 1: 1.0, 2: 1.0}, scaler=None,
            feature_names=[f"f{i}" for i in range(N_FEATURES)],
            symbol="SYN.NS", n_features=N_FEATURES, sequence_len=SEQ_LEN,
        )
        model = build_lstm_model(SEQ_LEN, N_FEATURES, lstm_units=(8, 4))
        model, history = train_lstm(model=model, data=data, epochs=2, patience=2)
        assert len(history.history["loss"]) == 2   # ran full epochs, no early stop possible


# ═════════════════════════════════════════════════════════════════════════════
# TestEnsemble
# ═════════════════════════════════════════════════════════════════════════════

class TestEnsemble:

    def test_combine_probabilities_weighted_average(self):
        lstm_probs = np.array([[0.2, 0.3, 0.5]])
        xgb_probs  = np.array([[0.6, 0.2, 0.2]])
        cfg = EnsembleConfig(lstm_weight=0.6, xgb_weight=0.4, min_confidence=0.0)

        combined = combine_probabilities(lstm_probs, xgb_probs, cfg)
        expected = 0.6 * lstm_probs + 0.4 * xgb_probs
        assert np.allclose(combined, expected)

    def test_combine_probabilities_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            combine_probabilities(np.zeros((5, 3)), np.zeros((4, 3)))

    def test_weights_normalized_when_not_summing_to_one(self):
        cfg = EnsembleConfig(lstm_weight=0.8, xgb_weight=0.8)
        assert cfg.lstm_weight == pytest.approx(0.5)
        assert cfg.xgb_weight == pytest.approx(0.5)

    def test_confidence_gating_forces_hold(self):
        # Uniform (1/3, 1/3, 1/3) -> confidence exactly 1/3, below any
        # reasonable min_confidence -> must be gated to HOLD.
        uniform_probs = np.full((5, 3), 1 / 3)
        cfg = EnsembleConfig(min_confidence=0.6)
        result = probs_to_signals(uniform_probs, cfg)
        assert (result["signal"] == "HOLD").all()

    def test_high_confidence_up_maps_to_buy(self):
        probs = np.array([[0.05, 0.05, 0.90]])
        cfg = EnsembleConfig(min_confidence=0.5)
        result = probs_to_signals(probs, cfg)
        assert result["signal"][0] == "BUY"
        assert result["predicted_class"][0] == 2

    def test_high_confidence_down_maps_to_sell(self):
        probs = np.array([[0.90, 0.05, 0.05]])
        cfg = EnsembleConfig(min_confidence=0.5)
        result = probs_to_signals(probs, cfg)
        assert result["signal"][0] == "SELL"
        assert result["predicted_class"][0] == 0

    def test_predict_ensemble_end_to_end(self, synthetic_data, trained_xgb, trained_lstm):
        lstm_model, _ = trained_lstm
        lstm_probs = predict_lstm(lstm_model, synthetic_data.X_test)
        xgb_probs  = predict_xgboost(trained_xgb, synthetic_data.X_test)

        result = predict_ensemble(lstm_probs, xgb_probs)
        assert result["probs"].shape == (N_TEST, 3)
        assert len(result["signal"]) == N_TEST
        assert set(result["signal"]) <= {"BUY", "HOLD", "SELL"}

        metrics = evaluate_ensemble(lstm_probs, xgb_probs, synthetic_data.y_test)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["pct_actionable"] <= 1.0

    def test_predictions_to_db_rows_shape_and_keys(self, synthetic_data, trained_xgb, trained_lstm):
        lstm_model, _ = trained_lstm
        lstm_probs = predict_lstm(lstm_model, synthetic_data.X_test)
        xgb_probs  = predict_xgboost(trained_xgb, synthetic_data.X_test)
        result = predict_ensemble(lstm_probs, xgb_probs)

        rows = predictions_to_db_rows(lstm_probs, xgb_probs, result)
        assert len(rows) == N_TEST

        expected_keys = {
            "lstm_prob_down", "lstm_prob_hold", "lstm_prob_up",
            "xgb_prob_down", "xgb_prob_hold", "xgb_prob_up",
            "ensemble_signal", "confidence",
        }
        assert set(rows[0].keys()) == expected_keys
        assert rows[0]["ensemble_signal"] in {"BUY", "HOLD", "SELL"}


# ═════════════════════════════════════════════════════════════════════════════
# TestModelSaver
# ═════════════════════════════════════════════════════════════════════════════

class TestModelSaver:

    def test_xgboost_save_load_roundtrip(self, trained_xgb, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr(model_saver, "MODELS_DIR", str(tmp_path))

        path = model_saver.save_xgboost(trained_xgb, "SYN.NS", metadata={"n_features": N_FEATURES})
        assert (tmp_path / "xgb_SYN_NS.pkl").exists()
        assert (tmp_path / "xgb_SYN_NS_meta.json").exists()

        loaded = model_saver.load_xgboost("SYN.NS")
        assert loaded is not None

        original_probs = predict_xgboost(trained_xgb, synthetic_data.X_test)
        loaded_probs   = predict_xgboost(loaded, synthetic_data.X_test)
        assert np.allclose(original_probs, loaded_probs, atol=1e-5)

    def test_lstm_save_load_roundtrip(self, trained_lstm, synthetic_data, tmp_path, monkeypatch):
        monkeypatch.setattr(model_saver, "MODELS_DIR", str(tmp_path))
        model, _ = trained_lstm

        path = model_saver.save_lstm(model, "SYN.NS", metadata={"n_features": N_FEATURES})
        assert (tmp_path / "lstm_SYN_NS.keras").exists()

        loaded = model_saver.load_lstm("SYN.NS")
        assert loaded is not None

        original_probs = predict_lstm(model, synthetic_data.X_test)
        loaded_probs   = predict_lstm(loaded, synthetic_data.X_test)
        assert np.allclose(original_probs, loaded_probs, atol=1e-5)

    def test_ensemble_config_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_saver, "MODELS_DIR", str(tmp_path))

        config = {"lstm_weight": 0.6, "xgb_weight": 0.4, "min_confidence": 0.6}
        model_saver.save_ensemble_config("SYN.NS", config)

        loaded = model_saver.load_ensemble_config("SYN.NS")
        assert loaded is not None
        assert loaded["lstm_weight"] == 0.6
        assert loaded["xgb_weight"] == 0.4

    def test_load_missing_model_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_saver, "MODELS_DIR", str(tmp_path))
        assert model_saver.load_xgboost("NOPE.NS") is None
        assert model_saver.load_ensemble_config("NOPE.NS") is None


# ═════════════════════════════════════════════════════════════════════════════
# TestOptimizer
# Kept intentionally small/fast — real search behavior, tiny grids.
# ═════════════════════════════════════════════════════════════════════════════

class TestOptimizer:

    def test_optimize_xgboost_returns_best_model(self, synthetic_data):
        result = optimize_xgboost(
            synthetic_data,
            param_grid={"n_estimators": [50, 100], "max_depth": [3, 4], "learning_rate": [0.1]},
        )
        assert result["best_model"] is not None
        assert len(result["all_results"]) == 4
        assert result["best_score"] > 0.5

    def test_optimize_ensemble_weights_returns_best_config(self, synthetic_data, trained_xgb, trained_lstm):
        lstm_model, _ = trained_lstm
        lstm_probs = predict_lstm(lstm_model, synthetic_data.X_val)
        xgb_probs  = predict_xgboost(trained_xgb, synthetic_data.X_val)

        result = optimize_ensemble_weights(
            lstm_probs, xgb_probs, synthetic_data.y_val,
            weight_grid=[0.0, 0.5, 1.0], confidence_grid=[0.5, 0.6],
        )
        assert result["best_config"] is not None
        assert 0.0 <= result["best_config"].lstm_weight <= 1.0
        assert result["best_score"] >= 0.0

    def test_optimize_lstm_single_trial(self, synthetic_data):
        result = optimize_lstm(
            synthetic_data,
            param_grid={"lstm_units": [(8, 4)], "dropout": [0.2]},
            max_trials=1, epochs=3, patience=2,
        )
        assert result["best_model"] is not None
        assert len(result["all_results"]) == 1
