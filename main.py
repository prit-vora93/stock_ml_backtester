"""
main.py
-------
Day 4 test script.
Tests the full preprocessing pipeline:
    feature engineering → normalize → split → sequences → class weights

Run:
    python main.py
"""

from data.preprocessor import (
    preprocess,
    preprocess_all_stocks,
    save_scaler,
    load_scaler,
    SequenceConfig,
)
from api.database  import test_connection, create_tables
from utils.logger  import logger


def run_day4():

    logger.info("=" * 60)
    logger.info("  Day 4 — Preprocessing Pipeline")
    logger.info("=" * 60)

    # ── Pre-check ─────────────────────────────────────────────────────────────
    logger.info("\n🔌 Checking database...")
    if not test_connection():
        logger.error("Database not ready. Run: sudo systemctl start postgresql")
        return
    create_tables()

    # ── Test 1: Single stock default config ───────────────────────────────────
    logger.info("\n📊 Test 1: Preprocess RELIANCE.NS (default config)")
    data = preprocess("RELIANCE.NS", "2020-01-01", "2024-01-01")

    if data is None:
        logger.error("Test 1 FAILED — preprocess returned None")
        return

    logger.success(f"X_train shape  : {data.X_train.shape}")
    logger.success(f"X_val shape    : {data.X_val.shape}")
    logger.success(f"X_test shape   : {data.X_test.shape}")
    logger.success(f"Features       : {data.n_features}")
    logger.success(f"Memory         : {data.memory_mb:.1f} MB")
    logger.success(f"Version        : {data.preprocessor_version}")

    # ── Test 2: Verify scaling range ──────────────────────────────────────────
    logger.info("\n🔢 Test 2: Verify feature scaling")

    import numpy as np
    x_min = data.X_train.min()
    x_max = data.X_train.max()

    logger.info(f"  X_train min : {x_min:.6f}  (should be ~0.0)")
    logger.info(f"  X_train max : {x_max:.6f}  (should be ~1.0)")

    assert x_min >= -0.01, f"Min too low: {x_min}"
    assert x_max <=  1.01, f"Max too high: {x_max}"
    logger.success("Scaling range correct ✅")

    # ── Test 3: Verify no NaN or Inf ──────────────────────────────────────────
    logger.info("\n🔍 Test 3: Check for NaN / Inf in sequences")

    for name, X in [("X_train", data.X_train),
                    ("X_val",   data.X_val),
                    ("X_test",  data.X_test)]:
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        logger.info(f"  {name}: NaN={nan_count}  Inf={inf_count}")
        assert nan_count == 0, f"NaN found in {name}"
        assert inf_count == 0, f"Inf found in {name}"

    logger.success("No NaN or Inf values ✅")

    # ── Test 4: Verify sequence shape ─────────────────────────────────────────
    logger.info("\n📐 Test 4: Verify sequence dimensions")

    seq_len    = data.seq_config.sequence_length
    n_features = data.n_features

    assert data.X_train.ndim == 3,                  "X_train should be 3D"
    assert data.X_train.shape[1] == seq_len,         f"Wrong seq length: {data.X_train.shape[1]}"
    assert data.X_train.shape[2] == n_features,      f"Wrong features: {data.X_train.shape[2]}"
    assert len(data.X_train) == len(data.y_train),   "X/y train length mismatch"
    assert len(data.X_val)   == len(data.y_val),     "X/y val length mismatch"
    assert len(data.X_test)  == len(data.y_test),    "X/y test length mismatch"

    logger.success(f"Shape: ({len(data.X_train)}, {seq_len}, {n_features}) ✅")

    # ── Test 5: Verify class weights ──────────────────────────────────────────
    logger.info("\n⚖️  Test 5: Class weights")

    for cls, name in [(0, "DOWN"), (1, "HOLD"), (2, "UP")]:
        w = data.class_weights.get(cls, 0)
        logger.info(f"  {name}: {w:.3f}")

    assert 0 in data.class_weights, "Missing DOWN weight"
    assert 1 in data.class_weights, "Missing HOLD weight"
    assert 2 in data.class_weights, "Missing UP weight"
    assert data.class_weights[1] < data.class_weights[0], \
        "HOLD should have lower weight than DOWN"
    logger.success("Class weights correct ✅")

    # ── Test 6: Save and reload scaler ────────────────────────────────────────
    logger.info("\n💾 Test 6: Save and reload scaler")

    path    = save_scaler(data.scaler, "RELIANCE.NS", data)
    scaler2 = load_scaler("RELIANCE.NS")

    assert scaler2 is not None, "Failed to reload scaler"

    # Verify reloaded scaler produces identical output
    sample   = data.X_train[0][0].reshape(1, -1)
    out1     = data.scaler.transform(sample)
    out2     = scaler2.transform(sample)
    assert np.allclose(out1, out2), "Reloaded scaler produces different output"
    logger.success("Scaler save/load identical ✅")

    # ── Test 7: Custom SequenceConfig ─────────────────────────────────────────
    logger.info("\n⚙️  Test 7: Custom SequenceConfig (stride=5)")

    cfg   = SequenceConfig(sequence_length=60, prediction_horizon=1, stride=5)
    data2 = preprocess(
        "RELIANCE.NS", "2020-01-01", "2024-01-01",
        seq_config=cfg
    )

    if data2:
        expected_approx = len(data.X_train) // 5
        logger.success(
            f"Stride=1 sequences: {len(data.X_train)} | "
            f"Stride=5 sequences: {len(data2.X_train)} "
            f"(~{len(data2.X_train)/len(data.X_train):.0%} of stride=1)"
        )
        logger.success("SequenceConfig working ✅")

    # ── Test 8: Feature set filtering ─────────────────────────────────────────
    logger.info("\n🔬 Test 8: Feature set filtering")

    data_tech = preprocess(
        "RELIANCE.NS", "2020-01-01", "2024-01-01",
        feature_set="technical"
    )
    data_all = preprocess(
        "RELIANCE.NS", "2020-01-01", "2024-01-01",
        feature_set="all"
    )

    if data_tech and data_all:
        logger.success(
            f"technical only : {data_tech.n_features} features | "
            f"all features   : {data_all.n_features} features"
        )
        assert data_tech.n_features < data_all.n_features, \
            "Technical subset should have fewer features than full set"
        logger.success("Feature filtering working ✅")

    # ── Test 9: All 5 stocks ──────────────────────────────────────────────────
    logger.info("\n📈 Test 9: Preprocess all 5 stocks")
    logger.info("This will take 2-5 minutes...")

    all_data = preprocess_all_stocks("2020-01-01", "2024-01-01")

    logger.info(f"\n{'─'*60}")
    logger.info(f"{'Symbol':<15} {'Train':>10} {'Val':>8} {'Test':>8} {'Features':>10} {'MB':>6}")
    logger.info(f"{'─'*60}")

    for symbol, d in all_data.items():
        logger.info(
            f"{symbol:<15} "
            f"{str(d.X_train.shape):>10} "
            f"{len(d.y_val):>8} "
            f"{len(d.y_test):>8} "
            f"{d.n_features:>10} "
            f"{d.memory_mb:>6.1f}"
        )

    logger.info(f"{'─'*60}")

    # ── Test 10: Save all scalers ─────────────────────────────────────────────
    logger.info("\n💾 Test 10: Save all scalers")

    for symbol, d in all_data.items():
        save_scaler(d.scaler, symbol, d)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.success(f"All tests passed!")
    logger.success(f"Preprocessed {len(all_data)}/5 stocks")
    logger.success(f"All scalers saved to models/saved/")
    logger.success(f"Day 4 complete — ready for Day 5 (tests + Week 1 review)")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_day4()