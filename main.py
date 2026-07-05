"""
main.py
-------
Entry point for the project.
Run this file to test your Day 1 setup is working correctly.

Usage:
    python main.py
"""

from utils.logger import logger
from api.database import test_connection, create_tables
from config.settings import (
    DATABASE_URL, STOCKS, INITIAL_CAPITAL,
    SEQUENCE_LENGTH, MODELS_DIR
)


def run_day1_checks():
    """
    Run all Day 1 setup checks.
    Everything should pass before moving to Day 2.
    """

    logger.info("=" * 55)
    logger.info("  Stock ML Backtester — Day 1 Setup Check")
    logger.info("=" * 55)

    passed = 0
    failed = 0

    # ── Check 1: Config loads ─────────────────────────────────────────────────
    logger.info("\n📋 Check 1: Configuration")
    try:
        logger.info(f"  Database URL : {DATABASE_URL[:30]}...")
        logger.info(f"  Stocks       : {STOCKS}")
        logger.info(f"  Capital      : ₹{INITIAL_CAPITAL:,.0f}")
        logger.info(f"  Seq length   : {SEQUENCE_LENGTH} days")
        logger.info(f"  Models dir   : {MODELS_DIR}")
        logger.success("  ✅ Config loaded successfully")
        passed += 1
    except Exception as e:
        logger.error(f"  ❌ Config failed: {e}")
        failed += 1

    # ── Check 2: Database connection ──────────────────────────────────────────
    logger.info("\n🗄️  Check 2: Database Connection")
    if test_connection():
        passed += 1
    else:
        failed += 1

    # ── Check 3: Create tables ────────────────────────────────────────────────
    logger.info("\n📊 Check 3: Create Database Tables")
    try:
        create_tables()
        passed += 1
    except Exception as e:
        logger.error(f"  ❌ Table creation failed: {e}")
        failed += 1

    # ── Check 4: Verify tables exist ──────────────────────────────────────────
    logger.info("\n🔍 Check 4: Verify Tables in Database")
    try:
        from sqlalchemy import inspect
        from api.database import engine

        inspector = inspect(engine)
        tables    = inspector.get_table_names()
        expected  = ["stock_data", "indicators", "predictions",
                     "backtest_results", "trades"]

        for table in expected:
            if table in tables:
                logger.success(f"  ✅ Table '{table}' exists")
            else:
                logger.error(f"  ❌ Table '{table}' MISSING")
                failed += 1
        passed += 1
    except Exception as e:
        logger.error(f"  ❌ Table verification failed: {e}")
        failed += 1

    # ── Check 5: Write a test row + read it back ──────────────────────────────
    logger.info("\n✍️  Check 5: Insert + Read Test Row")
    try:
        from api.database import SessionLocal, StockData
        from datetime import date

        db = SessionLocal()

        # Create a test stock row
        test_row = StockData(
            symbol = "TEST.NS",
            date   = date(2024, 1, 1),
            open   = 100.0,
            high   = 105.0,
            low    = 98.0,
            close  = 103.0,
            volume = 1000000
        )
        db.add(test_row)
        db.commit()
        db.refresh(test_row)
        logger.success(f"  ✅ Inserted row — ID: {test_row.id}")

        # Read it back
        fetched = db.query(StockData).filter(StockData.symbol == "TEST.NS").first()
        assert fetched is not None
        assert fetched.close == 103.0
        logger.success(f"  ✅ Read row back — Close price: ₹{fetched.close}")

        # Clean up test row
        db.delete(fetched)
        db.commit()
        logger.success("  ✅ Test row cleaned up")
        db.close()
        passed += 1

    except Exception as e:
        logger.error(f"  ❌ Insert/Read test failed: {e}")
        failed += 1

    # ── Check 6: Logger works ─────────────────────────────────────────────────
    logger.info("\n📝 Check 6: Logger")
    logger.debug("   This is a DEBUG message")
    logger.info("   This is an INFO message")
    logger.warning("   This is a WARNING message")
    logger.success("  ✅ Logger working")
    passed += 1

    # ── Final Summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info(f"  Results: {passed} passed, {failed} failed")

    if failed == 0:
        logger.success("  🎉 ALL CHECKS PASSED! You're ready for Day 2.")
        logger.info("  Tomorrow: data/fetcher.py — fetch real stock data!")
    else:
        logger.error(f"  ⚠️  {failed} check(s) failed. Fix errors above before Day 2.")

    logger.info("=" * 55)


if __name__ == "__main__":
    run_day1_checks()