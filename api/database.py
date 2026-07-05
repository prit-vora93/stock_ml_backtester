"""
api/database.py
---------------
Database connection, table definitions, and session management.

Three jobs:
  1. Create connection to PostgreSQL (Engine)
  2. Define all table structures (SQLAlchemy Models)
  3. Provide sessions for reading/writing data

Other files use it like:
    from api.database import SessionLocal, StockData, BacktestResult
    db = SessionLocal()
    rows = db.query(StockData).filter(StockData.symbol == "RELIANCE.NS").all()
    db.close()
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Date,
    DateTime,
    BigInteger,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from config.settings import DATABASE_URL
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: ENGINE
# Creates the actual connection to PostgreSQL.
# Think of it as the pipeline between your app and the database.
# Created ONCE at startup and reused for the entire app lifetime.
# ─────────────────────────────────────────────────────────────────────────────

engine = create_engine(
    DATABASE_URL,

    # echo=True prints every SQL query to terminal (useful for debugging)
    # Set False once everything is working — it's very noisy
    echo=False,

    # pool_size = how many connections to keep open permanently
    # (like having 5 delivery trucks always ready)
    pool_size=5,

    # max_overflow = extra connections allowed during heavy load
    # (like calling 10 extra trucks when warehouse is busy)
    max_overflow=10,

    # If a connection sits idle for 30 minutes, close it
    # Prevents stale/broken connections
    pool_recycle=1800,

    # If getting a connection takes longer than 30 seconds, give up
    pool_timeout=30,
)


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: SESSION FACTORY
# SessionLocal is a factory — call it to get a new session object.
# Each session is like one conversation with the database.
# Always close the session when done to free up the connection.
# ─────────────────────────────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    # Which database connection to use
    bind=engine,

    # autocommit=False means changes aren't saved until you call db.commit()
    # This gives you control — if something fails midway, nothing gets saved
    autocommit=False,

    # autoflush=False means SQLAlchemy won't automatically sync pending changes
    # before every query. More control, better performance.
    autoflush=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: BASE CLASS
# All table models inherit from Base.
# Base keeps track of every table you define below.
# When create_tables() runs, it uses Base to know what to create.
# ─────────────────────────────────────────────────────────────────────────────

Base = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: StockData
# Stores raw daily OHLCV price data for each stock.
# This is the foundation — every other table depends on this data.
#
# One row = one trading day for one stock
# Example row:
#   symbol = "RELIANCE.NS"
#   date   = 2024-01-15
#   open   = 2450.00
#   high   = 2480.00
#   low    = 2430.00
#   close  = 2465.00
#   volume = 5000000
# ─────────────────────────────────────────────────────────────────────────────

class StockData(Base):
    __tablename__ = "stock_data"

    # Primary key — unique ID for each row, auto-increments (1, 2, 3...)
    id     = Column(Integer, primary_key=True, index=True)

    # index=True on symbol and date makes queries MUCH faster
    # Without index: PostgreSQL scans every row to find RELIANCE.NS
    # With index: PostgreSQL jumps directly to RELIANCE.NS rows
    symbol = Column(String(20), nullable=False, index=True)
    date   = Column(Date,       nullable=False, index=True)

    # OHLCV price data
    open   = Column(Float, nullable=False)
    high   = Column(Float, nullable=False)
    low    = Column(Float, nullable=False)
    close  = Column(Float, nullable=False)

    # BigInteger because volume can be in hundreds of millions
    # Regular Integer maxes out at ~2 billion, BigInteger handles more
    volume = Column(BigInteger, nullable=False)

    # Automatically set to current timestamp when row is created
    # You never need to set this manually
    created_at = Column(DateTime, server_default=func.now())

    # Prevent duplicate rows for same stock + same date
    # If you try to insert RELIANCE.NS for 2024-01-15 twice, DB rejects it
    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_stock_symbol_date"),
    )

    # Relationship: one StockData row links to one Indicator row
    # uselist=False means it's one-to-one (not one-to-many)
    # This lets you do: stock_row.indicators.rsi_14
    indicators = relationship(
        "Indicator",
        back_populates="stock_data",
        uselist=False,
        cascade="all, delete-orphan"   # Delete indicator if stock row deleted
    )

    def __repr__(self):
        return f"<StockData {self.symbol} {self.date} close={self.close}>"


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: Indicator
# Stores pre-calculated technical indicators for each stock day.
# Linked one-to-one with StockData via foreign key.
#
# Why store in DB instead of calculating on the fly?
#   - Calculation takes time (especially for 1250 days × 5 stocks)
#   - Storing means you calculate ONCE, read MANY times
#   - Consistent values across all model training runs
#
# One row = all indicators for one stock on one date
# ─────────────────────────────────────────────────────────────────────────────

class Indicator(Base):
    __tablename__ = "indicators"

    id           = Column(Integer, primary_key=True, index=True)

    # Foreign key links this row to its parent StockData row
    # unique=True enforces one-to-one (one indicator set per stock day)
    stock_data_id = Column(
        Integer,
        ForeignKey("stock_data.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )

    # Store symbol + date here too for easier direct queries
    # (avoids always needing a JOIN with stock_data table)
    symbol = Column(String(20), nullable=False, index=True)
    date   = Column(Date,       nullable=False, index=True)

    # ── Trend Indicators ──────────────────────────────────────────────────────
    # Moving averages smooth out price noise to reveal the trend direction
    # SMA = Simple Moving Average (equal weight to all days)
    # EMA = Exponential Moving Average (more weight to recent days)
    sma_20  = Column(Float)    # Average of last 20 days close price
    sma_50  = Column(Float)    # Average of last 50 days close price
    sma_200 = Column(Float)    # Average of last 200 days (long-term trend)
    ema_12  = Column(Float)    # Fast EMA (reacts quickly to price changes)
    ema_26  = Column(Float)    # Slow EMA (reacts slowly to price changes)

    # ── Momentum Indicators ───────────────────────────────────────────────────
    # Measure speed and strength of price movement

    # RSI: 0-100 scale
    # Above 70 = overbought (might drop soon)
    # Below 30 = oversold (might rise soon)
    rsi_14      = Column(Float)

    # MACD = EMA_12 - EMA_26 (positive = bullish momentum)
    # Signal = 9-day EMA of MACD
    # Histogram = MACD - Signal (shows momentum acceleration)
    macd        = Column(Float)
    macd_signal = Column(Float)
    macd_hist   = Column(Float)

    # Stochastic: 0-100, similar to RSI
    # %K = current position within recent high-low range
    # %D = 3-day moving average of %K
    stoch_k = Column(Float)
    stoch_d = Column(Float)

    # ── Volatility Indicators ─────────────────────────────────────────────────
    # Measure how much price is moving (risk indicator)

    # Bollinger Bands: price channel around moving average
    # Upper/Lower bands expand when volatile, contract when calm
    bb_upper = Column(Float)   # Upper band (resistance level)
    bb_lower = Column(Float)   # Lower band (support level)
    bb_mid   = Column(Float)   # Middle band (= SMA 20)
    bb_width = Column(Float)   # Band width (measures current volatility)

    # ATR: average range of price movement per day
    # High ATR = volatile stock, Low ATR = calm stock
    atr_14 = Column(Float)

    # ── Volume Indicators ─────────────────────────────────────────────────────
    # Volume confirms price moves — a price rise WITH high volume is stronger

    # OBV: running total of volume (up days add, down days subtract)
    # Rising OBV with rising price = strong uptrend confirmed by volume
    obv = Column(Float)

    # ── Price Pattern Features ────────────────────────────────────────────────
    # Simple features derived from raw price

    # How much did price change over different periods?
    price_change_1d  = Column(Float)   # vs yesterday (%)
    price_change_5d  = Column(Float)   # vs 1 week ago (%)
    price_change_20d = Column(Float)   # vs 1 month ago (%)

    # How wide was today's candle? (high-low range / close)
    # Large ratio = volatile day, small ratio = quiet day
    high_low_ratio = Column(Float)

    # Where is price relative to 52-week high?
    # 1.0 = AT the 52-week high, 0.5 = halfway below it
    dist_52w_high = Column(Float)

    # Correlation with Nifty 50 over last 20 days
    # High correlation = stock moves with market
    # Low correlation = stock has independent behavior
    nifty_corr_20d = Column(Float)

    created_at = Column(DateTime, server_default=func.now())

    # Relationship back to parent StockData row
    stock_data = relationship("StockData", back_populates="indicators")

    # Composite index on symbol + date for fast lookups
    __table_args__ = (
        Index("ix_indicators_symbol_date", "symbol", "date"),
    )

    def __repr__(self):
        return f"<Indicator {self.symbol} {self.date} rsi={self.rsi_14:.1f}>"


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: Prediction
# Stores ML model predictions for each stock day.
# One row = predictions from all 3 models for one stock on one date.
#
# Stores raw probabilities (not just the final signal) so you can:
#   - Analyze model confidence over time
#   - Compare LSTM vs XGBoost vs Ensemble accuracy
#   - Tune the confidence threshold (MIN_CONFIDENCE in settings.py)
# ─────────────────────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id     = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date   = Column(Date,       nullable=False, index=True)

    # LSTM output probabilities (always sum to 1.0)
    # e.g. [0.20, 0.10, 0.70] means 70% chance of UP movement
    lstm_prob_down = Column(Float)
    lstm_prob_hold = Column(Float)
    lstm_prob_up   = Column(Float)

    # XGBoost output probabilities
    xgb_prob_down = Column(Float)
    xgb_prob_hold = Column(Float)
    xgb_prob_up   = Column(Float)

    # Ensemble final decision after combining LSTM + XGBoost
    ensemble_signal = Column(String(10))  # "BUY", "HOLD", or "SELL"

    # How confident is the ensemble? (0.0 to 1.0)
    # e.g. 0.72 = 72% confident in the signal
    confidence = Column(Float)

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_pred_symbol_date"),
    )

    def __repr__(self):
        return f"<Prediction {self.symbol} {self.date} {self.ensemble_signal} ({self.confidence:.0%})>"


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4: BacktestResult
# Stores summary performance metrics for each backtest run.
# One row = one complete backtest (e.g. RELIANCE ensemble strategy 2022-2024)
#
# Why store results?
#   - Compare different strategies side by side
#   - Track improvements as you tune models
#   - Display history on the React dashboard
# ─────────────────────────────────────────────────────────────────────────────

class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id          = Column(Integer,     primary_key=True, index=True)

    # UUID string that uniquely identifies this backtest run
    # e.g. "bt_20240115_143205_RELIANCE_ensemble"
    backtest_id = Column(String(60),  unique=True, nullable=False, index=True)

    symbol      = Column(String(20),  nullable=False)
    strategy    = Column(String(20),  nullable=False)  # "lstm", "xgboost", "ensemble"
    start_date  = Column(Date,        nullable=False)
    end_date    = Column(Date,        nullable=False)

    initial_capital = Column(Float, nullable=False)
    final_capital   = Column(Float)

    # ── Return Metrics ────────────────────────────────────────────────────────
    # total_return: e.g. 0.87 means 87% total return over the period
    total_return  = Column(Float)
    # annual_return: return compounded to yearly equivalent
    annual_return = Column(Float)

    # ── Risk Metrics ──────────────────────────────────────────────────────────
    # sharpe_ratio: return per unit of risk
    #   > 1.0 = good, > 2.0 = excellent, < 0 = worse than risk-free rate
    sharpe_ratio  = Column(Float)

    # sortino_ratio: like Sharpe but only penalizes downside volatility
    #   More relevant for trading strategies
    sortino_ratio = Column(Float)

    # max_drawdown: largest peak-to-trough drop during the period
    #   e.g. -0.12 = portfolio fell 12% from its peak at some point
    max_drawdown  = Column(Float)

    # ── Trade Metrics ─────────────────────────────────────────────────────────
    # win_rate: e.g. 0.62 = 62% of trades were profitable
    win_rate      = Column(Float)

    # profit_factor: gross profit / gross loss
    #   > 1.0 = profitable overall, 2.0 = made 2x what you lost
    profit_factor = Column(Float)

    total_trades   = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades  = Column(Integer)

    created_at = Column(DateTime, server_default=func.now())

    # One backtest has MANY individual trades (one-to-many relationship)
    trades = relationship(
        "Trade",
        back_populates="backtest",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"<BacktestResult {self.symbol} {self.strategy} "
            f"return={self.total_return:.0%} sharpe={self.sharpe_ratio:.2f}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 5: Trade
# Stores every individual trade made during a backtest.
# One row = one complete trade (entry + exit).
#
# Why store individual trades?
#   - Analyze which trades worked and which didn't
#   - Display trade list on dashboard
#   - Calculate win/loss streaks
#   - Debug strategy behavior
# ─────────────────────────────────────────────────────────────────────────────

class Trade(Base):
    __tablename__ = "trades"

    id          = Column(Integer, primary_key=True, index=True)

    # Links this trade to its parent backtest
    backtest_id = Column(
        String(60),
        ForeignKey("backtest_results.backtest_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    symbol      = Column(String(20), nullable=False)

    # Entry (when you BUY)
    entry_date  = Column(Date,  nullable=False)
    entry_price = Column(Float, nullable=False)

    # Exit (when you SELL) — nullable because trade might still be open
    exit_date   = Column(Date,  nullable=True)
    exit_price  = Column(Float, nullable=True)

    quantity    = Column(Integer, nullable=False)

    # P&L in rupees: e.g. 2500.0 = made ₹2500, -1200.0 = lost ₹1200
    pnl         = Column(Float)

    # P&L as percentage: e.g. 0.025 = 2.5% gain
    pnl_pct     = Column(Float)

    # Why did this trade exit?
    # "signal"        = model said SELL
    # "stop_loss"     = price hit stop loss level
    # "end_of_period" = backtest period ended, forced close
    exit_reason = Column(String(20))

    created_at  = Column(DateTime, server_default=func.now())

    # Relationship back to parent BacktestResult
    backtest = relationship("BacktestResult", back_populates="trades")

    def __repr__(self):
        return (
            f"<Trade {self.symbol} "
            f"{self.entry_date}@{self.entry_price} → "
            f"{self.exit_date}@{self.exit_price} "
            f"pnl=₹{self.pnl:,.0f}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def create_tables():
    """
    Creates all 5 tables in PostgreSQL.
    Safe to run multiple times — won't overwrite existing data.
    Uses checkfirst=True internally so existing tables are skipped.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.success("All database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


def get_db():
    """
    Yields a database session and guarantees it's closed after use.
    Use this in every function that needs to read/write the database.

    Usage:
        db = next(get_db())
        try:
            rows = db.query(StockData).all()
            db.commit()
        finally:
            db.close()

    Or with FastAPI dependency injection (Week 5):
        @app.get("/stocks")
        def get_stocks(db: Session = Depends(get_db)):
            return db.query(StockData).all()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()   # Undo any uncommitted changes if error occurs
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()      # Always close, even if exception occurred


def test_connection():
    """
    Tests that Python can connect to PostgreSQL.
    Call this at startup to catch connection issues early.
    Returns True if connected, False if failed.
    """
    try:
        with engine.connect() as conn:
            logger.success("Database connection successful")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.info("Checklist:")
        logger.info("  1. Is PostgreSQL running?  →  sudo systemctl start postgresql")
        logger.info("  2. Is DB_PASSWORD correct in .env?")
        logger.info("  3. Does database 'stock_ml' exist?")
        return False