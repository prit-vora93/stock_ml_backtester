"""
config/settings.py
------------------
Central configuration file for the entire project.

Every setting, constant, and parameter lives here.
All other files import from here — never hardcode values elsewhere.

Usage (in any other file):
    from config.settings import STOCKS, DATABASE_URL, INITIAL_CAPITAL
"""

import os
from dotenv import load_dotenv

# Load .env file first (must be before any os.getenv calls)
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATABASE
# ─────────────────────────────────────────────────────────────────────────────
# Read individual values from .env
DB_HOST     = os.getenv("DB_HOST", "localhost")   # "localhost" is default fallback
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "stock_ml")
DB_USER     = os.getenv("DB_USER", "stock_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "stockpass123")

# Build the full connection URL that SQLAlchemy uses
# Format: postgresql://user:password@host:port/database
# Example: postgresql://stock_user:stockpass123@localhost:5432/stock_ml
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
APP_ENV    = os.getenv("APP_ENV", "development")

# os.getenv always returns a STRING, so we compare to "True" to get a boolean
DEBUG      = os.getenv("DEBUG", "True") == "True"

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: API SERVER
# ─────────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))   # Convert string → integer


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: STOCKS
# ─────────────────────────────────────────────────────────────────────────────
# Indian stocks use ".NS" suffix in yfinance (NS = National Stock Exchange)
# These 5 are large-cap, liquid, reliable data — perfect for ML training
#
# Why these 5?
# RELIANCE   → Largest Indian company, high volume, good data
# INFY       → IT sector leader, international exposure
# TCS        → IT sector, very stable, consistent data
# WIPRO      → IT sector, good for comparison with INFY/TCS
# HDFCBANK   → Banking sector leader, different behavior from IT stocks
#
# Having stocks from 2 sectors (IT + Banking) helps model learn
# different market behaviors.

STOCKS = [
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "WIPRO.NS",
    "HDFCBANK.NS",
]

# Nifty 50 index — used as market benchmark in feature engineering
# (How is a stock moving compared to the overall market?)
MARKET_INDEX = "^NSEI"

# How many years of historical data to fetch per stock
# 5 years = ~1250 trading days = enough for ML training
DATA_YEARS = 5


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: ML MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# LSTM looks at last 60 days to predict next day
# Why 60? Covers ~3 months of trading — enough context without too much noise
SEQUENCE_LENGTH = 60

# Label thresholds — what counts as UP or DOWN movement?
# If next day close is 1% ABOVE today → label = UP
# If next day close is 1% BELOW today → label = DOWN
# Within ±1% → label = HOLD (market noise, not meaningful signal)
# Increasing this (e.g. 2%) = fewer but more reliable signals
LABEL_THRESHOLD = 0.01   # 1%

# Label classes (what model predicts)
# 0 = DOWN, 1 = HOLD, 2 = UP
LABEL_DOWN = 0
LABEL_HOLD = 1
LABEL_UP   = 2

# Train / Validation / Test split ratios (must sum to 1.0)
# CRITICAL: These are TIME-BASED splits, NOT random
# First 70% of data = training
# Next 15% of data  = validation (tune model during training)
# Last 15% of data  = test (final evaluation, never touch during training)
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# Random seed — ensures your results are reproducible
# With same seed, training always produces same model
RANDOM_SEED = 42

# LSTM training settings
LSTM_EPOCHS     = 50     # How many times model sees full training data
LSTM_BATCH_SIZE = 32     # How many samples per gradient update
LSTM_PATIENCE   = 10     # Stop early if no improvement for 10 epochs

# XGBoost settings
XGB_N_ESTIMATORS = 200   # Number of trees
XGB_MAX_DEPTH    = 6     # Max depth per tree
XGB_LEARNING_RATE = 0.05 # How fast model learns (lower = more stable)

# Ensemble weights (must sum to 1.0)
# Give LSTM slightly more weight since it captures temporal patterns better
LSTM_WEIGHT   = 0.6
XGB_WEIGHT    = 0.4

# Minimum confidence to act on a prediction
# If ensemble is less than 60% confident → HOLD (don't trade)
MIN_CONFIDENCE = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────

# Starting capital for all backtests
INITIAL_CAPITAL = 1_000_000   # ₹10,00,000 (10 lakhs)

# Risk per trade = 1% of current capital
# If capital = ₹10,00,000 → max risk per trade = ₹10,000
RISK_PER_TRADE = 0.01

# Maximum number of stocks open at the same time
# Prevents putting all money in one trade
MAX_POSITIONS = 5

# Stop loss — auto-exit if trade loses more than 2%
# Protects against large unexpected losses
STOP_LOSS_PCT = 0.02

# If total portfolio loses 5% in one day → stop trading for the day
# Prevents emotional panic trading
MAX_DAILY_LOSS_PCT = 0.05

# Transaction costs (realistic for Indian market)
# Zerodha charges ~0.03% per trade
COMMISSION_PCT = 0.0003   # 0.03%

# Slippage = difference between expected and actual execution price
# You think you buy at ₹2000 but actually get ₹2004 due to market impact
SLIPPAGE_PCT = 0.0002     # 0.02%


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────

# BASE_DIR = root of project (stock_ml_backtester/)
# __file__ = this file (config/settings.py)
# dirname(__file__) = config/
# dirname(dirname(__file__)) = stock_ml_backtester/  ← that's BASE_DIR
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where trained ML models get saved
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")

# Where log files get saved
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

# Create these directories if they don't exist yet
# exist_ok=True means "don't crash if folder already exists"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
# Run this file directly to verify everything loaded correctly:
#   python config/settings.py

if __name__ == "__main__":
    print("\n── Database ──────────────────────────────")
    print(f"  URL      : postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"  Name     : {DB_NAME}")

    print("\n── App ───────────────────────────────────")
    print(f"  Env      : {APP_ENV}")
    print(f"  Debug    : {DEBUG}")

    print("\n── Stocks ────────────────────────────────")
    for s in STOCKS:
        print(f"  {s}")

    print("\n── ML Settings ───────────────────────────")
    print(f"  Seq len  : {SEQUENCE_LENGTH} days")
    print(f"  Splits   : {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT}")
    print(f"  Seed     : {RANDOM_SEED}")
    print(f"  Ensemble : LSTM {LSTM_WEIGHT} / XGB {XGB_WEIGHT}")

    print("\n── Backtesting ───────────────────────────")
    print(f"  Capital  : ₹{INITIAL_CAPITAL:,.0f}")
    print(f"  Risk/trade: {RISK_PER_TRADE*100}%")
    print(f"  Stop loss : {STOP_LOSS_PCT*100}%")
    print(f"  Commission: {COMMISSION_PCT*100}%")

    print("\n── Paths ─────────────────────────────────")
    print(f"  Base dir : {BASE_DIR}")
    print(f"  Models   : {MODELS_DIR}")
    print(f"  Logs     : {LOGS_DIR}")

    print("\n✅ settings.py loaded correctly!\n")