"""
utils/logger.py
---------------
Centralized logging setup using loguru.

Gives you two outputs simultaneously:
  1. Colored terminal output  (see logs live while coding)
  2. File output              (permanent record saved to disk)

Usage in ANY other file:
    from utils.logger import logger

    logger.debug("Detailed info for debugging")
    logger.info("General information")
    logger.warning("Something unexpected but not breaking")
    logger.error("Something broke — needs attention")
    logger.success("Task completed successfully")
    logger.critical("App is about to crash")
"""

import sys
import os
from loguru import logger
from config.settings import LOGS_DIR, APP_ENV, DEBUG


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Remove default loguru handler
# ─────────────────────────────────────────────────────────────────────────────
# loguru adds a basic handler automatically when imported.
# We remove it here so we can add our own custom handlers below.
# Without this, you'd get duplicate log messages.

logger.remove()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Console Handler (colored output in terminal)
# ─────────────────────────────────────────────────────────────────────────────
# This is what you see in your terminal while the app is running.
# Colors help you spot errors and warnings instantly.
#
# Color scheme:
#   DEBUG    → gray     (detailed dev info, not important)
#   INFO     → white    (general info)
#   WARNING  → yellow   (something unexpected)
#   ERROR    → red      (something broke)
#   SUCCESS  → green    (task completed)
#   CRITICAL → bold red (app about to crash)

logger.add(
    sys.stdout,
    colorize=True,

    # Show DEBUG messages only in development mode
    # In production, only show INFO and above (less noise)
    level="DEBUG" if DEBUG else "INFO",

    # Format: TIME | LEVEL | FILE:LINE - MESSAGE
    # Example: 14:32:05 | INFO     | data/fetcher.py:42 - Fetching RELIANCE.NS...
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: File Handler (saves logs to disk)
# ─────────────────────────────────────────────────────────────────────────────
# Saves every log message to a file in your logs/ folder.
# Useful for:
#   - Reviewing what happened during a long training run
#   - Debugging issues that occurred while you weren't watching
#   - Keeping a history of backtest runs
#
# File naming: app_2024-01-15.log (one file per day)
# Rotation: New file created every day at midnight
# Retention: Files older than 7 days are automatically deleted

logger.add(
    os.path.join(LOGS_DIR, "app_{time:YYYY-MM-DD}.log"),

    # Rotate = start a new file every day
    rotation="1 day",

    # Retention = delete files older than 7 days
    # Prevents logs/ folder from filling up your disk
    retention="7 days",

    # Always log INFO and above to file (never DEBUG — too verbose for files)
    level="INFO",

    # File format includes full date (not just time like terminal)
    # Example: 2024-01-15 14:32:05 | INFO     | data/fetcher.py:42 - Fetching...
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",

    # Encode as UTF-8 so special characters (₹, symbols) don't break the file
    encoding="utf-8"
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Separate Error Log File
# ─────────────────────────────────────────────────────────────────────────────
# Errors and critical messages get saved to a SEPARATE file.
# This means you can quickly check errors_YYYY-MM-DD.log
# without scrolling through thousands of INFO messages.

logger.add(
    os.path.join(LOGS_DIR, "errors_{time:YYYY-MM-DD}.log"),
    rotation="1 day",
    retention="30 days",       # Keep error logs longer (30 days)
    level="ERROR",             # Only ERROR and CRITICAL go here
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    encoding="utf-8"
)