"""
backtesting/risk_manager.py
------------------------------
Position sizing, stop-loss levels, and the two circuit breakers that
cap how much a single trade or a single bad day can cost: max
concurrent positions and max daily loss.

Everything here is a pure function over plain values (capital, prices,
a Portfolio to *read* from) — no state of its own, no DB. Per-day state
(like "equity at the start of today") is the caller's (engine.py's) job
to track, since only it knows where a "day" begins/ends.

Usage:
    from backtesting.risk_manager import RiskConfig, calculate_position_size, calculate_stop_loss_price

    risk = RiskConfig()   # defaults from config/settings.py
    stop_price = calculate_stop_loss_price(entry_price=2450.0, risk=risk)
    quantity   = calculate_position_size(capital=1_000_000, entry_price=2450.0, risk=risk)
"""

from dataclasses import dataclass

from config.settings import (
    RISK_PER_TRADE,
    STOP_LOSS_PCT,
    MAX_POSITIONS,
    MAX_DAILY_LOSS_PCT,
)
from utils.logger import logger


@dataclass
class RiskConfig:
    """Bundles the risk knobs from settings.py so engine.py can pass one object around."""
    risk_per_trade:     float = RISK_PER_TRADE
    stop_loss_pct:      float = STOP_LOSS_PCT
    max_positions:       int   = MAX_POSITIONS
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT


def calculate_stop_loss_price(entry_price: float, risk: RiskConfig = None) -> float:
    """
    Long-only stop-loss price: entry_price * (1 - stop_loss_pct).

    Args:
        entry_price: The fill price the position was (or will be) opened at
        risk:        RiskConfig (defaults to settings.py values)

    Returns:
        The price at or below which the position should be stopped out.
    """
    risk = risk or RiskConfig()
    return entry_price * (1 - risk.stop_loss_pct)


def calculate_position_size(capital: float, entry_price: float, risk: RiskConfig = None) -> int:
    """
    Risk-based position sizing: how many shares can be bought such that,
    if the stop-loss is hit, the loss is at most `risk_per_trade` of
    current capital — capped so the trade also never exceeds what
    capital can actually afford outright.

    Args:
        capital:     Current available capital (cash) to size against
        entry_price: Intended entry (fill) price
        risk:        RiskConfig (defaults to settings.py values)

    Returns:
        Whole number of shares (0 if capital or entry_price is non-positive,
        or the risk budget rounds down to zero shares).
    """
    risk = risk or RiskConfig()

    if capital <= 0 or entry_price <= 0:
        return 0

    risk_amount    = capital * risk.risk_per_trade
    per_share_risk = entry_price * risk.stop_loss_pct
    if per_share_risk <= 0:
        return 0

    risk_based_qty = int(risk_amount // per_share_risk)
    affordable_qty = int(capital // entry_price)

    return max(0, min(risk_based_qty, affordable_qty))


def check_stop_loss_hit(stop_loss_price: float, bar_low_price: float) -> bool:
    """
    True if the bar's low touched or breached the stop-loss level —
    checked against the LOW (the worst intraday price), not the close,
    since a stop-loss order would have triggered intraday.
    """
    if stop_loss_price is None:
        return False
    return bar_low_price <= stop_loss_price


def can_open_new_position(current_position_count: int, risk: RiskConfig = None) -> bool:
    """True if there's room for another concurrent position under max_positions."""
    risk = risk or RiskConfig()
    return current_position_count < risk.max_positions


def check_daily_loss_limit(day_start_equity: float, current_equity: float, risk: RiskConfig = None) -> bool:
    """
    Circuit breaker: True if trading (opening NEW positions) is still
    allowed today, False if today's drawdown from day_start_equity has
    already breached max_daily_loss_pct.

    This only gates new entries — existing positions can still be
    closed (including stop-loss exits) even after the limit trips, so
    a halted day doesn't trap you in open risk.

    Args:
        day_start_equity: Portfolio equity at the start of today
        current_equity:   Portfolio equity right now
        risk:             RiskConfig (defaults to settings.py values)

    Returns:
        True = new positions still allowed. False = halt new entries for today.
    """
    risk = risk or RiskConfig()

    if day_start_equity <= 0:
        return True

    day_return = (current_equity - day_start_equity) / day_start_equity
    breached   = day_return <= -risk.max_daily_loss_pct

    if breached:
        logger.warning(
            f"Daily loss limit breached: {day_return:.2%} "
            f"(limit -{risk.max_daily_loss_pct:.0%}) — halting new entries for today"
        )

    return not breached
