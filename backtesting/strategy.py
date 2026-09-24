"""
backtesting/strategy.py
--------------------------
Turns a bar's ensemble signal (already produced by models/ensemble.py —
"BUY"/"HOLD"/"SELL", already confidence-gated against MIN_CONFIDENCE)
into an intended trading action, given whether a position is currently
open. Long-only, matching the rest of this project (no shorting
anywhere in config/settings.py).

Deliberately knows NOTHING about risk limits (position sizing, max
concurrent positions, the daily loss circuit breaker) — those live in
backtesting/risk_manager.py and are applied by backtesting/engine.py
after the strategy proposes an action. This keeps "what do I want to
do" (strategy) separate from "am I allowed to" (risk), so either can
change independently.

Usage:
    from backtesting.strategy import SignalStrategy, Action

    strategy = SignalStrategy()
    action = strategy.decide(signal="BUY", has_position=False)   # Action.OPEN
"""

from enum import Enum


class Action(str, Enum):
    OPEN  = "OPEN"    # enter a new long position
    CLOSE = "CLOSE"   # exit the current position
    HOLD  = "HOLD"    # do nothing this bar


class SignalStrategy:
    """
    Default strategy: act directly on the ensemble's signal.

        "BUY"  + no position open -> OPEN
        "SELL" + position open    -> CLOSE
        anything else             -> HOLD

    A "SELL" signal with no position is a no-op (nothing to sell, this
    project doesn't short). A "BUY" signal while already holding is
    also a no-op (no pyramiding into an existing position).

    Subclass and override decide() for a different strategy — e.g.
    requiring N consecutive BUY signals before entering — without
    touching backtesting/engine.py.
    """

    def decide(self, signal: str, has_position: bool) -> Action:
        if signal == "BUY" and not has_position:
            return Action.OPEN
        if signal == "SELL" and has_position:
            return Action.CLOSE
        return Action.HOLD
