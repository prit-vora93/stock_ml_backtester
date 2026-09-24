"""
backtesting/engine.py
------------------------
The day-by-day backtest loop tying together backtesting/portfolio.py,
risk_manager.py, strategy.py, and metrics.py.

Two entry points:

    run_backtest()
        Single symbol, single Portfolio. Maps directly onto one
        api.database.BacktestResult row + its Trade rows.

    run_portfolio_backtest()
        Multiple symbols sharing ONE capital pool / ONE Portfolio on a
        common calendar — this is what settings.MAX_POSITIONS ("max
        concurrent positions across the portfolio") actually governs.
        Returns an aggregate portfolio-level BacktestSummary (the only
        place a coherent equity curve/Sharpe/drawdown exists, since
        cash is pooled and reused across symbols) plus a per-symbol
        breakdown of TRADE-only stats (win rate, profit factor, trade
        count — nothing equity-curve-based, since that isn't
        well-defined per symbol when capital is shared).

Both are long-only, daily-bar backtests: each row of `prices` is one
trading day, decisions are made and filled at that day's close, and
stop-losses are checked against that day's low (the worst intraday
price a real stop order could have been hit at). The daily-loss circuit
breaker (MAX_DAILY_LOSS_PCT) is therefore a close-to-close comparison,
not a true intraday one — a known limitation of daily-bar backtesting,
not something finer-grained data would need code changes to fix (just
finer-grained `prices`).

This module is DB-agnostic by design (like models/ensemble.py): it
returns plain dataclasses/dicts shaped to match api.database's
BacktestResult/Trade columns, and leaves persistence to whatever caller
owns the DB session (future api/services.py).

Usage:
    from backtesting.engine import run_backtest
    from backtesting.risk_manager import RiskConfig
    from backtesting.strategy import SignalStrategy

    # prices: DataFrame indexed by date, columns open/high/low/close
    # signals: array-like of "BUY"/"HOLD"/"SELL", same length/order as prices
    #   (e.g. from models.ensemble.predict_ensemble(...)["signal"])
    summary = run_backtest("RELIANCE.NS", prices, signals)
    print(summary.metrics["sharpe_ratio"], len(summary.trades))
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtesting.metrics import compute_all_metrics, trade_stats
from backtesting.portfolio import Portfolio
from backtesting.risk_manager import (
    RiskConfig,
    calculate_position_size,
    calculate_stop_loss_price,
    check_stop_loss_hit,
    can_open_new_position,
    check_daily_loss_limit,
)
from backtesting.strategy import SignalStrategy, Action
from config.settings import INITIAL_CAPITAL
from utils.logger import logger


@dataclass
class BacktestSummary:
    """
    Result of a backtest run. `metrics` keys match
    api.database.BacktestResult's columns; `trades` items match
    api.database.Trade's columns — both ready for
    Model(**fields, backtest_id=..., symbol=...) once a caller owns a
    DB session.
    """
    symbol:        str
    start_date:    object
    end_date:      object
    metrics:       dict
    trades:        list[dict]
    equity_curve:  list[tuple] = field(default_factory=list)


def _validate_prices(prices: pd.DataFrame, signals) -> None:
    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")
    if len(prices) != len(signals):
        raise ValueError(
            f"prices ({len(prices)} rows) and signals ({len(signals)} items) "
            f"must be the same length and aligned by position"
        )
    if len(prices) == 0:
        raise ValueError("prices is empty — nothing to backtest")


def run_backtest(
    symbol:           str,
    prices:           pd.DataFrame,
    signals,
    initial_capital:  float = INITIAL_CAPITAL,
    strategy:         Optional[SignalStrategy] = None,
    risk:             Optional[RiskConfig]     = None,
) -> BacktestSummary:
    """
    Runs a single-symbol, long-only, daily-bar backtest.

    Args:
        symbol:          e.g. "RELIANCE.NS" (for logging/labeling only)
        prices:          DataFrame indexed by date, columns open/high/low/close,
                         chronological order — e.g. from data.storage.get_stock_data()
        signals:         Array-like of "BUY"/"HOLD"/"SELL", same length as
                         `prices`, aligned by position (row i's signal is
                         for prices.iloc[i]) — e.g. from
                         models.ensemble.predict_ensemble(...)["signal"]
        initial_capital: Starting cash (settings.INITIAL_CAPITAL)
        strategy:        SignalStrategy (defaults to the standard one)
        risk:            RiskConfig (defaults to settings.py values)

    Returns:
        BacktestSummary
    """
    _validate_prices(prices, signals)

    strategy = strategy or SignalStrategy()
    risk     = risk or RiskConfig()
    signals  = list(signals)

    portfolio = Portfolio(initial_capital=initial_capital)
    n = len(prices)

    logger.info(f"Backtest start: {symbol} | {n} bars | capital={initial_capital:,.0f}")

    for i in range(n):
        date = prices.index[i]
        bar  = prices.iloc[i]
        is_last_bar = (i == n - 1)

        day_start_equity = portfolio.equity_curve[-1][1] if portfolio.equity_curve else initial_capital
        has_position = portfolio.has_position(symbol)

        # ── Stop-loss check (worst intraday price) ────────────────────────
        if has_position:
            pos = portfolio.positions[symbol]
            if check_stop_loss_hit(pos.stop_loss_price, bar["low"]):
                portfolio.close_position(symbol, date, pos.stop_loss_price, reason="stop_loss")
                has_position = False

        # ── Forced close on the final bar — no time left to hold ──────────
        if has_position and is_last_bar:
            portfolio.close_position(symbol, date, bar["close"], reason="end_of_period")
            has_position = False

        # ── Strategy-driven action ─────────────────────────────────────────
        elif has_position:
            action = strategy.decide(signals[i], has_position=True)
            if action == Action.CLOSE:
                portfolio.close_position(symbol, date, bar["close"], reason="signal")
                has_position = False

        elif not is_last_bar:
            action = strategy.decide(signals[i], has_position=False)
            if action == Action.OPEN:
                current_equity = portfolio.total_equity({symbol: bar["close"]})
                if (
                    can_open_new_position(portfolio.position_count, risk)
                    and check_daily_loss_limit(day_start_equity, current_equity, risk)
                ):
                    stop_price = calculate_stop_loss_price(bar["close"], risk)
                    quantity   = calculate_position_size(portfolio.cash, bar["close"], risk)
                    if quantity > 0:
                        portfolio.open_position(
                            symbol, date, bar["close"], quantity, stop_loss_price=stop_price
                        )

        portfolio.mark_to_market(date, {symbol: bar["close"]})

    equity_curve = np.array([e for _, e in portfolio.equity_curve])
    metrics = compute_all_metrics(equity_curve, portfolio.trades, initial_capital)

    logger.success(
        f"Backtest complete: {symbol} | "
        f"return={metrics['total_return']:+.1%} | "
        f"sharpe={metrics['sharpe_ratio']:.2f} | "
        f"trades={metrics['total_trades']} | "
        f"win_rate={metrics['win_rate']:.1%}"
    )

    return BacktestSummary(
        symbol=symbol,
        start_date=prices.index[0],
        end_date=prices.index[-1],
        metrics=metrics,
        trades=portfolio.trades,
        equity_curve=portfolio.equity_curve,
    )


def run_portfolio_backtest(
    prices:           dict[str, pd.DataFrame],
    signals:          dict[str, object],
    initial_capital:  float = INITIAL_CAPITAL,
    strategy:         Optional[SignalStrategy] = None,
    risk:             Optional[RiskConfig]     = None,
) -> dict:
    """
    Runs a multi-symbol, long-only, daily-bar backtest sharing ONE
    capital pool across all symbols — this is what MAX_POSITIONS
    ("max concurrent positions across the portfolio") governs; a
    single-symbol run_backtest() never has more than one position, so
    that cap is meaningless there.

    All symbols must share an identical trading calendar (same dates,
    same order) — align/reindex before calling if your source data
    has gaps (e.g. one stock missing a trading holiday another had data
    for).

    Args:
        prices:          {symbol: DataFrame} — each indexed by date,
                         columns open/high/low/close. All DataFrames
                         must have the exact same index.
        signals:         {symbol: array-like} — each same length as
                         prices[symbol], aligned by position.
        initial_capital: Starting cash shared across all symbols.
        strategy:        SignalStrategy (defaults to the standard one)
        risk:            RiskConfig (defaults to settings.py values —
                         risk.max_positions caps concurrent positions
                         across ALL symbols combined)

    Returns:
        {
            "per_symbol": {symbol: trade_stats(...) dict, ...}
                          — win_rate/profit_factor/total_trades/
                          winning_trades/losing_trades/total_pnl and
                          that symbol's own trades list; NOT equity-curve
                          metrics (see note above on why),
            "portfolio":  BacktestSummary with symbol="PORTFOLIO" —
                          the full equity-curve-based metrics, computed
                          over every symbol's combined trades.
        }
    """
    symbols = list(prices.keys())
    if not symbols:
        raise ValueError("prices is empty — nothing to backtest")

    reference_index = prices[symbols[0]].index
    for symbol in symbols:
        _validate_prices(prices[symbol], signals[symbol])
        if not prices[symbol].index.equals(reference_index):
            raise ValueError(
                f"{symbol}'s price index doesn't match {symbols[0]}'s — "
                f"all symbols must share an identical trading calendar"
            )

    strategy = strategy or SignalStrategy()
    risk     = risk or RiskConfig()
    signals  = {s: list(v) for s, v in signals.items()}

    portfolio = Portfolio(initial_capital=initial_capital)
    n = len(reference_index)

    logger.info(
        f"Portfolio backtest start: {len(symbols)} symbols | {n} bars | "
        f"capital={initial_capital:,.0f} | max_positions={risk.max_positions}"
    )

    for i in range(n):
        date = reference_index[i]
        is_last_bar = (i == n - 1)
        bars = {symbol: prices[symbol].iloc[i] for symbol in symbols}

        day_start_equity = portfolio.equity_curve[-1][1] if portfolio.equity_curve else initial_capital

        # ── Pass 1: stop-losses, signal closes, and forced final closes ────
        # Processed for every symbol before any new opens, so capital and
        # position slots freed up today are available to this bar's opens.
        for symbol in symbols:
            if not portfolio.has_position(symbol):
                continue
            bar = bars[symbol]
            pos = portfolio.positions[symbol]

            if check_stop_loss_hit(pos.stop_loss_price, bar["low"]):
                portfolio.close_position(symbol, date, pos.stop_loss_price, reason="stop_loss")
                continue

            if is_last_bar:
                portfolio.close_position(symbol, date, bar["close"], reason="end_of_period")
                continue

            action = strategy.decide(signals[symbol][i], has_position=True)
            if action == Action.CLOSE:
                portfolio.close_position(symbol, date, bar["close"], reason="signal")

        # ── Pass 2: new opens, gated by shared max_positions / daily loss ──
        if not is_last_bar:
            for symbol in symbols:
                if portfolio.has_position(symbol):
                    continue
                action = strategy.decide(signals[symbol][i], has_position=False)
                if action != Action.OPEN:
                    continue

                close_prices = {s: bars[s]["close"] for s in symbols}
                current_equity = portfolio.total_equity(close_prices)

                if not can_open_new_position(portfolio.position_count, risk):
                    break   # no slots left at all — later symbols this bar can't open either
                if not check_daily_loss_limit(day_start_equity, current_equity, risk):
                    break   # halted for the day — same reasoning

                bar = bars[symbol]
                stop_price = calculate_stop_loss_price(bar["close"], risk)
                quantity   = calculate_position_size(portfolio.cash, bar["close"], risk)
                if quantity > 0:
                    portfolio.open_position(
                        symbol, date, bar["close"], quantity, stop_loss_price=stop_price
                    )

        portfolio.mark_to_market(date, {s: bars[s]["close"] for s in symbols})

    equity_curve = np.array([e for _, e in portfolio.equity_curve])
    portfolio_metrics = compute_all_metrics(equity_curve, portfolio.trades, initial_capital)

    portfolio_summary = BacktestSummary(
        symbol="PORTFOLIO",
        start_date=reference_index[0],
        end_date=reference_index[-1],
        metrics=portfolio_metrics,
        trades=portfolio.trades,
        equity_curve=portfolio.equity_curve,
    )

    per_symbol = {}
    for symbol in symbols:
        symbol_trades = [t for t in portfolio.trades if t["symbol"] == symbol]
        per_symbol[symbol] = {**trade_stats(symbol_trades), "trades": symbol_trades}

    logger.success(
        f"Portfolio backtest complete: return={portfolio_metrics['total_return']:+.1%} | "
        f"sharpe={portfolio_metrics['sharpe_ratio']:.2f} | "
        f"trades={portfolio_metrics['total_trades']}"
    )

    return {"per_symbol": per_symbol, "portfolio": portfolio_summary}
