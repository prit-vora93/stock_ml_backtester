"""
backtesting/metrics.py
------------------------
Pure functions computing performance metrics from a backtest's equity
curve and trade list. No DB, no I/O — everything here is a plain
function over numpy arrays / lists of dicts, so it's trivial to unit
test and reuse (e.g. from backtesting/engine.py, or later from
api/services.py when building a BacktestResult row).

compute_all_metrics() returns a dict whose keys match
api.database.BacktestResult's columns exactly, so a caller can do
BacktestResult(**compute_all_metrics(...), backtest_id=..., symbol=..., ...).

Usage:
    from backtesting.metrics import compute_all_metrics

    metrics = compute_all_metrics(equity_curve, trades, initial_capital=1_000_000)
    # metrics["sharpe_ratio"], metrics["max_drawdown"], ...
"""

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def compute_returns(equity_curve: np.ndarray) -> np.ndarray:
    """
    Daily percentage returns from an equity curve.

    Args:
        equity_curve: 1D array of portfolio equity values, one per bar,
                       chronological order.

    Returns:
        1D array of length len(equity_curve)-1 (empty if fewer than 2 points).
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) < 2:
        return np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(equity_curve) / equity_curve[:-1]
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def total_return(equity_curve: np.ndarray) -> float:
    """(final_equity - initial_equity) / initial_equity. 0.0 if <2 points or zero start."""
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return 0.0
    return float((equity_curve[-1] - equity_curve[0]) / equity_curve[0])


def annual_return(equity_curve: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Compound annual growth rate (CAGR), i.e. total_return scaled to a
    yearly-equivalent rate given the number of bars actually observed.

    Returns 0.0 for degenerate inputs (fewer than 2 points, non-positive
    starting equity, or a total loss that would make the exponent undefined).
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    n = len(equity_curve)
    if n < 2 or equity_curve[0] <= 0:
        return 0.0

    ratio = equity_curve[-1] / equity_curve[0]
    if ratio <= 0:
        return -1.0   # total loss

    years = (n - 1) / periods_per_year
    if years <= 0:
        return 0.0

    return float(ratio ** (1 / years) - 1)


def sharpe_ratio(
    returns:          np.ndarray,
    risk_free_rate:   float = 0.0,
    periods_per_year: int   = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sharpe ratio: mean excess return / std of returns, scaled
    by sqrt(periods_per_year). 0.0 if fewer than 2 return observations
    or zero volatility (flat/no returns).
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0

    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0

    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns:          np.ndarray,
    risk_free_rate:   float = 0.0,
    periods_per_year: int   = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sortino ratio: like Sharpe, but only penalizes downside
    volatility (returns below the risk-free rate). 0.0 if fewer than 2
    return observations or no downside deviation (never lost money).
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0

    period_rf = risk_free_rate / periods_per_year
    excess    = returns - period_rf
    downside  = excess[excess < 0]

    if len(downside) == 0:
        return 0.0   # no downside volatility to divide by

    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0:
        return 0.0

    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Largest peak-to-trough decline, as a negative fraction
    (e.g. -0.12 = a 12% drop from the running peak at some point).
    0.0 if fewer than 2 points.
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) < 2:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = (equity_curve - running_max) / running_max
    drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)

    return float(drawdowns.min())


def win_rate(trades: list[dict]) -> float:
    """Fraction of closed trades with pnl > 0. 0.0 if no trades."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades)


def profit_factor(trades: list[dict]) -> float:
    """
    Gross profit / gross loss. Convention for edge cases:
        - No trades at all           -> 0.0
        - Wins but zero/no losses    -> float("inf") (undefined upside, no downside)
        - No wins, only losses       -> 0.0
    """
    if not trades:
        return 0.0

    gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_loss   = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def trade_stats(trades: list[dict]) -> dict:
    """
    Trade-derived stats only — no equity curve needed. This is what's
    actually attributable to a single symbol inside a shared-capital,
    multi-symbol portfolio backtest, where a per-symbol equity curve
    isn't well-defined (cash is pooled and reused across symbols).

    Returns:
        {"total_trades": int, "winning_trades": int, "losing_trades": int,
         "win_rate": float, "profit_factor": float, "total_pnl": float}
    """
    winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losing_trades  = sum(1 for t in trades if t.get("pnl", 0) < 0)
    total_pnl      = sum(t.get("pnl", 0) for t in trades)

    return {
        "total_trades":   len(trades),
        "winning_trades": winning_trades,
        "losing_trades":  losing_trades,
        "win_rate":       win_rate(trades),
        "profit_factor":  profit_factor(trades),
        "total_pnl":      float(total_pnl),
    }


def compute_all_metrics(
    equity_curve:     np.ndarray,
    trades:           list[dict],
    initial_capital:  float,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Computes the full metrics set, keyed to match
    api.database.BacktestResult's columns exactly.

    Args:
        equity_curve:    1D array of portfolio equity per bar (chronological)
        trades:          list of dicts with at least a "pnl" key
                         (backtesting.portfolio.Portfolio.trades / Trade-shaped)
        initial_capital: starting capital (for final_capital / sanity)
        periods_per_year: trading periods per year for annualization (default 252)

    Returns:
        {
            "initial_capital": float, "final_capital": float,
            "total_return": float, "annual_return": float,
            "sharpe_ratio": float, "sortino_ratio": float, "max_drawdown": float,
            "win_rate": float, "profit_factor": float,
            "total_trades": int, "winning_trades": int, "losing_trades": int,
        }
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    returns = compute_returns(equity_curve)

    final_capital = float(equity_curve[-1]) if len(equity_curve) > 0 else float(initial_capital)
    stats = trade_stats(trades)

    return {
        "initial_capital": float(initial_capital),
        "final_capital":   final_capital,
        "total_return":    total_return(equity_curve),
        "annual_return":   annual_return(equity_curve, periods_per_year),
        "sharpe_ratio":    sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio":   sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown":    max_drawdown(equity_curve),
        "win_rate":        stats["win_rate"],
        "profit_factor":   stats["profit_factor"],
        "total_trades":    stats["total_trades"],
        "winning_trades":  stats["winning_trades"],
        "losing_trades":   stats["losing_trades"],
    }
