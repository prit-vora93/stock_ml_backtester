"""
tests/test_backtesting.py
---------------------------
Tests for the backtesting/ package: metrics, portfolio, risk_manager,
strategy, engine (single-symbol + portfolio-level).

All tests use small, synthetic, in-memory price/signal series (no DB,
no network) with hand-computable expected values, so assertions check
actual correctness rather than just "it didn't crash".

Run:
    pytest tests/test_backtesting.py -v
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtesting.metrics import (
    compute_returns, total_return, annual_return, sharpe_ratio, sortino_ratio,
    max_drawdown, win_rate, profit_factor, trade_stats, compute_all_metrics,
)
from backtesting.portfolio import Portfolio
from backtesting.risk_manager import (
    RiskConfig, calculate_position_size, calculate_stop_loss_price,
    check_stop_loss_hit, can_open_new_position, check_daily_loss_limit,
)
from backtesting.strategy import SignalStrategy, Action
from backtesting.engine import run_backtest, run_portfolio_backtest, BacktestSummary


# ═════════════════════════════════════════════════════════════════════════════
# TestMetrics
# ═════════════════════════════════════════════════════════════════════════════

class TestMetrics:

    def test_compute_returns(self):
        equity = np.array([100.0, 110.0, 105.0])
        returns = compute_returns(equity)
        assert np.allclose(returns, [0.10, (105 - 110) / 110])

    def test_compute_returns_too_short(self):
        assert compute_returns(np.array([100.0])).size == 0
        assert compute_returns(np.array([])).size == 0

    def test_total_return(self):
        assert total_return(np.array([100.0, 121.0])) == pytest.approx(0.21)
        assert total_return(np.array([100.0])) == 0.0

    def test_annual_return_one_year(self):
        # 252 periods = exactly 1 year; 21% total -> 21% annualized
        equity = np.array([100.0] + [100.0] * 251 + [121.0])
        assert annual_return(equity, periods_per_year=252) == pytest.approx(0.21, abs=1e-6)

    def test_annual_return_degenerate_inputs(self):
        assert annual_return(np.array([100.0])) == 0.0
        assert annual_return(np.array([0.0, 100.0])) == 0.0

    def test_max_drawdown(self):
        equity = np.array([100.0, 110.0, 105.0, 121.0])
        # peak 110 -> trough 105
        assert max_drawdown(equity) == pytest.approx((105 - 110) / 110)

    def test_max_drawdown_no_drop(self):
        equity = np.array([100.0, 105.0, 110.0, 115.0])
        assert max_drawdown(equity) == pytest.approx(0.0)

    def test_sortino_no_downside_is_zero(self):
        returns = compute_returns(np.array([100, 101, 102, 103, 104, 105], dtype=float))
        assert sortino_ratio(returns) == 0.0

    def test_sharpe_zero_volatility_is_zero(self):
        returns = np.array([0.01, 0.01, 0.01])   # constant -> zero std
        assert sharpe_ratio(returns) == 0.0

    def test_win_rate(self):
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -100}]
        assert win_rate(trades) == pytest.approx(0.5)
        assert win_rate([]) == 0.0

    def test_profit_factor(self):
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 200}, {"pnl": -100}]
        assert profit_factor(trades) == pytest.approx(300 / 150)

    def test_profit_factor_all_wins_is_inf(self):
        assert profit_factor([{"pnl": 100}, {"pnl": 50}]) == float("inf")

    def test_profit_factor_no_trades_is_zero(self):
        assert profit_factor([]) == 0.0

    def test_trade_stats_keys(self):
        trades = [{"pnl": 100}, {"pnl": -50}]
        stats = trade_stats(trades)
        assert set(stats.keys()) == {
            "total_trades", "winning_trades", "losing_trades",
            "win_rate", "profit_factor", "total_pnl",
        }
        assert stats["total_trades"] == 2
        assert stats["total_pnl"] == pytest.approx(50)

    def test_compute_all_metrics_matches_backtest_result_columns(self):
        equity = np.array([100.0, 105.0, 103.0, 110.0])
        trades = [{"pnl": 5}, {"pnl": -2}]
        metrics = compute_all_metrics(equity, trades, initial_capital=100.0)

        expected_keys = {
            "initial_capital", "final_capital", "total_return", "annual_return",
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate",
            "profit_factor", "total_trades", "winning_trades", "losing_trades",
        }
        assert set(metrics.keys()) == expected_keys
        assert metrics["final_capital"] == 110.0
        assert metrics["total_trades"] == 2


# ═════════════════════════════════════════════════════════════════════════════
# TestPortfolio
# ═════════════════════════════════════════════════════════════════════════════

class TestPortfolio:

    def test_open_position_applies_commission_and_slippage(self):
        p = Portfolio(initial_capital=100_000, commission_pct=0.001, slippage_pct=0.002)
        ok = p.open_position("A.NS", date(2023, 1, 1), price=100, quantity=10)
        assert ok

        fill = 100 * 1.002
        cost = fill * 10
        commission = cost * 0.001
        assert p.cash == pytest.approx(100_000 - cost - commission)
        assert p.has_position("A.NS")

    def test_cannot_open_second_position_same_symbol(self):
        p = Portfolio(initial_capital=100_000)
        p.open_position("A.NS", date(2023, 1, 1), price=100, quantity=10)
        assert p.open_position("A.NS", date(2023, 1, 2), price=100, quantity=5) is False

    def test_insufficient_cash_rejects_open(self):
        p = Portfolio(initial_capital=100)
        ok = p.open_position("B.NS", date(2023, 1, 1), price=1000, quantity=10)
        assert ok is False
        assert p.cash == 100   # unchanged

    def test_zero_or_negative_quantity_rejected(self):
        p = Portfolio(initial_capital=100_000)
        assert p.open_position("A.NS", date(2023, 1, 1), price=100, quantity=0) is False
        assert p.open_position("A.NS", date(2023, 1, 1), price=100, quantity=-5) is False

    def test_close_position_pnl_accounts_for_both_legs(self):
        p = Portfolio(initial_capital=100_000, commission_pct=0.001, slippage_pct=0.002)
        p.open_position("A.NS", date(2023, 1, 1), price=100, quantity=10)
        trade = p.close_position("A.NS", date(2023, 1, 5), price=110, reason="signal")

        entry_fill = 100 * 1.002
        entry_cost = entry_fill * 10 + (entry_fill * 10 * 0.001)
        exit_fill  = 110 * (1 - 0.002)
        exit_proceeds = exit_fill * 10 - (exit_fill * 10 * 0.001)
        expected_pnl = exit_proceeds - entry_cost

        assert trade["pnl"] == pytest.approx(expected_pnl)
        assert trade["exit_reason"] == "signal"
        assert not p.has_position("A.NS")

    def test_close_nonexistent_position_returns_none(self):
        p = Portfolio(initial_capital=100_000)
        assert p.close_position("A.NS", date(2023, 1, 1), price=100) is None

    def test_mark_to_market_includes_open_positions(self):
        p = Portfolio(initial_capital=100_000)
        p.open_position("A.NS", date(2023, 1, 1), price=100, quantity=10)
        equity = p.mark_to_market(date(2023, 1, 2), {"A.NS": 110})
        assert equity == pytest.approx(p.cash + 110 * 10)
        assert p.equity_curve == [(date(2023, 1, 2), equity)]

    def test_force_close_all(self):
        p = Portfolio(initial_capital=100_000)
        p.open_position("X.NS", date(2023, 1, 1), price=50, quantity=10)
        p.open_position("Y.NS", date(2023, 1, 1), price=60, quantity=5)

        closed = p.force_close_all(date(2023, 1, 10), {"X.NS": 55, "Y.NS": 58}, reason="end_of_period")

        assert len(closed) == 2
        assert p.position_count == 0
        assert all(t["exit_reason"] == "end_of_period" for t in closed)


# ═════════════════════════════════════════════════════════════════════════════
# TestRiskManager
# ═════════════════════════════════════════════════════════════════════════════

class TestRiskManager:

    def test_calculate_stop_loss_price(self):
        risk = RiskConfig(stop_loss_pct=0.02)
        assert calculate_stop_loss_price(2450.0, risk) == pytest.approx(2450 * 0.98)

    def test_calculate_position_size_risk_based(self):
        risk = RiskConfig(risk_per_trade=0.01, stop_loss_pct=0.02)
        qty = calculate_position_size(1_000_000, 2450.0, risk)
        risk_amount = 1_000_000 * 0.01
        per_share_risk = 2450.0 * 0.02
        assert qty == int(risk_amount // per_share_risk)

    def test_calculate_position_size_capped_by_affordability(self):
        risk = RiskConfig(risk_per_trade=0.5, stop_loss_pct=0.02)   # huge risk budget
        qty = calculate_position_size(1000, 2450.0, risk)
        assert qty == 0   # can't even afford 1 share

    def test_calculate_position_size_non_positive_inputs(self):
        risk = RiskConfig()
        assert calculate_position_size(0, 100, risk) == 0
        assert calculate_position_size(1000, 0, risk) == 0

    def test_check_stop_loss_hit(self):
        assert check_stop_loss_hit(2401.0, bar_low_price=2390.0) is True
        assert check_stop_loss_hit(2401.0, bar_low_price=2410.0) is False
        assert check_stop_loss_hit(None, bar_low_price=100) is False

    def test_can_open_new_position(self):
        risk = RiskConfig(max_positions=5)
        assert can_open_new_position(4, risk) is True
        assert can_open_new_position(5, risk) is False

    def test_check_daily_loss_limit(self):
        risk = RiskConfig(max_daily_loss_pct=0.05)
        assert check_daily_loss_limit(100_000, 96_000, risk) is True    # -4%, within limit
        assert check_daily_loss_limit(100_000, 94_000, risk) is False   # -6%, breached
        assert check_daily_loss_limit(100_000, 95_000, risk) is False   # exactly -5%, breached


# ═════════════════════════════════════════════════════════════════════════════
# TestStrategy
# ═════════════════════════════════════════════════════════════════════════════

class TestStrategy:

    def test_buy_signal_no_position_opens(self):
        assert SignalStrategy().decide("BUY", has_position=False) == Action.OPEN

    def test_buy_signal_with_position_holds(self):
        assert SignalStrategy().decide("BUY", has_position=True) == Action.HOLD

    def test_sell_signal_with_position_closes(self):
        assert SignalStrategy().decide("SELL", has_position=True) == Action.CLOSE

    def test_sell_signal_no_position_holds(self):
        assert SignalStrategy().decide("SELL", has_position=False) == Action.HOLD

    def test_hold_signal_always_holds(self):
        assert SignalStrategy().decide("HOLD", has_position=True) == Action.HOLD
        assert SignalStrategy().decide("HOLD", has_position=False) == Action.HOLD


# ═════════════════════════════════════════════════════════════════════════════
# TestEngineSingleSymbol
# ═════════════════════════════════════════════════════════════════════════════

def _flat_prices(dates, price):
    return pd.DataFrame(
        {"open": price, "high": price + 1, "low": price - 1, "close": price},
        index=dates,
    )


class TestEngineSingleSymbol:

    def test_signal_close_produces_trade(self):
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        close = np.array([100, 102, 104, 106, 108, 110, 111, 112, 113, 114], dtype=float)
        prices = pd.DataFrame(
            {"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close},
            index=dates,
        )
        signals = ["BUY"] + ["HOLD"] * 4 + ["SELL"] + ["HOLD"] * 4

        summary = run_backtest("TEST.NS", prices, signals, initial_capital=100_000)

        assert len(summary.trades) == 1
        assert summary.trades[0]["exit_reason"] == "signal"
        assert summary.trades[0]["pnl"] > 0   # price rose the whole way
        assert len(summary.equity_curve) == 10

    def test_stop_loss_triggers_before_signal(self):
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        close = np.array([100, 100, 80, 80, 80, 80, 80, 80, 80, 80], dtype=float)
        low = close - 1
        low[2] = 90   # dips to 90 (below the 98 stop) without closing there
        prices = pd.DataFrame(
            {"open": close, "high": close + 1, "low": low, "close": close}, index=dates
        )
        signals = ["BUY"] + ["HOLD"] * 9

        summary = run_backtest("TEST2.NS", prices, signals, initial_capital=100_000)

        assert len(summary.trades) == 1
        assert summary.trades[0]["exit_reason"] == "stop_loss"
        assert summary.trades[0]["pnl"] < 0
        # exit should be at/near the stop price (98), not the day's close (80)
        assert summary.trades[0]["exit_price"] > 90

    def test_open_position_force_closed_at_end(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        prices = _flat_prices(dates, 100.0)
        signals = ["BUY"] + ["HOLD"] * 4

        summary = run_backtest("TEST3.NS", prices, signals, initial_capital=100_000)

        assert len(summary.trades) == 1
        assert summary.trades[0]["exit_reason"] == "end_of_period"
        assert summary.trades[0]["exit_date"] == dates[-1]

    def test_no_signal_never_opens_a_position(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        prices = _flat_prices(dates, 100.0)
        signals = ["HOLD"] * 5

        summary = run_backtest("TEST4.NS", prices, signals, initial_capital=100_000)

        assert len(summary.trades) == 0
        assert summary.metrics["total_return"] == pytest.approx(0.0)

    def test_mismatched_lengths_raises(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        prices = _flat_prices(dates, 100.0)
        with pytest.raises(ValueError):
            run_backtest("TEST.NS", prices, ["BUY", "HOLD"])   # wrong length

    def test_missing_columns_raises(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        bad_prices = pd.DataFrame({"close": [100, 101, 102]}, index=dates)
        with pytest.raises(ValueError):
            run_backtest("TEST.NS", bad_prices, ["HOLD"] * 3)

    def test_empty_prices_raises(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close"])
        with pytest.raises(ValueError):
            run_backtest("TEST.NS", empty, [])


# ═════════════════════════════════════════════════════════════════════════════
# TestEnginePortfolio
# ═════════════════════════════════════════════════════════════════════════════

class TestEnginePortfolio:

    def test_max_positions_gates_across_symbols(self):
        dates = pd.date_range("2023-01-02", periods=6, freq="B")
        prices = {
            "A.NS": _flat_prices(dates, 100.0),
            "B.NS": _flat_prices(dates, 200.0),
            "C.NS": _flat_prices(dates, 50.0),
        }
        signals = {s: ["BUY"] + ["HOLD"] * 5 for s in prices}

        risk = RiskConfig(max_positions=2)
        result = run_portfolio_backtest(prices, signals, initial_capital=1_000_000, risk=risk)

        opened = [s for s, stats in result["per_symbol"].items() if stats["total_trades"] > 0]
        assert len(opened) == 2, "only 2 of 3 symbols should have opened a position"
        assert opened == ["A.NS", "B.NS"], "dict iteration order should act as tie-break priority"
        assert result["per_symbol"]["C.NS"]["total_trades"] == 0

    def test_per_symbol_stats_are_trade_only_not_equity_metrics(self):
        """
        Regression test: per_symbol entries must NOT contain equity-curve
        metrics (sharpe_ratio, max_drawdown, etc) — those aren't
        well-defined per symbol when capital is pooled, and returning
        them (backed by the shared portfolio equity curve) would show
        identical, misleading numbers for every symbol.
        """
        dates = pd.date_range("2023-01-02", periods=4, freq="B")
        prices = {"A.NS": _flat_prices(dates, 100.0)}
        signals = {"A.NS": ["BUY", "HOLD", "HOLD", "HOLD"]}

        result = run_portfolio_backtest(prices, signals, initial_capital=100_000)

        equity_only_keys = {"sharpe_ratio", "sortino_ratio", "max_drawdown", "annual_return", "total_return"}
        assert equity_only_keys.isdisjoint(result["per_symbol"]["A.NS"].keys())
        assert equity_only_keys <= result["portfolio"].metrics.keys()

    def test_portfolio_summary_combines_all_symbols_trades(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        prices = {
            "A.NS": _flat_prices(dates, 100.0),
            "B.NS": _flat_prices(dates, 50.0),
        }
        signals = {
            "A.NS": ["BUY"] + ["HOLD"] * 4,
            "B.NS": ["BUY"] + ["HOLD"] * 4,
        }

        result = run_portfolio_backtest(prices, signals, initial_capital=1_000_000)
        portfolio_summary = result["portfolio"]

        assert isinstance(portfolio_summary, BacktestSummary)
        assert portfolio_summary.symbol == "PORTFOLIO"
        assert portfolio_summary.metrics["total_trades"] == 2   # both forced-closed at end

    def test_mismatched_calendars_raises(self):
        dates_a = pd.date_range("2023-01-02", periods=5, freq="B")
        dates_b = pd.date_range("2023-02-01", periods=5, freq="B")
        prices = {"A.NS": _flat_prices(dates_a, 100.0), "B.NS": _flat_prices(dates_b, 50.0)}
        signals = {"A.NS": ["HOLD"] * 5, "B.NS": ["HOLD"] * 5}

        with pytest.raises(ValueError):
            run_portfolio_backtest(prices, signals)

    def test_empty_prices_raises(self):
        with pytest.raises(ValueError):
            run_portfolio_backtest({}, {})
