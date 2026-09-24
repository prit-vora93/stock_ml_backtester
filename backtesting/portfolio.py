"""
backtesting/portfolio.py
--------------------------
Tracks cash, open positions, and equity over time during a backtest.
Multi-symbol by design (positions is keyed by symbol) so the same
Portfolio can back either a single-symbol backtest or a shared-capital
multi-symbol one (see backtesting/engine.py's run_portfolio_backtest()).

Commission (settings.COMMISSION_PCT) and slippage (settings.SLIPPAGE_PCT)
are applied on every fill: slippage worsens the fill price in the
direction that hurts you (higher on buys, lower on sells), commission
is charged on both entry and exit — both baked into a trade's pnl so
metrics reflect realistic, not idealized, execution.

Usage:
    from backtesting.portfolio import Portfolio

    portfolio = Portfolio(initial_capital=1_000_000)
    portfolio.open_position("RELIANCE.NS", date(2023,1,3), price=2450.0,
                             quantity=40, stop_loss_price=2401.0)
    ...
    trade = portfolio.close_position("RELIANCE.NS", date(2023,1,10),
                                      price=2510.0, reason="signal")
    equity = portfolio.mark_to_market(date(2023,1,10), {"RELIANCE.NS": 2510.0})
"""

from dataclasses import dataclass
from typing import Optional

from config.settings import INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT
from utils.logger import logger


@dataclass
class Position:
    """An open long position. Long-only — this project doesn't short."""
    symbol:            str
    entry_date:        object   # date
    entry_price:       float    # fill price, slippage already applied
    quantity:           int
    stop_loss_price:    Optional[float]
    entry_commission:  float


class Portfolio:
    """
    Tracks cash, open positions (by symbol), closed trades, and the
    equity curve over the course of a backtest.
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission_pct:  float = COMMISSION_PCT,
        slippage_pct:    float = SLIPPAGE_PCT,
    ):
        self.initial_capital = float(initial_capital)
        self.commission_pct  = commission_pct
        self.slippage_pct    = slippage_pct

        self.cash:      float = float(initial_capital)
        self.positions: dict[str, Position] = {}
        self.trades:    list[dict] = []
        self.equity_curve: list[tuple] = []   # [(date, equity), ...]

    # ── Queries ────────────────────────────────────────────────────────────

    @property
    def position_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def position_value(self, prices: dict[str, float]) -> float:
        """Mark-to-market value of all open positions (excludes cash)."""
        total = 0.0
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.entry_price)
            total += pos.quantity * price
        return total

    def total_equity(self, prices: dict[str, float]) -> float:
        """cash + mark-to-market value of open positions."""
        return self.cash + self.position_value(prices)

    # ── Actions ────────────────────────────────────────────────────────────

    def open_position(
        self,
        symbol:          str,
        date,
        price:           float,
        quantity:        int,
        stop_loss_price: Optional[float] = None,
    ) -> bool:
        """
        Opens a long position, applying slippage (fills worse than the
        quoted price) and commission. Fails (returns False) if quantity
        is non-positive, a position in this symbol is already open, or
        there isn't enough cash to cover the fill + commission.

        Returns:
            True if the position was opened, False otherwise.
        """
        if quantity <= 0:
            logger.warning(f"{symbol}: cannot open position with quantity={quantity}")
            return False

        if symbol in self.positions:
            logger.warning(f"{symbol}: position already open — cannot open a second one")
            return False

        fill_price = price * (1 + self.slippage_pct)
        cost        = fill_price * quantity
        commission  = cost * self.commission_pct
        total_cost  = cost + commission

        if total_cost > self.cash:
            logger.warning(
                f"{symbol}: insufficient cash to open position "
                f"(need {total_cost:.2f}, have {self.cash:.2f})"
            )
            return False

        self.cash -= total_cost
        self.positions[symbol] = Position(
            symbol=symbol, entry_date=date, entry_price=fill_price,
            quantity=quantity, stop_loss_price=stop_loss_price,
            entry_commission=commission,
        )

        logger.info(
            f"{symbol}: OPENED {quantity} @ {fill_price:.2f} "
            f"(commission={commission:.2f}) on {date}"
        )
        return True

    def close_position(
        self,
        symbol: str,
        date,
        price:  float,
        reason: str = "signal",
    ) -> Optional[dict]:
        """
        Closes an open position, applying slippage (fills worse than the
        quoted price) and commission on the exit leg too.

        Args:
            symbol: Position to close
            date:   Exit date
            price:  Quoted exit price (before slippage)
            reason: "signal" | "stop_loss" | "end_of_period"
                    (matches api.database.Trade.exit_reason)

        Returns:
            Trade dict (matching api.database.Trade's columns, minus
            backtest_id/symbol which the caller attaches), or None if
            no position was open for this symbol.
        """
        pos = self.positions.pop(symbol, None)
        if pos is None:
            logger.warning(f"{symbol}: no open position to close")
            return None

        fill_price     = price * (1 - self.slippage_pct)
        proceeds        = fill_price * pos.quantity
        exit_commission = proceeds * self.commission_pct
        net_proceeds     = proceeds - exit_commission

        self.cash += net_proceeds

        entry_cost = pos.entry_price * pos.quantity + pos.entry_commission
        pnl        = net_proceeds - entry_cost
        pnl_pct    = pnl / entry_cost if entry_cost > 0 else 0.0

        trade = {
            "symbol":      symbol,
            "entry_date":  pos.entry_date,
            "entry_price": pos.entry_price,
            "exit_date":   date,
            "exit_price":  fill_price,
            "quantity":    pos.quantity,
            "pnl":         pnl,
            "pnl_pct":     pnl_pct,
            "exit_reason": reason,
        }
        self.trades.append(trade)

        logger.info(
            f"{symbol}: CLOSED {pos.quantity} @ {fill_price:.2f} "
            f"pnl={pnl:+.2f} ({pnl_pct:+.1%}) reason={reason} on {date}"
        )
        return trade

    def force_close_all(self, date, prices: dict[str, float], reason: str = "end_of_period") -> list[dict]:
        """
        Closes every remaining open position at the given date's prices —
        called at the end of a backtest so nothing is left "in the air".

        Args:
            date:   Close date
            prices: {symbol: price} for every currently open position
                    (falls back to entry_price if a symbol is missing,
                    same as mark-to-market)
            reason: exit_reason recorded on each resulting trade

        Returns:
            List of trade dicts for every position that was closed.
        """
        closed = []
        for symbol in list(self.positions.keys()):
            price = prices.get(symbol, self.positions[symbol].entry_price)
            trade = self.close_position(symbol, date, price, reason=reason)
            if trade:
                closed.append(trade)
        return closed

    def mark_to_market(self, date, prices: dict[str, float]) -> float:
        """
        Records today's total equity (cash + position value) in the
        equity curve and returns it.
        """
        equity = self.total_equity(prices)
        self.equity_curve.append((date, equity))
        return equity
