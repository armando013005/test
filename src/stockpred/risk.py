"""Stops, position sizing, and portfolio limits."""

from __future__ import annotations

import math

from .config import RiskConfig, StrategyConfig


def initial_stop(entry_price: float, atr: float, cfg: StrategyConfig) -> float:
    return entry_price - cfg.stop_atr * atr


def trailing_stop(current_stop: float, highest_close: float, atr: float, cfg: StrategyConfig) -> float:
    """Ratchets the stop up. It never moves down."""
    if not math.isfinite(atr):
        return current_stop
    return max(current_stop, highest_close - cfg.trail_atr * atr)


def risk_fraction(cfg: RiskConfig, market_vol: float | None = None) -> float:
    """Per-trade risk, scaled down when market volatility exceeds the target."""
    frac = cfg.risk_per_trade
    if cfg.vol_target and market_vol and math.isfinite(market_vol) and market_vol > 0:
        frac *= min(1.0, cfg.vol_target / market_vol)
    return frac


def size_position(
    equity: float,
    cash: float,
    entry_price: float,
    stop_price: float,
    sector_exposure: float,
    cfg: RiskConfig,
    market_vol: float | None = None,
    cash_buffer: float = 0.002,
) -> int:
    """Shares to buy so a stop-out loses `risk_per_trade` of equity, subject to every cap.

    shares = equity * risk_fraction / (entry - stop), then capped by the
    per-position limit, the remaining sector budget, and available cash.
    """
    risk_per_share = entry_price - stop_price
    if entry_price <= 0 or risk_per_share <= 0 or equity <= 0:
        return 0
    by_risk = equity * risk_fraction(cfg, market_vol) / risk_per_share
    by_position = equity * cfg.max_position_pct / entry_price
    by_sector = max(equity * cfg.max_sector_pct - sector_exposure, 0.0) / entry_price
    by_cash = max(cash, 0.0) / (entry_price * (1 + cash_buffer))
    return max(int(math.floor(min(by_risk, by_position, by_sector, by_cash))), 0)


class DrawdownGuard:
    """Blocks new entries while equity sits more than `max_drawdown_halt` below its peak.

    After `halt_cooldown_days` consecutive halted days the guard re-arms: the
    peak resets to current equity and entries resume. Without the reset, an
    account that has closed its positions sits flat in cash, can never climb
    back above the threshold, and stops trading forever.
    """

    def __init__(self, cfg: RiskConfig, peak: float | None = None, halted_days: int = 0):
        self.limit = cfg.max_drawdown_halt
        self.cooldown = cfg.halt_cooldown_days
        self.peak = peak if peak is not None else cfg.initial_equity
        self.halted_days = halted_days

    def update(self, equity: float) -> bool:
        """Records one day's equity and returns True when new entries are allowed."""
        self.peak = max(self.peak, equity)
        if equity >= self.peak * (1 - self.limit):
            self.halted_days = 0
            return True
        self.halted_days += 1
        if self.cooldown and self.halted_days > self.cooldown:
            self.peak = equity
            self.halted_days = 0
            return True
        return False
