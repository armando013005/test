"""Turns today's model scores into orders using the same rules as the backtest."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .config import Config
from .risk import DrawdownGuard, initial_stop, size_position, trailing_stop
from .signals import entry_ok, exit_reason

log = logging.getLogger(__name__)


@dataclass
class Holding:
    shares: int
    signal_date: str      # Close that generated the entry; the fill is the next open.
    stop: float
    highest_close: float


@dataclass
class Order:
    action: str           # "buy", "sell", or "update_stop".
    ticker: str
    shares: int
    ref_price: float
    stop: float | None = None
    reason: str = ""


def load_state(path: str | Path, initial_equity: float) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"peak_equity": initial_equity, "holdings": {}}


def save_state(path: str | Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


def days_held(calendar: pd.DatetimeIndex, signal_date: str) -> int:
    """Trading days since the fill, counting the fill day."""
    return int((calendar > pd.Timestamp(signal_date)).sum())


def has_upcoming_earnings(ticker: str, days: int) -> bool:
    """Best effort lookup. Returns False when the vendor has no data."""
    if days <= 0:
        return False
    try:
        import yfinance as yf

        dates = yf.Ticker(ticker).get_earnings_dates(limit=4)
        if dates is None or dates.empty:
            return False
        idx = pd.DatetimeIndex(dates.index).tz_localize(None)
        now = pd.Timestamp.now().normalize()
        return bool(((idx >= now) & (idx <= now + pd.tseries.offsets.BDay(days))).any())
    except Exception as exc:
        log.warning("Earnings lookup failed for %s: %s", ticker, exc)
        return False


def plan_orders(
    snapshot: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    holdings: dict[str, Holding],
    equity: float,
    cash: float,
    peak_equity: float,
    cfg: Config,
    blackout: set[str] | None = None,
) -> tuple[list[Order], dict[str, Holding], float]:
    """Applies exits, stop updates, and entries to the latest close.

    `snapshot` is indexed by ticker and holds p_up, exp_ret, and the feature
    columns for the last trading day. Returns orders, the updated holdings,
    and the updated peak equity.
    """
    s, r = cfg.strategy, cfg.risk
    blackout = blackout or set()
    asof = calendar[-1]
    orders: list[Order] = []
    keep: dict[str, Holding] = {}
    freed_cash = 0.0

    for ticker, h in holdings.items():
        if ticker not in snapshot.index:
            log.warning("No data for held %s; leaving it untouched", ticker)
            keep[ticker] = h
            continue
        row = snapshot.loc[ticker]
        h.highest_close = max(h.highest_close, float(row["close"]))
        new_stop = trailing_stop(h.stop, h.highest_close, float(row["atr14"]), s)
        reason = exit_reason(days_held(calendar, h.signal_date), float(row["p_up"]), s)
        if reason or row["close"] <= new_stop:
            orders.append(Order("sell", ticker, h.shares, float(row["close"]), reason=reason or "stop"))
            freed_cash += h.shares * float(row["close"])
            continue
        if new_stop > h.stop:
            orders.append(Order("update_stop", ticker, h.shares, float(row["close"]), stop=new_stop, reason="trail"))
        h.stop = new_stop
        keep[ticker] = h

    guard = DrawdownGuard(r, peak=peak_equity)
    if not guard.update(equity):
        log.warning("Drawdown halt: equity %.0f is below %.0f%% of peak %.0f", equity, 100 * (1 - r.max_drawdown_halt), guard.peak)
        return orders, keep, guard.peak

    slots = r.max_positions - len(keep)
    sector_exposure: dict[str, float] = {}
    for ticker, h in keep.items():
        sector = cfg.universe.get(ticker, "Unknown")
        price = float(snapshot.loc[ticker, "close"]) if ticker in snapshot.index else h.highest_close
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + h.shares * price
    free_cash = cash + freed_cash

    ranked = snapshot.sort_values("p_up", ascending=False)
    for ticker, row in ranked.iterrows():
        if slots <= 0:
            break
        if ticker in keep or ticker in holdings or ticker in blackout or not entry_ok(row, s):
            continue
        price, atr = float(row["close"]), float(row["atr14"])
        stop = initial_stop(price, atr, s)
        sector = cfg.universe.get(ticker, "Unknown")
        shares = size_position(
            equity, free_cash, price, stop, sector_exposure.get(sector, 0.0), r, float(row.get("mkt_vol20", float("nan")))
        )
        if shares <= 0:
            continue
        orders.append(Order("buy", ticker, shares, price, stop=stop, reason=f"p_up={row['p_up']:.3f}"))
        keep[ticker] = Holding(shares, str(asof.date()), stop, price)
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + shares * price
        free_cash -= shares * price * 1.002
        slots -= 1
    return orders, keep, guard.peak


def holdings_to_json(holdings: dict[str, Holding]) -> dict:
    return {t: asdict(h) for t, h in holdings.items()}


def holdings_from_json(raw: dict) -> dict[str, Holding]:
    return {t: Holding(**h) for t, h in raw.items()}
