"""Event-driven daily backtester, long only.

Order of events on each day t:
  1. At the open: fill exits and entries decided at the close of t-1.
  2. Intraday: a stop fills at the open if the bar gaps through it, else at
     the stop price if the low touches it.
  3. At the close: mark to market, ratchet trailing stops, queue exits
     (time stop, model flip), then queue entries for the next open.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import Config, CostConfig, RiskConfig, StrategyConfig
from .costs import CostModel
from .features import compute_features, market_features
from .risk import DrawdownGuard, initial_stop, size_position, trailing_stop
from .signals import entry_mask, exit_reason


@dataclass
class MarketData:
    """Wide (date x ticker) frames for prices and the features the rules need."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    atr: pd.DataFrame
    sma200: pd.DataFrame
    dollar_vol: pd.DataFrame
    adv: pd.DataFrame
    market_above_sma200: pd.Series
    market_vol: pd.Series
    benchmark_close: pd.Series

    @classmethod
    def from_prices(cls, prices: dict[str, pd.DataFrame], benchmark: str) -> "MarketData":
        market = market_features(prices[benchmark])
        feats = {t: compute_features(df, market) for t, df in prices.items()}

        def wide(source, col):
            return pd.DataFrame({t: source[t][col] for t in prices}).sort_index()

        return cls(
            open=wide(prices, "open"),
            high=wide(prices, "high"),
            low=wide(prices, "low"),
            close=wide(prices, "close"),
            atr=wide(feats, "atr14"),
            sma200=wide(feats, "sma200"),
            dollar_vol=wide(feats, "dollar_vol_med60"),
            adv=wide(feats, "adv20"),
            market_above_sma200=market["mkt_above_sma200"],
            market_vol=market["mkt_vol20"],
            benchmark_close=prices[benchmark]["close"],
        )


@dataclass
class Position:
    ticker: str
    sector: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float
    entry_commission: float
    stop: float
    highest_close: float
    last_close: float
    days_held: int = 0


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    trades: pd.DataFrame
    daily: pd.DataFrame


def run_backtest(
    md: MarketData,
    p_up: pd.DataFrame,
    sectors: dict[str, str],
    strategy: StrategyConfig,
    risk: RiskConfig,
    costs: CostConfig,
) -> BacktestResult:
    """Simulates the strategy on dates covered by the out-of-sample predictions."""
    cost = CostModel(costs)
    dates = p_up.index.intersection(md.close.index).sort_values()
    if len(dates) < 2:
        raise ValueError("Need at least two dates of predictions.")
    p_up = p_up.reindex(index=dates, columns=md.close.columns)
    signals = entry_mask(p_up, md.close, md.sma200, md.dollar_vol, md.market_above_sma200, strategy)

    o, h, l, c = (f.reindex(dates).to_numpy() for f in (md.open, md.high, md.low, md.close))
    atr = md.atr.reindex(dates).to_numpy()
    adv = md.adv.reindex(dates).to_numpy()
    probs = p_up.to_numpy()
    sig = signals.to_numpy()
    mvol = md.market_vol.reindex(dates).to_numpy()
    tickers = list(md.close.columns)
    col = {t: j for j, t in enumerate(tickers)}

    cash = risk.initial_equity
    positions: dict[str, Position] = {}
    pending_exits: dict[str, str] = {}
    pending_entries: list[tuple[str, int, float]] = []  # (ticker, shares, ATR at signal)
    guard = DrawdownGuard(risk)
    trades, daily = [], []

    def close_position(pos: Position, ref_price: float, date, reason: str, j: int) -> None:
        nonlocal cash
        atr_pct = atr[i, j] / ref_price if np.isfinite(atr[i, j]) else np.nan
        fill = cost.fill("sell", ref_price, pos.shares, adv[i, j], atr_pct)
        cash += pos.shares * fill.price - fill.commission
        gross = (ref_price - pos.entry_price) * pos.shares
        net = (fill.price - pos.entry_price) * pos.shares - fill.commission - pos.entry_commission
        trades.append(
            {
                "ticker": pos.ticker,
                "entry_date": pos.entry_date,
                "exit_date": date,
                "shares": pos.shares,
                "entry_price": pos.entry_price,
                "exit_price": fill.price,
                "exit_reason": reason,
                "days_held": pos.days_held,
                "gross_pnl": gross,
                "net_pnl": net,
                "costs": gross - net,
                "return": net / (pos.entry_price * pos.shares),
            }
        )
        del positions[pos.ticker]

    for i, date in enumerate(dates):
        # 1. Open: queued exits, then queued entries.
        for ticker, reason in pending_exits.items():
            j = col[ticker]
            if ticker in positions and np.isfinite(o[i, j]):
                close_position(positions[ticker], o[i, j], date, reason, j)
        pending_exits = {t: r for t, r in pending_exits.items() if t in positions}

        for ticker, shares, atr_sig in pending_entries:
            j = col[ticker]
            if ticker in positions or not np.isfinite(o[i, j]):
                continue
            atr_pct = atr_sig / o[i, j]
            fill = cost.fill("buy", o[i, j], shares, adv[i, j], atr_pct)
            affordable = int((cash - fill.commission) // fill.price)
            if affordable < shares:
                shares = affordable
                if shares <= 0:
                    continue
                fill = cost.fill("buy", o[i, j], shares, adv[i, j], atr_pct)
            cash -= shares * fill.price + fill.commission
            positions[ticker] = Position(
                ticker=ticker,
                sector=sectors.get(ticker, "Unknown"),
                shares=shares,
                entry_date=date,
                entry_price=fill.price,
                entry_commission=fill.commission,
                stop=initial_stop(fill.price, atr_sig, strategy),
                highest_close=fill.price,
                last_close=fill.price,
            )
        pending_entries = []

        # 2. Intraday stops.
        for ticker, pos in list(positions.items()):
            j = col[ticker]
            if not np.isfinite(o[i, j]):
                continue
            if o[i, j] <= pos.stop:
                close_position(pos, o[i, j], date, "stop_gap", j)
            elif l[i, j] <= pos.stop:
                close_position(pos, pos.stop, date, "stop", j)

        # 3. Close: mark to market, trail stops, queue exits.
        for ticker, pos in positions.items():
            j = col[ticker]
            if np.isfinite(c[i, j]):
                pos.last_close = c[i, j]
                pos.highest_close = max(pos.highest_close, c[i, j])
                pos.days_held += 1
                pos.stop = trailing_stop(pos.stop, pos.highest_close, atr[i, j], strategy)
            reason = exit_reason(pos.days_held, probs[i, j], strategy)
            if reason:
                pending_exits[ticker] = reason
        equity = cash + sum(p.shares * p.last_close for p in positions.values())
        can_enter = guard.update(equity)
        daily.append(
            {
                "date": date,
                "equity": equity,
                "cash": cash,
                "exposure": (equity - cash) / equity if equity > 0 else 0.0,
                "positions": len(positions),
                "halted": not can_enter,
            }
        )

        # Queue entries for the next open.
        if not can_enter or i == len(dates) - 1:
            continue
        staying = [p for t, p in positions.items() if t not in pending_exits]
        slots = risk.max_positions - len(staying)
        if slots <= 0:
            continue
        sector_exposure: dict[str, float] = {}
        for p in staying:
            sector_exposure[p.sector] = sector_exposure.get(p.sector, 0.0) + p.shares * p.last_close
        free_cash = cash + sum(positions[t].shares * positions[t].last_close for t in pending_exits)
        candidates = [
            tickers[j]
            for j in np.argsort(-np.nan_to_num(probs[i], nan=-1.0))
            if sig[i, j] and tickers[j] not in positions
        ]
        for ticker in candidates[:slots]:
            j = col[ticker]
            price, a = c[i, j], atr[i, j]
            if not (np.isfinite(price) and np.isfinite(a) and a > 0):
                continue
            sector = sectors.get(ticker, "Unknown")
            shares = size_position(
                equity=equity,
                cash=free_cash,
                entry_price=price,
                stop_price=initial_stop(price, a, strategy),
                sector_exposure=sector_exposure.get(sector, 0.0),
                cfg=risk,
                market_vol=mvol[i],
            )
            if shares <= 0:
                continue
            pending_entries.append((ticker, shares, a))
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + shares * price
            free_cash -= shares * price * 1.002

    # Liquidate at the final close so every trade is realized.
    last = len(dates) - 1
    i = last
    for ticker, pos in list(positions.items()):
        close_position(pos, pos.last_close, dates[last], "end_of_test", col[ticker])
    daily[-1]["equity"] = cash
    daily[-1]["cash"] = cash

    daily_df = pd.DataFrame(daily).set_index("date")
    bench = md.benchmark_close.reindex(dates).ffill()
    bench = risk.initial_equity * bench / bench.iloc[0]
    trades_df = pd.DataFrame(trades)
    return BacktestResult(daily_df["equity"], bench, trades_df, daily_df)


def sensitivity(
    md: MarketData, p_up: pd.DataFrame, cfg: Config, grid: dict[str, list[float]]
) -> pd.DataFrame:
    """Re-runs the backtest over a parameter grid to show how fragile the result is."""
    from .metrics import performance

    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        strat = replace(cfg.strategy, **dict(zip(keys, values)))
        res = run_backtest(md, p_up, cfg.universe, strat, cfg.risk, cfg.costs)
        perf = performance(res.equity, res.trades)
        rows.append({**dict(zip(keys, values)), **{k: perf[k] for k in ("cagr", "sharpe", "max_drawdown", "trades")}})
    return pd.DataFrame(rows)


def predictions_to_wide(preds: pd.DataFrame, column: str = "p_up") -> pd.DataFrame:
    return preds[column].unstack("ticker").sort_index()
