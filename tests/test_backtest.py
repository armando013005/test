from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stockpred.backtest import MarketData, run_backtest
from stockpred.config import CostConfig, RiskConfig, StrategyConfig
from stockpred.data import synthetic_prices

ZERO_COSTS = CostConfig(commission_per_share=0, min_commission=0, half_spread_bps=0, slippage_coef=0)
LOOSE = StrategyConfig(
    entry_prob=0.5, require_above_sma200=False, require_market_above_sma200=False, min_dollar_volume=0,
    time_stop_days=1000, exit_prob=0.0,
)


def _flat_md(tickers, n=300):
    prices = synthetic_prices(tickers, n_days=n, seed=3)
    return prices, MarketData.from_prices(prices, tickers[0])


def _signal_once(md, ticker, day):
    p = pd.DataFrame(0.0, index=md.close.index, columns=md.close.columns)
    p.iloc[day, p.columns.get_loc(ticker)] = 0.9
    return p


def test_entry_fills_next_open_and_stop_gap_fills_at_open():
    prices, md = _flat_md(["SPY", "AAA"])
    day = 100
    p = _signal_once(md, "AAA", day)
    # Force a gap down through any stop two days after entry.
    gap_day = md.close.index[day + 2]
    for f in (md.open, md.high, md.low, md.close):
        f.loc[gap_day:, "AAA"] *= 0.5
    res = run_backtest(md, p, {}, LOOSE, RiskConfig(), ZERO_COSTS)
    t = res.trades.iloc[0]
    assert t["entry_date"] == md.close.index[day + 1]
    assert t["entry_price"] == pytest.approx(md.open["AAA"].iloc[day + 1])
    assert t["exit_reason"] == "stop_gap"
    assert t["exit_price"] == pytest.approx(md.open.loc[gap_day, "AAA"])


def test_intraday_stop_fills_at_stop_price():
    prices, md = _flat_md(["SPY", "AAA"])
    day = 100
    p = _signal_once(md, "AAA", day)
    hit = md.close.index[day + 3]
    md.low.loc[hit, "AAA"] = md.open.loc[hit, "AAA"] * 0.5
    res = run_backtest(md, p, {}, LOOSE, RiskConfig(), ZERO_COSTS)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["exit_date"] == hit
    assert t["exit_price"] > md.low.loc[hit, "AAA"]


def test_time_stop_exits_at_open_after_n_days():
    prices, md = _flat_md(["SPY", "AAA"])
    p = _signal_once(md, "AAA", 100)
    strat = replace(LOOSE, time_stop_days=3, stop_atr=50, trail_atr=50)
    res = run_backtest(md, p, {}, strat, RiskConfig(), ZERO_COSTS)
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "time_stop"
    assert t["days_held"] == 3
    assert t["exit_date"] == md.close.index[100 + 4]


def test_costs_reduce_pnl_and_equity_accounting_balances():
    prices, md = _flat_md(["SPY", "AAA", "BBB"], n=600)
    rng = np.random.default_rng(0)
    p = pd.DataFrame(rng.uniform(0.3, 0.8, md.close.shape), index=md.close.index, columns=md.close.columns)
    free = run_backtest(md, p, {}, replace(LOOSE, exit_prob=0.45), RiskConfig(), ZERO_COSTS)
    paid = run_backtest(md, p, {}, replace(LOOSE, exit_prob=0.45), RiskConfig(), CostConfig())
    assert paid.trades["costs"].sum() > 0
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    # Final equity equals starting equity plus realized net P&L.
    assert paid.equity.iloc[-1] == pytest.approx(RiskConfig().initial_equity + paid.trades["net_pnl"].sum())


def test_position_limits_hold():
    tickers = ["SPY", *[f"T{i}" for i in range(15)]]
    prices, md = _flat_md(tickers, n=400)
    p = pd.DataFrame(0.9, index=md.close.index, columns=md.close.columns)
    risk = RiskConfig(max_positions=4)
    res = run_backtest(md, p, {t: "X" for t in tickers}, LOOSE, risk, ZERO_COSTS)
    assert res.daily["positions"].max() <= 4
    assert res.daily["exposure"].max() <= risk.max_sector_pct + 0.05  # One sector, 25% cap plus drift.
