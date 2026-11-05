import pandas as pd
import pytest

from stockpred.broker import LiveTradingRefused, _check_paper_only
from stockpred.live import Holding, plan_orders


def _snap(rows):
    base = {"p_up": 0.5, "exp_ret": 0.0, "close": 100.0, "atr14": 2.0, "sma200": 90.0,
            "dollar_vol_med60": 1e9, "mkt_above_sma200": True, "mkt_vol20": 0.15}
    return pd.DataFrame({t: {**base, **r} for t, r in rows.items()}).T.infer_objects()


CAL = pd.bdate_range("2024-01-01", periods=30)


def test_entry_respects_rules_and_sizing(cfg):
    snap = _snap({"AAPL": {"p_up": 0.7}, "MSFT": {"p_up": 0.7, "close": 80.0}, "NVDA": {"p_up": 0.5}})
    orders, holdings, _ = plan_orders(snap, CAL, {}, 100_000, 100_000, 100_000, cfg)
    buys = {o.ticker: o for o in orders if o.action == "buy"}
    assert set(buys) == {"AAPL"}  # MSFT is below its 200-day SMA, NVDA below threshold.
    assert buys["AAPL"].stop == pytest.approx(96.0)
    assert buys["AAPL"].shares == 100  # $10k position cap.
    assert "AAPL" in holdings


def test_regime_filter_blocks_entries(cfg):
    snap = _snap({"AAPL": {"p_up": 0.9, "mkt_above_sma200": False}})
    orders, _, _ = plan_orders(snap, CAL, {}, 100_000, 100_000, 100_000, cfg)
    assert orders == []


def test_time_stop_flip_and_trail(cfg):
    held = {
        "AAPL": Holding(10, str(CAL[-11].date()), 90.0, 100.0),   # 10 days held -> time stop.
        "MSFT": Holding(10, str(CAL[-3].date()), 90.0, 100.0),    # Model flip.
        "NVDA": Holding(10, str(CAL[-3].date()), 90.0, 120.0),    # Trail to 120 - 6 = 114? above close -> stop.
        "GOOGL": Holding(10, str(CAL[-3].date()), 90.0, 105.0),   # Trail to 99.
    }
    snap = _snap({"AAPL": {"p_up": 0.6}, "MSFT": {"p_up": 0.3}, "NVDA": {"p_up": 0.6}, "GOOGL": {"p_up": 0.6}})
    orders, holdings, _ = plan_orders(snap, CAL, held, 100_000, 50_000, 100_000, cfg)
    by = {o.ticker: o for o in orders}
    assert by["AAPL"].reason == "time_stop"
    assert by["MSFT"].reason == "model_flip"
    assert by["NVDA"].reason == "stop"
    assert by["GOOGL"].action == "update_stop" and by["GOOGL"].stop == pytest.approx(99.0)
    assert set(holdings) == {"GOOGL"}


def test_drawdown_halt_blocks_entries(cfg):
    snap = _snap({"AAPL": {"p_up": 0.9}})
    orders, _, peak = plan_orders(snap, CAL, {}, 80_000, 80_000, 100_000, cfg)
    assert orders == [] and peak == 100_000


def test_live_endpoint_refused(monkeypatch):
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    with pytest.raises(LiveTradingRefused):
        _check_paper_only()
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    _check_paper_only()
