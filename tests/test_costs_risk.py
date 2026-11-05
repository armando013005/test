import math

import pytest

from stockpred.config import CostConfig, RiskConfig, StrategyConfig
from stockpred.costs import CostModel
from stockpred.risk import DrawdownGuard, initial_stop, risk_fraction, size_position, trailing_stop


def test_commission_minimum_per_share_and_cap():
    m = CostModel(CostConfig())
    assert m.commission(10, 100.0) == pytest.approx(1.0)          # Minimum.
    assert m.commission(1000, 100.0) == pytest.approx(5.0)        # $0.005/share.
    assert m.commission(10, 1.0) == pytest.approx(0.10)           # 1% of $10 trade value.


def test_fill_applies_spread_and_slippage_against_trader():
    m = CostModel(CostConfig())
    buy = m.fill("buy", 100.0, 1000, adv_shares=1e6, atr_pct=0.02)
    sell = m.fill("sell", 100.0, 1000, adv_shares=1e6, atr_pct=0.02)
    impact = 2e-4 + 0.1 * 0.02 * math.sqrt(1000 / 1e6)
    assert buy.price == pytest.approx(100.0 * (1 + impact))
    assert sell.price == pytest.approx(100.0 * (1 - impact))


def test_cost_multiplier_scales_everything():
    base = CostModel(CostConfig()).fill("buy", 100.0, 1000, 1e6, 0.02)
    stressed = CostModel(CostConfig(multiplier=2.0)).fill("buy", 100.0, 1000, 1e6, 0.02)
    assert stressed.cost_vs(100.0, 1000) == pytest.approx(2 * base.cost_vs(100.0, 1000), rel=1e-3)


def test_size_by_risk():
    # Risk 0.5% of 100k = $500. Stop $2 away -> 250 shares ($25k, under the 10% cap? no: capped).
    shares = size_position(100_000, 100_000, 100.0, 98.0, 0.0, RiskConfig())
    assert shares == 100  # 10% position cap: $10k / $100.
    shares = size_position(100_000, 100_000, 100.0, 90.0, 0.0, RiskConfig())
    assert shares == 50   # Risk: $500 / $10.


def test_size_respects_sector_cash_and_bad_stops():
    r = RiskConfig()
    assert size_position(100_000, 100_000, 100.0, 90.0, 24_000.0, r) == 10   # $1k sector room left.
    assert size_position(100_000, 1_000, 100.0, 90.0, 0.0, r) == 9           # Cash limited.
    assert size_position(100_000, 100_000, 100.0, 101.0, 0.0, r) == 0        # Stop above entry.


def test_vol_target_scales_risk_down_only():
    r = RiskConfig(vol_target=0.15)
    assert risk_fraction(r, 0.30) == pytest.approx(0.0025)
    assert risk_fraction(r, 0.10) == pytest.approx(0.005)


def test_stops():
    s = StrategyConfig(stop_atr=2.0, trail_atr=3.0)
    assert initial_stop(100.0, 2.0, s) == 96.0
    assert trailing_stop(96.0, 110.0, 2.0, s) == 104.0
    assert trailing_stop(104.0, 105.0, 2.0, s) == 104.0  # Never moves down.


def test_drawdown_guard():
    g = DrawdownGuard(RiskConfig(initial_equity=100.0, max_drawdown_halt=0.15))
    assert g.update(120.0)
    assert g.update(103.0)
    assert not g.update(101.0)  # 120 * 0.85 = 102.
    assert g.update(110.0)
