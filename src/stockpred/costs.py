"""Transaction cost model: commission, half spread, and size-dependent slippage."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import CostConfig


@dataclass(frozen=True)
class Fill:
    price: float        # Execution price after spread and slippage.
    commission: float

    def cost_vs(self, ref_price: float, shares: int) -> float:
        """Total dollars lost to costs relative to the reference price."""
        return abs(self.price - ref_price) * shares + self.commission


class CostModel:
    def __init__(self, cfg: CostConfig):
        self.cfg = cfg

    def commission(self, shares: int, price: float) -> float:
        c = self.cfg
        raw = max(c.min_commission, c.commission_per_share * shares)
        return min(raw, c.max_commission_pct * shares * price) * c.multiplier

    def impact_frac(self, shares: int, adv_shares: float, atr_pct: float) -> float:
        """Price impact as a fraction of price: half spread plus sqrt-participation slippage."""
        c = self.cfg
        participation = shares / adv_shares if adv_shares and adv_shares > 0 else 1.0
        atr_pct = atr_pct if atr_pct and math.isfinite(atr_pct) else 0.02
        slippage = c.slippage_coef * atr_pct * math.sqrt(min(participation, 1.0))
        return (c.half_spread_bps / 1e4 + slippage) * c.multiplier

    def fill(self, side: str, ref_price: float, shares: int, adv_shares: float, atr_pct: float) -> Fill:
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        impact = self.impact_frac(shares, adv_shares, atr_pct)
        price = ref_price * (1 + impact) if side == "buy" else ref_price * (1 - impact)
        return Fill(price=price, commission=self.commission(shares, price))
