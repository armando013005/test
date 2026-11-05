"""Entry and exit rules. Each rule reads only information available at the signal close."""

from __future__ import annotations

import pandas as pd

from .config import StrategyConfig


def entry_mask(
    p_up: pd.DataFrame,
    close: pd.DataFrame,
    sma200: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    market_above_sma200: pd.Series,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """Wide (date x ticker) boolean frame. True means every entry condition holds."""
    mask = p_up >= cfg.entry_prob
    mask &= dollar_vol.reindex_like(p_up) >= cfg.min_dollar_volume
    if cfg.require_above_sma200:
        mask &= close.reindex_like(p_up) > sma200.reindex_like(p_up)
    if cfg.require_market_above_sma200:
        regime = market_above_sma200.reindex(p_up.index).fillna(False).astype(bool)
        mask = mask.mul(regime, axis=0).astype(bool)
    return mask.fillna(False).astype(bool)


def entry_ok(row: pd.Series, cfg: StrategyConfig) -> bool:
    """Scalar version of `entry_mask` for live scoring of one ticker."""
    checks = [
        row["p_up"] >= cfg.entry_prob,
        row["dollar_vol_med60"] >= cfg.min_dollar_volume,
    ]
    if cfg.require_above_sma200:
        checks.append(row["close"] > row["sma200"])
    if cfg.require_market_above_sma200:
        checks.append(bool(row["mkt_above_sma200"]))
    return bool(all(pd.notna(c) and bool(c) for c in checks))


def exit_reason(days_held: int, p_up: float, cfg: StrategyConfig) -> str | None:
    """Close-of-day exits that fill at the next open. Stops are handled intraday."""
    if days_held >= cfg.time_stop_days:
        return "time_stop"
    if pd.notna(p_up) and p_up < cfg.exit_prob:
        return "model_flip"
    return None
