"""Loads the YAML configuration into typed sections."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass
class StrategyConfig:
    entry_prob: float = 0.55
    require_above_sma200: bool = True
    require_market_above_sma200: bool = True
    min_dollar_volume: float = 20e6
    earnings_blackout_days: int = 3
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    time_stop_days: int = 10
    exit_prob: float = 0.42


@dataclass
class RiskConfig:
    initial_equity: float = 100_000.0
    risk_per_trade: float = 0.005
    max_position_pct: float = 0.10
    max_positions: int = 8
    max_sector_pct: float = 0.25
    max_drawdown_halt: float = 0.15
    halt_cooldown_days: int = 20
    vol_target: float | None = None


@dataclass
class CostConfig:
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    max_commission_pct: float = 0.01
    half_spread_bps: float = 2.0
    slippage_coef: float = 0.1
    multiplier: float = 1.0


@dataclass
class ModelConfig:
    horizon: int = 5
    label_threshold: float = 0.002
    label_mode: str = "cross_sectional"
    train_years: int = 5
    retrain_months: int = 3
    embargo_days: int = 2
    calib_frac: float = 0.2
    min_train_rows: int = 5000
    lgbm: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Pass/fail gates for `stockpred evaluate`. Fix these before looking at results."""

    controls: int = 20                 # Permuted-score control backtests.
    seed: int = 0
    max_leak_ic: float = 0.01          # Shuffled-label |mean rank IC| must stay below this.
    max_leak_auc_dev: float = 0.01     # Shuffled-label |AUC - 0.5| must stay below this.
    min_rank_ic: float = 0.02
    min_ic_tstat: float = 2.0
    control_percentile: float = 0.95   # Strategy Sharpe must beat this share of controls.
    stress_cost_mult: float = 2.0
    min_stress_sharpe: float = 0.0
    min_sweep_positive_frac: float = 0.75  # Share of sweep cells with Sharpe > 0.
    max_drawdown: float = 0.25
    min_trades: int = 100


@dataclass
class Config:
    data: dict[str, Any]
    universe: dict[str, str]
    model: ModelConfig
    strategy: StrategyConfig
    risk: RiskConfig
    costs: CostConfig
    sweep: dict[str, list[float]]
    paths: dict[str, str]
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def benchmark(self) -> str:
        return self.data.get("benchmark", "SPY")


def _build(cls, values: dict[str, Any] | None):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    """Reads a YAML config. Missing sections fall back to the defaults."""
    with open(DEFAULT_CONFIG) as f:
        raw = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path) as f:
            override = yaml.safe_load(f) or {}
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(raw.get(key), dict) and key != "universe":
                raw[key] = {**raw[key], **value}
            else:
                raw[key] = value
    raw = copy.deepcopy(raw)
    return Config(
        data=raw.get("data", {}),
        universe=raw.get("universe", {}),
        model=_build(ModelConfig, raw.get("model")),
        strategy=_build(StrategyConfig, raw.get("strategy")),
        risk=_build(RiskConfig, raw.get("risk")),
        costs=_build(CostConfig, raw.get("costs")),
        sweep=raw.get("sweep", {}),
        paths=raw.get("paths", {}),
        evaluation=_build(EvaluationConfig, raw.get("evaluation")),
    )
