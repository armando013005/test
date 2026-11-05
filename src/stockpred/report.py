"""Writes backtest results to disk: metrics, trade log, equity curve, and a chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .backtest import BacktestResult  # noqa: E402
from .metrics import performance  # noqa: E402


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return v


def write_report(result: BacktestResult, out_dir: str | Path, label: str = "backtest") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    strat = performance(result.equity, result.trades)
    bench = performance(result.benchmark)
    metrics = {"strategy": strat, "benchmark_buy_and_hold": bench}

    with open(out / f"{label}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    result.trades.to_csv(out / f"{label}_trades.csv", index=False)
    result.daily.join(result.benchmark.rename("benchmark")).to_csv(out / f"{label}_equity.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1])
    ax1.plot(result.equity.index, result.equity, label="Strategy (net of costs)", color="#1f5fa8")
    ax1.plot(result.benchmark.index, result.benchmark, label="Benchmark buy and hold", color="#999999")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(alpha=0.3)
    dd = result.equity / result.equity.cummax() - 1
    ax2.fill_between(dd.index, dd, 0, color="#c0392b", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{label}_equity.png", dpi=120)
    plt.close(fig)
    return metrics


def print_metrics(metrics: dict) -> None:
    keys = ["start", "end", "total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown",
            "calmar", "trades", "hit_rate", "profit_factor", "avg_days_held", "total_costs", "annual_turnover"]
    strat, bench = metrics["strategy"], metrics["benchmark_buy_and_hold"]
    print(f"{'metric':<18}{'strategy':>14}{'benchmark':>14}")
    for k in keys:
        if k in strat or k in bench:
            print(f"{k:<18}{_fmt(strat.get(k, '')):>14}{_fmt(bench.get(k, '')):>14}")
    if "exit_reasons" in strat:
        print("exit reasons:", strat["exit_reasons"])
