"""Decides whether the model has a real, tradable edge, using gates fixed in advance.

The evaluation compares the strategy against controls that trade the same
universe with the same rules but no skill:

- Shuffled-label model: retrained on shuffled labels. Any edge it shows
  means information is leaking into the features.
- Permuted-score controls: the real scores, reassigned to the wrong tickers
  within each retrain block. Each score series keeps its persistence, so
  turnover and exposure match the real strategy, and the controls benefit
  from the hand-picked universe just as much. Beating them measures stock
  selection skill rather than survivorship bias.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult, MarketData, predictions_to_wide, run_backtest, sensitivity
from .config import Config
from .metrics import performance
from .model import oos_metrics
from .report import write_report

log = logging.getLogger(__name__)


def permuted_scores(p_up: pd.DataFrame, block_months: int, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffles which ticker each score column belongs to, separately in each block."""
    out = p_up.copy()
    starts = pd.date_range(p_up.index[0], p_up.index[-1], freq=pd.DateOffset(months=block_months))
    edges = [*starts, p_up.index[-1] + pd.Timedelta(days=1)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = (p_up.index >= lo) & (p_up.index < hi)
        block = p_up.loc[rows]
        cols = block.columns[block.notna().any()]
        if len(cols) < 2:
            continue
        out.loc[rows, cols] = block[rng.permutation(cols)].to_numpy()
    return out


def equal_weight_benchmark(md: MarketData, dates: pd.DatetimeIndex, tickers: list[str], initial: float) -> pd.Series:
    """Daily rebalanced equal-weight buy-and-hold of the tradable universe, without costs."""
    rets = md.close[tickers].reindex(dates).pct_change(fill_method=None)
    ew = rets.mean(axis=1, skipna=True).fillna(0.0)
    return (initial * (1 + ew).cumprod()).rename("equal_weight")


def _summary(result: BacktestResult) -> dict:
    perf = performance(result.equity, result.trades)
    perf["avg_exposure"] = float(result.daily["exposure"].mean())
    perf["halted_days"] = int(result.daily["halted"].sum())
    return perf


def _finite(x) -> float:
    return float(x) if x is not None and np.isfinite(x) else float("-inf")


def score_gates(
    model_m: dict, shuffled_m: dict, base: dict, stress: dict, control_sharpes: list[float],
    sweep: pd.DataFrame, ev,
) -> list[dict]:
    """Applies the pre-registered pass/fail rules. Returns one row per gate."""
    gates = []

    def gate(name: str, passed: bool, detail: str) -> None:
        gates.append({"gate": name, "pass": bool(passed), "detail": detail})

    leak_ic, leak_auc = shuffled_m["mean_daily_rank_ic"], shuffled_m["auc"]
    gate(
        "Leakage",
        abs(leak_ic) < ev.max_leak_ic and abs(leak_auc - 0.5) < ev.max_leak_auc_dev,
        f"shuffled-label model: rank IC {leak_ic:+.4f} (limit ±{ev.max_leak_ic}), AUC {leak_auc:.4f} (limit 0.5 ± {ev.max_leak_auc_dev})",
    )
    ic, t = model_m["mean_daily_rank_ic"], model_m["rank_ic_tstat"]
    gate(
        "Signal",
        _finite(ic) >= ev.min_rank_ic and _finite(t) >= ev.min_ic_tstat,
        f"rank IC {ic:+.4f} (need ≥ {ev.min_rank_ic}), t-stat {t:.2f} (need ≥ {ev.min_ic_tstat})",
    )
    s = _finite(base["sharpe"])
    ctrl = np.array([_finite(x) for x in control_sharpes])
    beat = float((s > ctrl).mean()) if len(ctrl) else 0.0
    gate(
        "Skill",
        beat >= ev.control_percentile,
        f"Sharpe {base['sharpe']:.2f} beats {beat:.0%} of {len(ctrl)} permuted-score controls "
        f"(median {np.median(ctrl):.2f}, need ≥ {ev.control_percentile:.0%})",
    )
    gate(
        "Costs",
        _finite(stress["sharpe"]) > ev.min_stress_sharpe,
        f"Sharpe at {ev.stress_cost_mult:g}x costs {stress['sharpe']:.2f} (need > {ev.min_stress_sharpe})",
    )
    frac = float((sweep["sharpe"].fillna(-np.inf) > 0).mean()) if len(sweep) else 0.0
    gate(
        "Robustness",
        frac >= ev.min_sweep_positive_frac,
        f"{frac:.0%} of {len(sweep)} sweep cells have Sharpe > 0 (need ≥ {ev.min_sweep_positive_frac:.0%})",
    )
    gate(
        "Risk",
        base["max_drawdown"] >= -ev.max_drawdown,
        f"max drawdown {base['max_drawdown']:.1%} (limit -{ev.max_drawdown:.0%})",
    )
    gate(
        "Sample",
        base.get("trades", 0) >= ev.min_trades,
        f"{base.get('trades', 0)} trades (need ≥ {ev.min_trades})",
    )
    return gates


def evaluate(
    cfg: Config,
    prices: dict[str, pd.DataFrame],
    preds: pd.DataFrame,
    shuffled_preds: pd.DataFrame,
    out_dir: str | Path,
    n_controls: int | None = None,
    jobs: int = -1,
) -> dict:
    """Runs every check, writes report.md and metrics.json, and returns the results."""
    from joblib import Parallel, delayed

    ev = cfg.evaluation
    n_controls = ev.controls if n_controls is None else n_controls
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = MarketData.from_prices(prices, cfg.benchmark)
    p_up = predictions_to_wide(preds)
    p_shuffled = predictions_to_wide(shuffled_preds)
    stress_costs = replace(cfg.costs, multiplier=cfg.costs.multiplier * ev.stress_cost_mult)
    rng = np.random.default_rng(ev.seed)
    control_scores = [permuted_scores(p_up, cfg.model.retrain_months, rng) for _ in range(n_controls)]
    log.info("Running %d backtests", 3 + n_controls)
    runs = Parallel(n_jobs=jobs)(
        delayed(run_backtest)(md, s, cfg.universe, cfg.strategy, cfg.risk, c)
        for s, c in [(p_up, cfg.costs), (p_up, stress_costs), (p_shuffled, cfg.costs)]
        + [(cs, cfg.costs) for cs in control_scores]
    )
    base_res, stress_res, shuffled_res, controls = runs[0], runs[1], runs[2], runs[3:]
    sweep = sensitivity(md, p_up, cfg, cfg.sweep)

    write_report(base_res, out, "strategy")
    dates = base_res.equity.index
    tradable = [t for t in md.close.columns if t != cfg.benchmark and t in p_up.columns]
    ew = equal_weight_benchmark(md, dates, tradable, cfg.risk.initial_equity)

    model_m = oos_metrics(preds, cfg.model.horizon)
    shuffled_m = oos_metrics(shuffled_preds, cfg.model.horizon)
    base, stress, shuffled = _summary(base_res), _summary(stress_res), _summary(shuffled_res)
    control_perf = [_summary(r) for r in controls]
    control_sharpes = [c["sharpe"] for c in control_perf]
    gates = score_gates(model_m, shuffled_m, base, stress, control_sharpes, sweep, ev)
    verdict = "PASS" if all(g["pass"] for g in gates) else "FAIL"

    results = {
        "verdict": verdict,
        "gates": gates,
        "period": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
        "tickers": tradable,
        "model": model_m,
        "shuffled_model": shuffled_m,
        "strategy": base,
        "strategy_stress": stress,
        "shuffled_strategy": shuffled,
        "controls": {
            "n": n_controls,
            "sharpe": control_sharpes,
            "cagr": [c["cagr"] for c in control_perf],
            "max_drawdown": [c["max_drawdown"] for c in control_perf],
        },
        "spy": performance(base_res.benchmark),
        "equal_weight": performance(ew),
        "sweep": sweep.to_dict(orient="records"),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=_jsonable)
    (out / "report.md").write_text(render_report(results, cfg), encoding="utf-8")
    return results


def _jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return str(v)


def _pct(v) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:+.1%}"


def _num(v, fmt="{:.2f}") -> str:
    return "n/a" if v is None or not np.isfinite(v) else fmt.format(v)


def render_report(r: dict, cfg: Config) -> str:
    """Markdown report: verdict, gates, model quality, strategy versus controls."""
    m, sm = r["model"], r["shuffled_model"]
    c = r["controls"]
    ctrl_sharpe = np.array(c["sharpe"], dtype=float)
    ctrl_cagr = np.array(c["cagr"], dtype=float)
    ctrl_dd = np.array(c["max_drawdown"], dtype=float)
    next_step = (
        "Every gate passed. Next: a paper-trading dry run."
        if r["verdict"] == "PASS"
        else "At least one gate failed. Next: model research, not trading."
    )
    lines = [
        "# Evaluation report",
        "",
        f"**Verdict: {r['verdict']}.** {next_step}",
        "",
        f"Out-of-sample period {r['period']['start']} to {r['period']['end']}, "
        f"{len(r['tickers'])} tradable tickers, benchmark {cfg.benchmark}. "
        "Every number below is out of sample and net of modeled costs unless noted.",
        "",
        "## Gates",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
        *[f"| {g['gate']} | {'PASS' if g['pass'] else 'FAIL'} | {g['detail']} |" for g in r["gates"]],
        "",
        "## Model quality",
        "",
        "| Metric | Model | Shuffled labels |",
        "|---|---|---|",
        f"| AUC | {_num(m['auc'], '{:.4f}')} | {_num(sm['auc'], '{:.4f}')} |",
        f"| Log loss (base-rate guess) | {_num(m['log_loss'], '{:.4f}')} ({_num(m['log_loss_base_rate'], '{:.4f}')}) "
        f"| {_num(sm['log_loss'], '{:.4f}')} ({_num(sm['log_loss_base_rate'], '{:.4f}')}) |",
        f"| Mean daily rank IC | {_num(m['mean_daily_rank_ic'], '{:+.4f}')} | {_num(sm['mean_daily_rank_ic'], '{:+.4f}')} |",
        f"| Rank IC t-stat | {_num(m['rank_ic_tstat'])} | {_num(sm['rank_ic_tstat'])} |",
        f"| Top minus bottom quintile, 5-day return | {_num(m['top_bottom_quintile_bps'], '{:+.1f}')} bps "
        f"| {_num(sm['top_bottom_quintile_bps'], '{:+.1f}')} bps |",
        f"| Price forecast R² vs no-change forecast | {_num(m['reg_r2_vs_no_change'], '{:+.4f}')} "
        f"| {_num(sm['reg_r2_vs_no_change'], '{:+.4f}')} |",
        f"| Price forecast direction hit rate | {_num(m['reg_sign_hit_rate'], '{:.1%}')} | {_num(sm['reg_sign_hit_rate'], '{:.1%}')} |",
        "",
        "A rank IC of 0 means no ability to tell better stocks from worse ones. "
        "An R² at or below 0 means the 5-day price forecast is no better than assuming the price stays put.",
        "",
        "## Strategy versus controls",
        "",
        "| Run | CAGR | Sharpe | Max drawdown | Trades | Avg exposure |",
        "|---|---|---|---|---|---|",
    ]

    def row(name, p):
        return (
            f"| {name} | {_pct(p['cagr'])} | {_num(p['sharpe'])} | {_pct(p['max_drawdown'])} "
            f"| {p.get('trades', '-')} | {_num(p.get('avg_exposure', float('nan')), '{:.0%}')} |"
        )

    lines += [
        row("Strategy", r["strategy"]),
        row(f"Strategy at {cfg.evaluation.stress_cost_mult:g}x costs", r["strategy_stress"]),
        row("Shuffled-label model", r["shuffled_strategy"]),
        f"| Permuted-score controls, median of {c['n']} | {_pct(np.median(ctrl_cagr))} | {_num(np.median(ctrl_sharpe))} "
        f"| {_pct(np.median(ctrl_dd))} | - | - |",
        f"| Permuted-score controls, 95th percentile | {_pct(np.quantile(ctrl_cagr, 0.95))} "
        f"| {_num(np.quantile(ctrl_sharpe, 0.95))} | - | - | - |",
        row(f"{cfg.benchmark} buy and hold (no costs)", {**r["spy"], "trades": "-"}),
        row("Equal-weight universe (no costs)", {**r["equal_weight"], "trades": "-"}),
        "",
        "## Parameter sensitivity",
        "",
        "| " + " | ".join(cfg.sweep) + " | CAGR | Sharpe | Max drawdown | Trades |",
        "|" + "---|" * (len(cfg.sweep) + 4),
        *[
            "| " + " | ".join(f"{s[k]:g}" for k in cfg.sweep)
            + f" | {_pct(s['cagr'])} | {_num(s['sharpe'])} | {_pct(s['max_drawdown'])} | {s['trades']} |"
            for s in r["sweep"]
        ],
        "",
        "## Trading detail",
        "",
        f"- Exit reasons: {r['strategy'].get('exit_reasons', {})}",
        f"- Total modeled costs: ${r['strategy'].get('total_costs', 0):,.0f}; "
        f"hit rate {_num(r['strategy'].get('hit_rate', float('nan')), '{:.1%}')}; "
        f"average hold {_num(r['strategy'].get('avg_days_held', float('nan')), '{:.1f}')} days",
        f"- Days with new entries halted by the drawdown guard: {r['strategy']['halted_days']}",
        "",
        "## Caveats",
        "",
        "- The universe is today's large caps, chosen with hindsight. That flatters every long-only "
        "result here, which is why the permuted-score controls, not SPY, are the bar for skill.",
        "- Sharpe uses a zero risk-free rate, and idle cash earns nothing.",
        "- Prices are Yahoo's split- and dividend-adjusted closes. Commissions use split-corrected share counts.",
    ]
    return "\n".join(lines) + "\n"
