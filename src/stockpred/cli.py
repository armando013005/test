"""Command-line entry point: stockpred fetch | train | backtest | signal | trade."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd

from . import data
from .backtest import MarketData, predictions_to_wide, run_backtest, sensitivity
from .config import Config, load_config
from .model import build_panel, fit_final, oos_metrics, walk_forward
from .report import print_metrics, write_report

log = logging.getLogger("stockpred")


def _artifacts(cfg: Config) -> Path:
    path = Path(cfg.paths.get("artifacts", "artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_prices(cfg: Config) -> dict[str, pd.DataFrame]:
    d = cfg.data
    return data.load_universe(
        list(cfg.universe), d["cache_dir"], cfg.benchmark, d.get("min_history_days", 600), d.get("max_missing_frac", 0.02)
    )


def cmd_fetch(cfg: Config, args) -> None:
    tickers = list(dict.fromkeys([cfg.benchmark, *cfg.universe]))
    if args.synthetic:
        prices = data.synthetic_prices(tickers, n_days=args.days, seed=args.seed, autocorr=args.autocorr)
        for ticker, df in prices.items():
            data.save(df, cfg.data["cache_dir"], ticker)
        print(f"Wrote synthetic data for {len(prices)} tickers to {cfg.data['cache_dir']}")
        return
    failed = data.fetch_universe(tickers, cfg.data["start"], cfg.data["cache_dir"])
    print(f"Fetched {len(tickers) - len(failed)}/{len(tickers)} tickers. Failed: {failed or 'none'}")


def cmd_train(cfg: Config, args) -> None:
    prices = _load_prices(cfg)
    panel = build_panel(prices, cfg.benchmark, cfg.model)
    preds = walk_forward(panel, cfg.model, shuffle_labels=args.shuffle_labels, seed=args.seed)
    out = _artifacts(cfg)
    suffix = "_shuffled" if args.shuffle_labels else ""
    preds.to_parquet(out / f"predictions{suffix}.parquet")
    metrics = oos_metrics(preds)
    with open(out / f"oos_metrics{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Out-of-sample model quality:")
    for k, v in metrics.items():
        print(f"  {k:<20}{v:.4f}" if isinstance(v, float) else f"  {k:<20}{v}")
    if not args.shuffle_labels:
        joblib.dump(fit_final(panel, cfg.model, seed=args.seed), out / "model.joblib")
        print(f"Saved final model to {out / 'model.joblib'}")


def cmd_backtest(cfg: Config, args) -> None:
    out = _artifacts(cfg)
    preds = pd.read_parquet(args.predictions or out / "predictions.parquet")
    prices = _load_prices(cfg)
    md = MarketData.from_prices(prices, cfg.benchmark)
    p_up = predictions_to_wide(preds)
    costs = replace(cfg.costs, multiplier=args.cost_mult) if args.cost_mult else cfg.costs
    result = run_backtest(md, p_up, cfg.universe, cfg.strategy, cfg.risk, costs)
    label = args.label or (f"backtest_cost{args.cost_mult:g}x" if args.cost_mult else "backtest")
    metrics = write_report(result, out, label)
    print_metrics(metrics)
    print(f"Report written to {out}/{label}_*")
    if args.sweep:
        table = sensitivity(md, p_up, replace(cfg, costs=costs), cfg.sweep)
        table.to_csv(out / f"{label}_sensitivity.csv", index=False)
        print("\nParameter sensitivity:")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _snapshot(cfg: Config) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    prices = _load_prices(cfg)
    model = joblib.load(_artifacts(cfg) / "model.joblib")
    panel = build_panel(prices, cfg.benchmark, cfg.model)
    calendar = prices[cfg.benchmark].index
    latest = panel.xs(calendar[-1], level="date").dropna(subset=["atr14"])
    scores = model.predict(latest.dropna(subset=model.clf.feature_name_))
    snap = latest.join(scores, how="inner")
    snap["pred_close"] = snap["close"] * (1 + snap["exp_ret"])
    return snap, calendar


def cmd_signal(cfg: Config, args) -> None:
    from .signals import entry_ok

    snap, calendar = _snapshot(cfg)
    snap["entry"] = [entry_ok(row, cfg.strategy) for _, row in snap.iterrows()]
    snap["stop"] = snap["close"] - cfg.strategy.stop_atr * snap["atr14"]
    cols = ["close", "p_up", "exp_ret", "pred_close", "stop", "entry"]
    print(f"Signals as of close {calendar[-1].date()} (forecast horizon {cfg.model.horizon} days):")
    print(snap.sort_values("p_up", ascending=False)[cols].head(args.top).to_string(float_format=lambda v: f"{v:.3f}"))


def cmd_trade(cfg: Config, args) -> None:
    from .live import (
        has_upcoming_earnings,
        holdings_from_json,
        holdings_to_json,
        load_state,
        plan_orders,
        save_state,
    )

    if not args.no_fetch and not args.dry_run:
        cmd_fetch(cfg, argparse.Namespace(synthetic=False))
    snap, calendar = _snapshot(cfg)
    state_path = cfg.paths.get("state", "state/live.json")
    state = load_state(state_path, cfg.risk.initial_equity)
    holdings = holdings_from_json(state["holdings"])

    broker = None
    if args.dry_run:
        equity = state.get("equity", cfg.risk.initial_equity)
        cash = state.get("cash", equity)
    else:
        from .broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()
        equity, cash = broker.account()
        live_positions = broker.positions()
        holdings = {t: h for t, h in holdings.items() if t in live_positions}  # Stopped-out names drop.
        for ticker, qty in live_positions.items():
            if ticker in holdings:
                holdings[ticker].shares = qty
            elif ticker in snap.index:
                close, atr = snap.loc[ticker, "close"], snap.loc[ticker, "atr14"]
                from .live import Holding

                holdings[ticker] = Holding(qty, str(calendar[-2].date()), close - cfg.strategy.stop_atr * atr, close)

    candidates = snap[snap["p_up"] >= cfg.strategy.entry_prob].index
    blackout = {t for t in candidates if has_upcoming_earnings(t, cfg.strategy.earnings_blackout_days)}
    orders, new_holdings, peak = plan_orders(
        snap, calendar, holdings, equity, cash, state.get("peak_equity", equity), cfg, blackout
    )

    print(f"As of close {calendar[-1].date()}: equity {equity:,.0f}, cash {cash:,.0f}, {len(holdings)} held")
    if blackout:
        print(f"Earnings blackout: {sorted(blackout)}")
    if not orders:
        print("No orders.")
    for o in orders:
        stop = f" stop {o.stop:.2f}" if o.stop else ""
        print(f"  {o.action:<12}{o.ticker:<7}{o.shares:>6} @ ~{o.ref_price:.2f}{stop}  ({o.reason})")

    if args.dry_run:
        print("Dry run: no orders sent, state unchanged.")
        return
    for o in orders:
        broker.submit(o)
    save_state(state_path, {"peak_equity": peak, "equity": equity, "cash": cash, "holdings": holdings_to_json(new_holdings)})
    print(f"Sent {len(orders)} orders to Alpaca paper. State saved to {state_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stockpred", description=__doc__)
    parser.add_argument("--config", default=None, help="YAML file that overrides config/default.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="Download daily bars to the cache")
    p.add_argument("--synthetic", action="store_true", help="Write fake data instead (offline demo)")
    p.add_argument("--days", type=int, default=3500)
    p.add_argument("--autocorr", type=float, default=0.0, help="Plant an AR(1) edge in synthetic data")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("train", help="Walk-forward train and save out-of-sample predictions")
    p.add_argument("--shuffle-labels", action="store_true", help="Leakage check: train on shuffled labels")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("backtest", help="Simulate the strategy on out-of-sample predictions")
    p.add_argument("--predictions", default=None)
    p.add_argument("--cost-mult", type=float, default=None, help="Scale all costs, for example 2.0")
    p.add_argument("--sweep", action="store_true", help="Run the parameter sensitivity grid")
    p.add_argument("--label", default=None)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("signal", help="Score the latest close and show entry candidates")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_signal)

    p = sub.add_parser("trade", help="Plan and send orders to Alpaca paper")
    p.add_argument("--dry-run", action="store_true", help="Print orders without sending them")
    p.add_argument("--no-fetch", action="store_true", help="Skip refreshing data first")
    p.set_defaults(func=cmd_trade)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.func(load_config(args.config), args)


if __name__ == "__main__":
    main()
