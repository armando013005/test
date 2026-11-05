# Plan: Daily US stock prediction and trading system

## Context

The repo `armando013005/test` is empty (no commits). You want a system that
forecasts stock prices and trades on those forecasts. It must follow strict
rules for entries, stop-losses, position sizing, historical data, and
transaction costs.

Reality check: no model predicts prices accurately. The goal is a small,
measurable edge in forecasting the direction of returns over the next 5 days.
Risk rules and an honest, cost-aware walk-forward backtest turn that edge into
a strategy. Paper trading comes last, and only if the backtest earns it.

Decisions made: US equities/ETFs, daily bars, LightGBM, backtest plus Alpaca
paper trading. Language is Python 3.11.

## Layout

```
pyproject.toml            # deps: pandas, numpy, yfinance, lightgbm, scikit-learn,
                          # pyarrow, pyyaml, matplotlib, alpaca-py, pytest
config/default.yaml       # every tunable rule lives here
src/stockpred/
  data.py        # download + cache OHLCV (yfinance, auto-adjusted) to parquet
  features.py    # leakage-safe features
  labels.py      # forward-return / direction labels
  model.py       # LightGBM wrapper, purged walk-forward training
  signals.py     # entry/exit rules
  risk.py        # position sizing, stop-loss, portfolio limits
  costs.py       # commission + spread + slippage model
  backtest.py    # event-driven daily simulator
  metrics.py     # Sharpe, Sortino, max DD, CAGR, hit rate, turnover
  report.py      # equity curve, drawdown, trade log CSV
  broker.py      # Alpaca paper client (orders, positions, bracket stops)
  cli.py         # `stockpred fetch|train|backtest|signal|trade`
tests/                    # unit tests per module + leakage tests
```

## Components

**Historical data (`data.py`).** Pull 15+ years of split- and
dividend-adjusted daily OHLCV for a liquid universe: SPY, QQQ, and about 50
large caps. Cache it as parquet. Drop tickers with gaps; screen for
liquidity (median dollar volume > $20M). Flag survivorship bias as a known
limitation, because yfinance has no delisted tickers.

**Features (`features.py`).** Each feature at row t uses only data up to the
close of day t:
- Returns over 1, 5, 20, and 60 days
- Gaps between the price and its 10, 50, and 200-day moving averages
- RSI(14), MACD histogram
- ATR(14)/close, 20-day realized volatility
- Volume z-score
- Market context: SPY's 20-day return and the volatility regime

**Labels (`labels.py`).** The target is 1 when the forward 5-day return
beats the cost-adjusted threshold, otherwise 0. The model also outputs a
probability, which a separate regressor converts into an expected return
for the "current/future value" estimate.

**Model (`model.py`).** LightGBM classifier, shallow trees, strong
regularization. Training is a purged, embargoed walk-forward:
- Train on a rolling 5-year window and retrain every 3 months.
- Purge 5 days before each test window to stop label overlap.
- Calibrate the probabilities.
- Score out of sample only: AUC, log loss, hit rate.

**Entry rules (`signals.py`).** All five conditions must hold:
1. Calibrated P(up) ≥ 0.58 (configurable).
2. The close is above the 200-day moving average (trend filter).
3. SPY is above its 200-day moving average (regime filter; no new longs when
   it is not).
4. No earnings in the next 3 days (optional and best effort).
5. Liquidity screen passed.

Orders fill at the next day's open. Signals never fill at the same close
that generated them.

**Exit rules.** A position closes on the first of these:
- Initial stop: entry − 2×ATR.
- Trailing stop: highest close − 3×ATR.
- Time stop: 10 trading days.
- Model flip: P(up) < 0.45.

The backtest checks stops against intraday lows. A gap below the stop fills
at the open, not at the stop price.

**Position sizing (`risk.py`).** Fixed-fractional risk:

```
shares = (equity × 0.5%) / (entry − stop)
```

The system then applies these caps:
- 10% of equity per position
- 8 open positions at most
- 25% gross exposure per sector
- Halt new entries at a 15% portfolio drawdown (kill switch)
- Optional volatility targeting

**Transaction costs (`costs.py`).** Each trade pays:
- $0.005 per share commission, $1 minimum
- Half the spread, about 2 bps for large caps
- Slippage of 0.1×ATR% × √(participation)

The model charges these on entry and exit. The backtest also runs at 2× costs
as a stress test.

**Backtest (`backtest.py`).** Daily, event-driven, long-only to start. Inputs
are out-of-sample predictions only. Outputs:
- Trade log
- Equity curve
- Metrics, compared against SPY buy-and-hold
- Parameter sensitivity (threshold and ATR multiples), to show robustness
  rather than a single tuned point

**Paper trading (`broker.py`, `cli.py trade`).** A daily job runs after the
close:
1. Fetch data.
2. Score.
3. Apply the rules.
4. Submit bracket orders (entry plus stop) to Alpaca paper for the next open.
5. Sync positions and log everything.

API keys come from environment variables. The client points at the paper
endpoint only, and a hard guard refuses the live URL.

## Order of work

1. Scaffold, config, data cache, and tests.
2. Features, labels, and the leakage tests.
3. Walk-forward model, plus an out-of-sample evaluation report.
4. Costs, risk, signals, and the backtester.
5. Metrics, report, and the sensitivity sweep.
6. Alpaca paper broker and the `trade` CLI.
7. A README in Google developer style that covers setup, usage, and
   limitations.

Commit after each step on `ai/eager-fermi-h956f5` and push.

## Verification

- `pytest`:
  - A leakage test shuffles future rows and asserts that features at t don't
    change.
  - Stop and gap fill logic.
  - Sizing math and caps.
  - Cost arithmetic.
  - A tiny synthetic-price backtest with known results.
- `stockpred fetch && stockpred train && stockpred backtest`: produces an
  out-of-sample report with net-of-cost metrics versus SPY.
- Sanity check: running with random labels must give a Sharpe of about 0 or
  worse after costs. If it doesn't, something is leaking.
- `stockpred trade --dry-run`: prints the orders it would place. A real paper
  run needs your Alpaca paper keys.

## Risks

- Expect a modest edge or none. The backtest may show the strategy fails to
  beat SPY after costs. That counts as a valid result, not a bug.
- Survivorship bias in the yfinance universe inflates results.
- Network access: yfinance (query1.finance.yahoo.com) and Alpaca hosts may
  need allowlisting in this cloud environment.
