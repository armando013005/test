"""Downloads, cleans, caches, and loads daily OHLCV bars."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drops bad rows and enforces OHLC consistency."""
    df = df[COLUMNS].astype(float).copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return df


def fetch_yfinance(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Downloads split- and dividend-adjusted daily bars from Yahoo Finance."""
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True, actions=False, progress=False, threads=False
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    return clean(raw)


def cache_path(cache_dir: str | Path, ticker: str) -> Path:
    return Path(cache_dir) / f"{ticker.upper()}.parquet"


def save(df: pd.DataFrame, cache_dir: str | Path, ticker: str) -> None:
    path = cache_path(cache_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def fetch_universe(
    tickers: list[str], start: str, cache_dir: str | Path, end: str | None = None
) -> list[str]:
    """Downloads every ticker to the cache. Returns the tickers that failed."""
    failed = []
    for ticker in tickers:
        try:
            save(fetch_yfinance(ticker, start, end), cache_dir, ticker)
            log.info("Fetched %s", ticker)
        except Exception as exc:  # Network and vendor errors vary widely.
            log.warning("Failed to fetch %s: %s", ticker, exc)
            failed.append(ticker)
    return failed


def load_universe(
    tickers: list[str],
    cache_dir: str | Path,
    benchmark: str,
    min_history_days: int = 600,
    max_missing_frac: float = 0.02,
) -> dict[str, pd.DataFrame]:
    """Loads cached bars and drops tickers that are short or have calendar gaps.

    The benchmark defines the trading calendar, so it must be in the cache.
    """
    bench_path = cache_path(cache_dir, benchmark)
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark {benchmark} not cached. Run `stockpred fetch` first.")
    calendar = pd.read_parquet(bench_path).index

    prices: dict[str, pd.DataFrame] = {}
    for ticker in dict.fromkeys([benchmark, *tickers]):
        path = cache_path(cache_dir, ticker)
        if not path.exists():
            log.warning("Skipping %s: not cached", ticker)
            continue
        df = clean(pd.read_parquet(path))
        df = df[df.index.isin(calendar)]
        if len(df) < min_history_days:
            log.warning("Skipping %s: only %d rows", ticker, len(df))
            continue
        expected = calendar[(calendar >= df.index[0]) & (calendar <= df.index[-1])]
        missing = 1.0 - len(df) / max(len(expected), 1)
        if missing > max_missing_frac:
            log.warning("Skipping %s: %.1f%% of days missing", ticker, 100 * missing)
            continue
        prices[ticker] = df
    return prices


def synthetic_prices(
    tickers: list[str],
    start: str = "2008-01-01",
    n_days: int = 3000,
    seed: int = 0,
    autocorr: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Generates fake bars for offline tests and demos.

    Returns follow a one-factor model. A nonzero `autocorr` plants a weak AR(1)
    edge in each stock's idiosyncratic return so tests can check that the model
    finds a real signal. With `autocorr=0.0` the series are unpredictable.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days, name="date")
    mkt = rng.normal(0.0003, 0.011, n_days)
    out: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        is_market = i == 0
        beta = 1.0 if is_market else rng.uniform(0.6, 1.4)
        idio_vol = 0.0 if is_market else rng.uniform(0.010, 0.020)
        eps = rng.normal(0.0, 1.0, n_days) * idio_vol
        idio = np.zeros(n_days)
        for t in range(n_days):
            idio[t] = (autocorr * idio[t - 1] if t else 0.0) + eps[t]
        ret = beta * mkt + idio
        close = 50.0 * np.exp(np.cumsum(ret))
        prev = np.concatenate([[close[0]], close[:-1]])
        gap = rng.normal(0.0, 0.002, n_days)
        open_ = prev * np.exp(gap)
        wick = np.abs(rng.normal(0.0, 0.006, (2, n_days)))
        high = np.maximum(open_, close) * (1 + wick[0])
        low = np.minimum(open_, close) * (1 - wick[1])
        volume = rng.lognormal(np.log(5e6), 0.3, n_days)
        out[ticker] = clean(
            pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=dates,
            )
        )
    return out
