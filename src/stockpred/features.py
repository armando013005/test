"""Builds features. The value at row t uses only data up to the close of day t."""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "dist_sma10",
    "dist_sma50",
    "dist_sma200",
    "rsi14",
    "macd_hist",
    "atr_pct",
    "vol20",
    "volume_z",
    "mkt_ret20",
    "mkt_vol20",
    "mkt_dist_sma200",
]


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's average true range."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's relative strength index, scaled to 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def market_features(bench: pd.DataFrame) -> pd.DataFrame:
    """Benchmark context shared by every ticker."""
    close = bench["close"]
    sma200 = close.rolling(200).mean()
    out = pd.DataFrame(index=bench.index)
    out["mkt_ret20"] = close.pct_change(20)
    out["mkt_vol20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    out["mkt_dist_sma200"] = close / sma200 - 1
    out["mkt_above_sma200"] = close > sma200
    return out


def compute_features(df: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns model features plus the raw columns the trading rules need."""
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    for n in (1, 5, 20, 60):
        out[f"ret_{n}"] = close.pct_change(n)
    for n in (10, 50, 200):
        out[f"sma{n}"] = close.rolling(n).mean()
        out[f"dist_sma{n}"] = close / out[f"sma{n}"] - 1
    out["rsi14"] = rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    out["atr14"] = atr(df)
    out["atr_pct"] = out["atr14"] / close
    out["vol20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    log_vol = np.log1p(df["volume"])
    out["volume_z"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()
    out["dollar_vol_med60"] = (close * df["volume"]).rolling(60).median()
    out["adv20"] = df["volume"].rolling(20).mean()
    out["close"] = close

    if market is None:
        market = market_features(df)
    out = out.join(market.reindex(df.index))
    out["mkt_above_sma200"] = out["mkt_above_sma200"].astype("boolean").fillna(False).astype(bool)
    return out.replace([np.inf, -np.inf], np.nan)
