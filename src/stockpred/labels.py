"""Builds forward-looking targets. These are the only columns allowed to see the future."""

from __future__ import annotations

import pandas as pd


def forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Return from the next day's open to the close `horizon` days ahead.

    This matches execution: a signal at close t fills at open t+1.
    """
    entry = df["open"].shift(-1)
    exit_ = df["close"].shift(-horizon)
    return exit_ / entry - 1


def direction_label(fwd_ret: pd.Series, threshold: float) -> pd.Series:
    """1.0 when the forward return beats the threshold, 0.0 otherwise, NaN when unknown."""
    return (fwd_ret > threshold).astype(float).where(fwd_ret.notna())
