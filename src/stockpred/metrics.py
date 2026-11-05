"""Performance statistics for an equity curve and its trade log."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough loss, as a negative fraction."""
    return float((equity / equity.cummax() - 1).min())


def performance(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, float]:
    rets = equity.pct_change().dropna()
    years = max(len(rets) / TRADING_DAYS, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    # Downside deviation: RMS of returns below zero, over all days.
    downside = float(np.sqrt((rets.clip(upper=0) ** 2).mean()) * np.sqrt(TRADING_DAYS))
    mean_ann = float(rets.mean() * TRADING_DAYS)
    cagr = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    mdd = max_drawdown(equity)
    out = {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": mean_ann / vol if vol > 0 else np.nan,
        "sortino": mean_ann / downside if downside and downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "trades": 0,
    }
    if trades is not None and not trades.empty:
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] <= 0]
        notional = (trades["entry_price"] * trades["shares"]).sum() * 2
        out.update(
            {
                "trades": int(len(trades)),
                "hit_rate": float(len(wins) / len(trades)),
                "avg_win": float(wins["return"].mean()) if len(wins) else 0.0,
                "avg_loss": float(losses["return"].mean()) if len(losses) else 0.0,
                "profit_factor": float(wins["net_pnl"].sum() / -losses["net_pnl"].sum())
                if losses["net_pnl"].sum() < 0
                else np.nan,
                "avg_days_held": float(trades["days_held"].mean()),
                "total_costs": float(trades["costs"].sum()),
                "annual_turnover": float(notional / equity.mean() / years),
                "exit_reasons": trades["exit_reason"].value_counts().to_dict(),
            }
        )
    return out
