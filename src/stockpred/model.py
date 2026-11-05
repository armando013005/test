"""LightGBM direction model with purged, embargoed walk-forward training."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .config import ModelConfig
from .features import MODEL_FEATURES, compute_features, market_features
from .labels import direction_label, forward_return

log = logging.getLogger(__name__)


def build_panel(prices: dict[str, pd.DataFrame], benchmark: str, cfg: ModelConfig) -> pd.DataFrame:
    """Stacks per-ticker features and labels into one (date, ticker) frame.

    Label modes:
      cross_sectional: the stock's forward return beats that day's median
        across the universe by `label_threshold`. The base rate stays near
        constant over time, so probabilities from different folds are
        comparable and the model learns to rank stocks, not to time the market.
      excess: the forward return beats the benchmark's.
      absolute: the raw forward return beats the threshold.

    In the relative modes the benchmark is left out. Its excess return over
    itself is always 0, so keeping it would let the model score well just by
    recognizing the benchmark.
    """
    if cfg.label_mode not in ("cross_sectional", "excess", "absolute"):
        raise ValueError(f"Unknown label_mode {cfg.label_mode!r}")
    market = market_features(prices[benchmark])
    bench_fwd = forward_return(prices[benchmark], cfg.horizon)
    frames = []
    for ticker, df in prices.items():
        if cfg.label_mode != "absolute" and ticker == benchmark:
            continue
        feats = compute_features(df, market)
        feats["fwd_ret"] = forward_return(df, cfg.horizon)
        feats["target"] = feats["fwd_ret"]
        if cfg.label_mode == "excess":
            feats["target"] -= bench_fwd.reindex(df.index)
        feats["ticker"] = ticker
        frames.append(feats)
    panel = pd.concat(frames).reset_index().set_index(["date", "ticker"]).sort_index()
    if cfg.label_mode == "cross_sectional":
        panel["target"] -= panel.groupby(level="date")["target"].transform("median")
    panel["label"] = direction_label(panel["target"], cfg.label_threshold)
    return panel.drop(columns="target")


@dataclass
class FittedModel:
    """A classifier, its Platt calibrator, and a return regressor."""

    clf: LGBMClassifier
    calibrator: LogisticRegression | None
    reg: LGBMRegressor

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[MODEL_FEATURES]
        raw = self.clf.predict_proba(X)[:, 1]
        p_up = self.calibrator.predict_proba(_logit(raw))[:, 1] if self.calibrator is not None else raw
        return pd.DataFrame({"p_up": p_up, "exp_ret": self.reg.predict(X)}, index=X.index)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _trainable(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=[*MODEL_FEATURES, "label", "fwd_ret"])


def fit_model(
    train: pd.DataFrame, cfg: ModelConfig, shuffle_labels: bool = False, seed: int = 0
) -> FittedModel:
    """Fits on the head of the window and calibrates on the purged tail."""
    train = _trainable(train)
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        train = train.copy()
        perm = rng.permutation(len(train))
        train["label"] = train["label"].to_numpy()[perm]
        train["fwd_ret"] = train["fwd_ret"].to_numpy()[perm]

    dates = train.index.get_level_values("date").unique().sort_values()
    cal_start_pos = int(len(dates) * (1 - cfg.calib_frac))
    purge = cfg.horizon + cfg.embargo_days
    fit_end = dates[max(cal_start_pos - purge, 1)]
    cal_start = dates[min(cal_start_pos, len(dates) - 1)]
    date_idx = train.index.get_level_values("date")
    fit_part = train[date_idx < fit_end]
    cal_part = train[date_idx >= cal_start]

    params = {"random_state": seed, **cfg.lgbm}
    clf = LGBMClassifier(**params).fit(fit_part[MODEL_FEATURES], fit_part["label"])
    calibrator = None
    if cal_part["label"].nunique() == 2 and len(cal_part) >= 500:
        raw = clf.predict_proba(cal_part[MODEL_FEATURES])[:, 1]
        calibrator = LogisticRegression(C=1.0).fit(_logit(raw), cal_part["label"])

    lo, hi = fit_part["fwd_ret"].quantile([0.01, 0.99])
    reg = LGBMRegressor(**params).fit(fit_part[MODEL_FEATURES], fit_part["fwd_ret"].clip(lo, hi))
    return FittedModel(clf, calibrator, reg)


def walk_forward(
    panel: pd.DataFrame, cfg: ModelConfig, shuffle_labels: bool = False, seed: int = 0
) -> pd.DataFrame:
    """Produces out-of-sample predictions for every date after the first training window.

    Each test block is scored by a model trained only on data that ends
    `horizon + embargo_days` trading days before the block starts, so no
    training label overlaps the test period.
    """
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    date_idx = panel.index.get_level_values("date")
    first_test = all_dates[0] + pd.DateOffset(years=cfg.train_years)
    boundaries = pd.date_range(first_test, all_dates[-1], freq=pd.DateOffset(months=cfg.retrain_months))
    boundaries = [*boundaries, all_dates[-1] + pd.Timedelta(days=1)]
    purge = cfg.horizon + cfg.embargo_days

    preds = []
    for i, (test_start, test_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        start_pos = all_dates.searchsorted(test_start)
        if start_pos - purge <= 0:
            continue
        train_end = all_dates[start_pos - purge]
        train_start = test_start - pd.DateOffset(years=cfg.train_years)
        train = panel[(date_idx >= train_start) & (date_idx < train_end)]
        test = panel[(date_idx >= test_start) & (date_idx < test_end)]
        test = test.dropna(subset=MODEL_FEATURES)
        if test.empty or len(_trainable(train)) < cfg.min_train_rows:
            continue
        model = fit_model(train, cfg, shuffle_labels=shuffle_labels, seed=seed + i)
        out = model.predict(test)
        out["fwd_ret"] = test["fwd_ret"]
        out["label"] = test["label"]
        preds.append(out)
        log.info("Fold %s..%s: train rows %d, test rows %d", test_start.date(), test_end.date(), len(train), len(test))
    if not preds:
        raise ValueError("No walk-forward folds. Need more history than train_years.")
    return pd.concat(preds).sort_index()


def fit_final(panel: pd.DataFrame, cfg: ModelConfig, seed: int = 0) -> FittedModel:
    """Fits on the most recent training window for live scoring."""
    labeled = _trainable(panel)
    last = labeled.index.get_level_values("date").max()
    start = last - pd.DateOffset(years=cfg.train_years)
    return fit_model(labeled[labeled.index.get_level_values("date") >= start], cfg, seed=seed)


def oos_metrics(preds: pd.DataFrame) -> dict[str, float]:
    """Out-of-sample classification and ranking quality."""
    scored = preds.dropna(subset=["label", "fwd_ret"])
    y, p = scored["label"].to_numpy(), scored["p_up"].clip(1e-6, 1 - 1e-6).to_numpy()
    base_rate = float(y.mean())
    ic_by_day = scored.groupby(level="date").apply(
        lambda g: spearmanr(g["p_up"], g["fwd_ret"]).statistic if len(g) > 5 else np.nan
    )
    return {
        "rows": int(len(scored)),
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "log_loss_base_rate": float(log_loss(y, np.full_like(p, base_rate))),
        "hit_rate": float(((p > 0.5) == (y == 1)).mean()),
        "base_rate": base_rate,
        "mean_daily_rank_ic": float(np.nanmean(ic_by_day)),
    }
