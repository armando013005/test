from dataclasses import replace

import pytest

from stockpred.data import synthetic_prices
from stockpred.model import build_panel, oos_metrics, walk_forward


@pytest.fixture(scope="module")
def model_cfg():
    from stockpred.config import load_config

    m = load_config().model
    return replace(m, train_years=3, retrain_months=12, min_train_rows=2000, lgbm={**m.lgbm, "n_estimators": 80})


def _panel(autocorr, model_cfg):
    tickers = ["SPY", *[f"S{i:02d}" for i in range(10)]]
    prices = synthetic_prices(tickers, n_days=1600, seed=7, autocorr=autocorr)
    return build_panel(prices, "SPY", model_cfg)


def test_walk_forward_is_out_of_sample(model_cfg):
    panel = _panel(0.0, model_cfg)
    preds = walk_forward(panel, model_cfg)
    first_date = panel.index.get_level_values("date").min()
    assert preds.index.get_level_values("date").min() >= first_date + pd_years(model_cfg.train_years)
    assert preds["p_up"].between(0, 1).all()


def test_model_finds_planted_edge_and_not_noise(model_cfg):
    # Out-of-sample AUC on pure noise wanders about +/-0.02 at this sample size,
    # so the planted edge must be strong enough to clear that band.
    edge = oos_metrics(walk_forward(_panel(-0.5, model_cfg), model_cfg))
    noise = oos_metrics(walk_forward(_panel(0.0, model_cfg), model_cfg))
    assert edge["auc"] > 0.54
    assert edge["mean_daily_rank_ic"] > 0.05
    assert abs(noise["auc"] - 0.5) < 0.04


def test_shuffled_labels_give_no_edge(model_cfg):
    panel = _panel(-0.5, model_cfg)
    shuffled = oos_metrics(walk_forward(panel, model_cfg, shuffle_labels=True))
    assert abs(shuffled["auc"] - 0.5) < 0.04


def pd_years(n):
    import pandas as pd

    return pd.DateOffset(years=n)
