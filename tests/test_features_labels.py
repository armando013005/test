import numpy as np
import pandas as pd

from stockpred.data import clean
from stockpred.features import MODEL_FEATURES, compute_features
from stockpred.labels import direction_label, forward_return


def test_features_do_not_use_future_data(small_prices):
    df = small_prices["S00"]
    cut = 1500
    base = compute_features(df)
    tampered = df.copy()
    rng = np.random.default_rng(0)
    tampered.iloc[cut + 1 :] *= rng.uniform(0.5, 1.5, size=tampered.iloc[cut + 1 :].shape)
    after = compute_features(clean(tampered))
    pd.testing.assert_frame_equal(base.iloc[: cut + 1], after.iloc[: cut + 1])


def test_all_model_features_present(small_prices):
    feats = compute_features(small_prices["S01"])
    assert set(MODEL_FEATURES) <= set(feats.columns)
    assert feats[MODEL_FEATURES].iloc[250:].notna().all().all()


def test_forward_return_enters_next_open_exits_close_h_days_later(small_prices):
    df = small_prices["S02"]
    fwd = forward_return(df, horizon=5)
    t = 100
    expected = df["close"].iloc[t + 5] / df["open"].iloc[t + 1] - 1
    assert np.isclose(fwd.iloc[t], expected)
    assert fwd.iloc[-5:].isna().all()


def test_direction_label_keeps_unknown_as_nan():
    fwd = pd.Series([0.01, -0.01, 0.001, np.nan])
    lab = direction_label(fwd, threshold=0.002)
    assert lab.iloc[:3].tolist() == [1.0, 0.0, 0.0]
    assert np.isnan(lab.iloc[3])
