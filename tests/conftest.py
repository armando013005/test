import pandas as pd
import pytest

from stockpred.config import load_config
from stockpred.data import synthetic_prices


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_prices():
    tickers = ["SPY", *[f"S{i:02d}" for i in range(12)]]
    return synthetic_prices(tickers, n_days=2200, seed=1, autocorr=0.0)


def bars(closes, opens=None, highs=None, lows=None, volume=5e6, start="2020-01-01"):
    """Builds a small hand-made OHLCV frame."""
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    opens = opens or closes
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs or [max(o, c) for o, c in zip(opens, closes)],
            "low": lows or [min(o, c) for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [volume] * len(closes),
        },
        index=idx,
    )
