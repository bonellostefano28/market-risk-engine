"""Price download and return calculation.

Prices come from yfinance. A parquet snapshot committed under data/ acts as
an offline fallback: Yahoo occasionally rate-limits requests from Streamlit
Cloud, and without the fallback the public demo would just break. Refresh it
locally with scripts/update_snapshot.py.
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_PATH = os.path.join(_ROOT, "data", "prices_snapshot.parquet")


def _as_list(tickers) -> list:
    return [tickers] if isinstance(tickers, str) else list(tickers)


def _clean(prices: pd.DataFrame, tickers) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    # yfinance sorts columns alphabetically; restore the requested order so
    # weights, covariances and labels stay aligned downstream
    prices = prices.reindex(columns=_as_list(tickers))
    # different markets, different holidays: forward-fill then drop the head
    return prices.ffill().dropna()


def _from_snapshot(tickers, start, end):
    if not os.path.exists(SNAPSHOT_PATH):
        return None
    snap = pd.read_parquet(SNAPSHOT_PATH)
    cols = _as_list(tickers)
    if not set(cols).issubset(snap.columns):
        return None
    out = snap.reindex(columns=cols).loc[str(start):str(end)].dropna()
    return out if not out.empty else None


def load_prices(tickers, start, end, use_snapshot: bool = True) -> pd.DataFrame:
    """Daily adjusted-close prices, falling back to the parquet snapshot if
    the live download fails or comes back incomplete."""
    cols = _as_list(tickers)
    prices = None
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        prices = _clean(raw["Close"].copy(), tickers)
        if prices.empty or prices.shape[1] != len(cols):
            prices = None
    except Exception:
        prices = None

    if prices is not None:
        return prices

    if use_snapshot:
        snap = _from_snapshot(tickers, start, end)
        if snap is not None:
            return snap

    raise RuntimeError(
        f"Price download failed and no usable snapshot for {cols}. "
        "Check the tickers and connectivity, or refresh the snapshot with "
        "`python scripts/update_snapshot.py`."
    )


def save_snapshot(prices: pd.DataFrame, path: str = SNAPSHOT_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prices.to_parquet(path)
    return path


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()
