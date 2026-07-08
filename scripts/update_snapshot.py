"""Refresh the offline price snapshot (fallback for the deployed demo).

Run locally, where yfinance works reliably, then commit the parquet:

    python scripts/update_snapshot.py
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from riskengine.data import load_prices, save_snapshot

TICKERS = ["SPY", "EZU", "EWI", "TLT", "GLD"]
START = "2018-01-01"

if __name__ == "__main__":
    end = datetime.date.today().isoformat()
    # use_snapshot=False forces a live download, never re-saves stale data
    prices = load_prices(TICKERS, START, end, use_snapshot=False)
    path = save_snapshot(prices)
    print(f"Snapshot saved to {path}")
    print(f"Tickers: {list(prices.columns)}")
    print(f"Period : {prices.index[0].date()} -> {prices.index[-1].date()} "
          f"({len(prices)} rows)")
