"""Stress testing: historical event replay and hypothetical factor shocks.

Shocks are simple (not log) returns applied to the current weights:
P&L = notional * sum_i w_i * r_i.
"""
import numpy as np
import pandas as pd


def apply_scenario(asset_returns, weights, notional=1_000_000):
    w = np.asarray(weights, float)
    r = np.asarray(asset_returns, float)
    port_ret = float(w @ r)
    return port_ret, port_ret * notional, notional * w * r


def historical_scenario(returns, weights, start, end, notional=1_000_000):
    """Replay: cumulative return of each asset over [start, end] applied to
    today's portfolio. Log-returns add over time, then convert to simple."""
    window = returns.loc[start:end]
    shocks = np.expm1(window.sum().values)
    port_ret, pnl, pnl_asset = apply_scenario(shocks, weights, notional)
    detail = pd.DataFrame({"shock": shocks, "pnl": pnl_asset},
                          index=returns.columns)
    return port_ret, pnl, detail


def worst_day_scenario(returns, weights, notional=1_000_000):
    """The single worst portfolio day in the sample."""
    port = returns @ np.asarray(weights, float)
    day = port.idxmin()
    shocks = np.expm1(returns.loc[day].values)
    port_ret, pnl, pnl_asset = apply_scenario(shocks, weights, notional)
    detail = pd.DataFrame({"shock": shocks, "pnl": pnl_asset},
                          index=returns.columns)
    return day, port_ret, pnl, detail


def hypothetical_scenario(shocks, tickers, weights, notional=1_000_000):
    """shocks: {ticker: simple return}; tickers not mentioned get 0."""
    r = np.array([shocks.get(t, 0.0) for t in tickers], float)
    port_ret, pnl, pnl_asset = apply_scenario(r, weights, notional)
    detail = pd.DataFrame({"shock": r, "pnl": pnl_asset}, index=tickers)
    return port_ret, pnl, detail
