import numpy as np
import pandas as pd
import pytest

from riskengine.decomposition import component_var, risk_table
from riskengine.stress import (historical_scenario, hypothetical_scenario,
                               worst_day_scenario)


@pytest.fixture
def cov():
    # deliberately asymmetric: asset 0 is much riskier
    return np.array([
        [4.0, 0.8, 0.2],
        [0.8, 1.0, 0.1],
        [0.2, 0.1, 0.5],
    ]) * 1e-4


def test_component_var_sums_to_portfolio_var(cov):
    w = [0.5, 0.3, 0.2]
    df, var_p = risk_table(w, cov, alpha=0.99)
    assert df["component_var"].sum() == pytest.approx(var_p)
    assert df["pct_contribution"].sum() == pytest.approx(1.0)
    np.testing.assert_allclose(component_var(w, cov), df["component_var"])


def test_diversification_shows_up(cov):
    df, var_p = risk_table([1 / 3] * 3, cov)
    # sum of standalone VaRs must exceed the diversified portfolio VaR
    assert df["standalone_var"].sum() > var_p
    # the high-vol asset contributes more than its weight
    assert df["pct_contribution"].iloc[0] > 1 / 3


def _fake_returns():
    idx = pd.bdate_range("2021-01-01", periods=10)
    data = np.full((10, 2), 0.001)
    data[4] = [-0.05, -0.02]  # one bad day
    return pd.DataFrame(data, index=idx, columns=["A", "B"])


def test_hypothetical_scenario_pnl():
    _, pnl, detail = hypothetical_scenario(
        {"A": -0.10}, ["A", "B"], [0.5, 0.5], notional=1_000_000)
    assert pnl == pytest.approx(-50_000)
    assert detail.loc["B", "shock"] == 0.0


def test_worst_day_is_the_bad_day():
    rets = _fake_returns()
    day, port_ret, _, _ = worst_day_scenario(rets, [0.5, 0.5])
    assert day == rets.index[4]
    assert port_ret < 0


def test_historical_scenario_compounds_log_returns():
    rets = _fake_returns()
    port_ret, _, detail = historical_scenario(
        rets, [1.0, 0.0], rets.index[0], rets.index[-1])
    expected = np.expm1(rets["A"].sum())
    assert port_ret == pytest.approx(expected)
    assert detail.loc["A", "shock"] == pytest.approx(expected)
