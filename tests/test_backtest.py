import numpy as np
import pandas as pd
import pytest

from riskengine import backtest as bt


def test_kupiec_exact_rate_not_rejected():
    # 10 exceptions out of 1000 at 99% is exactly the expected rate
    LR, p = bt.kupiec_pof(1000, 10, alpha=0.99)
    assert LR == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0)


def test_kupiec_rejects_gross_miscalibration():
    _, p = bt.kupiec_pof(1000, 40, alpha=0.99)
    assert p < 0.001


def test_christoffersen_rejects_clustering():
    exc = np.zeros(1000, int)
    exc[100:110] = 1  # 10 consecutive exceptions
    _, p_clustered, _ = bt.christoffersen_ind(exc)

    rng = np.random.default_rng(0)
    scattered = np.zeros(1000, int)
    scattered[rng.choice(1000, 10, replace=False)] = 1
    _, p_scattered, _ = bt.christoffersen_ind(scattered)

    assert p_clustered < 0.05 < p_scattered


def test_christoffersen_transition_counts():
    _, _, c = bt.christoffersen_ind([0, 1, 1, 0, 0])
    assert (c["n00"], c["n01"], c["n10"], c["n11"]) == (1, 1, 1, 1)


def test_basel_zones():
    assert bt.basel_zone(0) == ("green", 3.00)
    assert bt.basel_zone(4) == ("green", 3.00)
    assert bt.basel_zone(5) == ("yellow", 3.40)
    assert bt.basel_zone(9) == ("yellow", 3.85)
    assert bt.basel_zone(10) == ("red", 4.00)


def test_rolling_backtest_no_lookahead():
    # constant VaR of 2%: exceptions are exactly the days below -2%,
    # and the estimator only ever sees past data
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0, 0.01, 300), index=idx)

    seen = []

    def const_var(window, alpha):
        seen.append(len(window))
        return 0.02

    out = bt.rolling_backtest(r, const_var, window=250, alpha=0.99)
    assert len(out) == 50
    assert all(n == 250 for n in seen)
    expected_exc = (r.iloc[250:] < -0.02).values
    np.testing.assert_array_equal(out["exception"].values, expected_exc)


def test_garch_var_reacts_to_volatility_regime():
    # low-vol history followed by a high-vol burst: the forecast VaR at the
    # end of the burst should be well above the one before it
    rng = np.random.default_rng(1)
    calm = rng.normal(0, 0.005, 400)
    burst = rng.normal(0, 0.025, 100)
    r = np.concatenate([calm, burst])

    g = bt.GarchVaR(refit_every=1)
    var_calm = g(r[150:400], 0.99)
    var_burst = g(r[250:500], 0.99)
    assert var_burst > 1.5 * var_calm


def test_validation_report_recommends_best_calibrated():
    rng = np.random.default_rng(2)
    good = np.zeros(1000, int)
    good[rng.choice(1000, 11, replace=False)] = 1
    bad = np.zeros(1000, int)
    bad[rng.choice(1000, 45, replace=False)] = 1

    results = {
        "good": (1000, int(good.sum()), good),
        "bad": (1000, int(bad.sum()), bad),
    }
    df, recommended = bt.validation_report(results, alpha=0.99)
    assert recommended == "good"
    assert bool(df.loc["bad", "overall_pass"]) is False
