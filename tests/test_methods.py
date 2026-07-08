import numpy as np
import pytest

from riskengine import methods as m


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_hist_var_matches_quantile(rng):
    x = rng.normal(0, 0.01, 5000)
    assert m.hist_var(x, 0.99) == pytest.approx(-np.quantile(x, 0.01))


def test_hist_es_exceeds_var(rng):
    x = rng.normal(0, 0.01, 5000)
    assert m.hist_es(x, 0.99) > m.hist_var(x, 0.99)


def test_normal_var_es_closed_form():
    sigma = 0.012
    # against known z values at 99%
    assert m.normal_var(sigma, 0.99) == pytest.approx(2.3263 * sigma, rel=1e-3)
    assert m.normal_es(sigma, 0.99) == pytest.approx(2.6652 * sigma, rel=1e-3)
    assert m.normal_es(sigma, 0.99) > m.normal_var(sigma, 0.99)


def test_normal_var_converges_to_empirical(rng):
    sigma = 0.01
    x = rng.normal(0, sigma, 2_000_000)
    assert m.hist_var(x, 0.99) == pytest.approx(m.normal_var(sigma, 0.99), rel=0.02)


def test_ewma_vol_matches_naive_recursion(rng):
    x = rng.normal(0, 0.01, 300)
    lam = 0.94
    s2 = np.var(x, ddof=1)
    expected = []
    for r in x:
        s2 = lam * s2 + (1 - lam) * r * r
        expected.append(np.sqrt(s2))
    np.testing.assert_allclose(m.ewma_vol(x, lam), expected, rtol=1e-10)


def test_simulate_t_recovers_covariance(rng):
    cov = np.array([[1.0, 0.3], [0.3, 2.0]]) * 1e-4
    sim = m.simulate_t(np.zeros(2), cov, nu=6, n_sims=400_000, seed=1)
    np.testing.assert_allclose(np.cov(sim.T), cov, rtol=0.05)


def test_mc_t_has_fatter_tail_than_normal():
    cov = np.eye(2) * 1e-4
    mu = np.zeros(2)
    w = [0.5, 0.5]
    v_n, e_n = m.mc_var_es(m.simulate_normal(mu, cov, seed=3), w)
    v_t, e_t = m.mc_var_es(m.simulate_t(mu, cov, nu=4, seed=3), w)
    assert e_t / v_t > e_n / v_n


def test_portfolio_sigma():
    cov = np.array([[0.04, 0.0], [0.0, 0.01]])
    assert m.portfolio_sigma([1, 0], cov) == pytest.approx(0.2)
    assert m.portfolio_sigma([0.5, 0.5], cov) == pytest.approx(np.sqrt(0.0125))
