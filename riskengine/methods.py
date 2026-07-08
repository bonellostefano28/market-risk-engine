"""VaR and Expected Shortfall estimators.

Conventions used throughout:
- returns are daily log-returns in decimal form
- VaR and ES are reported as positive numbers (loss magnitudes)
- alpha is the confidence level (0.99 means 99% VaR)
"""
import numpy as np
import pandas as pd
from scipy import signal, stats


def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    return returns @ np.asarray(weights, float)


# --- historical --------------------------------------------------------------

def hist_var(x, alpha=0.99):
    return -np.quantile(x, 1 - alpha)


def hist_es(x, alpha=0.99):
    x = np.asarray(x)
    q = np.quantile(x, 1 - alpha)
    return -x[x <= q].mean()


# --- parametric (normal) ------------------------------------------------------

def normal_var(sigma, alpha=0.99, mu=0.0):
    return -mu + stats.norm.ppf(alpha) * sigma


def normal_es(sigma, alpha=0.99, mu=0.0):
    z = stats.norm.ppf(alpha)
    return -mu + sigma * stats.norm.pdf(z) / (1 - alpha)


def portfolio_sigma(weights, cov):
    w = np.asarray(weights, float)
    return float(np.sqrt(w @ cov @ w))


def ewma_vol(x, lam=0.94):
    """RiskMetrics-style EWMA volatility of a return series.

    Returns the full series of sigma estimates; the last value is the
    forecast for the next day. For fixed weights, running EWMA on the
    portfolio series is equivalent to the full covariance recursion.
    Implemented as an IIR filter, so no Python loop.
    """
    x2 = np.asarray(x, float) ** 2
    seed = np.var(x, ddof=1)
    s2, _ = signal.lfilter([1 - lam], [1, -lam], x2, zi=[lam * seed])
    return np.sqrt(s2)


# --- Monte Carlo --------------------------------------------------------------

def simulate_normal(mu, cov, n_sims=200_000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, cov, size=n_sims)


def simulate_t(mu, cov, nu=5, n_sims=200_000, seed=0):
    """Multivariate Student-t scenarios. The (nu-2)/nu scaling makes the
    resulting covariance equal to cov."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov * (nu - 2) / nu)
    z = rng.standard_normal((n_sims, len(mu)))
    g = rng.chisquare(nu, n_sims) / nu
    return mu + (z @ L.T) / np.sqrt(g)[:, None]


def mc_var_es(sim_returns, weights, alpha=0.99):
    port = sim_returns @ np.asarray(weights, float)
    return hist_var(port, alpha), hist_es(port, alpha)


# --- GARCH(1,1) + filtered historical simulation ------------------------------

def garch_fhs(port_returns, alpha=0.99, n_sims=200_000, seed=0):
    """Single-shot GARCH(1,1) + FHS estimate on the full sample.

    Fits the model, bootstraps the standardized residuals (which keep the
    empirical fat tails) and rescales them by tomorrow's forecast volatility.
    Returns (VaR, ES, fitted arch result). arch is happier with returns
    scaled by 100, hence the round trip.
    """
    from arch import arch_model

    r = np.asarray(port_returns) * 100
    res = arch_model(r, vol="GARCH", p=1, q=1,
                     mean="Constant", dist="normal").fit(disp="off")
    z = res.resid / res.conditional_volatility
    sigma_next = np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])

    rng = np.random.default_rng(seed)
    boot = rng.choice(z, size=n_sims, replace=True)
    sim = (res.params["mu"] + sigma_next * boot) / 100
    return hist_var(sim, alpha), hist_es(sim, alpha), res
