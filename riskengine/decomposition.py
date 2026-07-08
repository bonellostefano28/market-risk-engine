"""Marginal and component VaR in the variance-covariance framework.

Component VaRs sum exactly to the portfolio VaR (Euler decomposition), which
is what makes them usable as risk contributions.
"""
import numpy as np
import pandas as pd
from scipy import stats


def marginal_var(weights, cov, alpha=0.99):
    """dVaR/dw_i = z_alpha * (cov @ w)_i / sigma_p."""
    w = np.asarray(weights, float)
    cov = np.asarray(cov, float)
    sigma_p = np.sqrt(w @ cov @ w)
    return stats.norm.ppf(alpha) * (cov @ w) / sigma_p


def component_var(weights, cov, alpha=0.99):
    w = np.asarray(weights, float)
    return w * marginal_var(w, cov, alpha)


def risk_table(weights, cov, alpha=0.99, names=None):
    """Per-asset decomposition table plus the portfolio VaR it sums to."""
    w = np.asarray(weights, float)
    cov = np.asarray(cov, float)
    z = stats.norm.ppf(alpha)
    sigma_p = np.sqrt(w @ cov @ w)
    var_p = z * sigma_p

    mvar = z * (cov @ w) / sigma_p
    cvar = w * mvar
    df = pd.DataFrame({
        "weight": w,
        "standalone_var": z * np.sqrt(np.diag(cov)),
        "beta": (cov @ w) / sigma_p ** 2,
        "marginal_var": mvar,
        "component_var": cvar,
        "pct_contribution": cvar / var_p,
    }, index=names)
    return df, var_p
