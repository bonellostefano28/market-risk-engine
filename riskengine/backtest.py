"""Rolling out-of-sample VaR backtest, coverage tests and Basel traffic light.

The engine takes any callable with signature f(window_returns, alpha) -> VaR.
Plain functions cover the static methods; GarchVaR is a small stateful
callable that can optionally skip refits to speed the backtest up.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .methods import hist_var, normal_var


# --- per-method VaR estimators (one rolling window -> one VaR) ----------------

def var_historical(window, alpha):
    return hist_var(window, alpha)


def var_parametric(window, alpha):
    return normal_var(np.std(window, ddof=1), alpha)


def var_ewma(window, alpha, lam=0.94):
    r = np.asarray(window)
    s2 = np.var(r, ddof=1)
    for x in r:
        s2 = lam * s2 + (1 - lam) * x * x
    return normal_var(np.sqrt(s2), alpha)


class GarchVaR:
    """GARCH(1,1) + FHS VaR for the rolling backtest.

    For a 1-day horizon the FHS bootstrap is redundant: the empirical
    quantile of the standardized innovations gives the same VaR in closed
    form, VaR = -(mu + sigma_next * q_{1-alpha}(z)).

    Refitting is the bottleneck of the whole backtest, so refit_every > 1
    reuses the last parameters and just re-filters the window in between.
    That's roughly a k-fold speedup, but stale parameters can make the model
    react a day or two late right after a volatility shock, and the
    Christoffersen independence test does pick that up (on my sample,
    refit_every=2 matches the daily refit exactly, 5 does not). Default is
    the exact daily refit; loosen it only for quick experiments.
    """

    def __init__(self, refit_every=1):
        self.refit_every = refit_every
        self._calls = 0
        self._params = None  # (mu, omega, a, b)

    def _fit(self, r):
        from arch import arch_model
        res = arch_model(r, vol="GARCH", p=1, q=1,
                         mean="Constant", dist="normal").fit(disp="off")
        p = res.params
        self._params = (p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"])

    def __call__(self, window, alpha):
        r = np.asarray(window) * 100  # arch works better on scaled returns
        if self._params is None or self._calls % self.refit_every == 0:
            try:
                self._fit(r)
            except Exception:
                if self._params is None:  # never converged: fall back
                    self._calls += 1
                    return hist_var(window, alpha)
        self._calls += 1

        mu, omega, a, b = self._params
        eps = r - mu
        # filter the conditional variance through the window
        s2 = np.empty(len(r) + 1)
        pers = a + b
        s2[0] = omega / (1 - pers) if pers < 0.999 else eps.var()
        for t in range(len(r)):
            s2[t + 1] = omega + a * eps[t] ** 2 + b * s2[t]

        z = eps / np.sqrt(s2[:-1])
        q = np.quantile(z, 1 - alpha)
        return -(mu + np.sqrt(s2[-1]) * q) / 100


# name -> factory, so each backtest gets a fresh (stateless or reset) estimator
VAR_ESTIMATORS = {
    "Historical": lambda: var_historical,
    "Parametric": lambda: var_parametric,
    "Parametric EWMA": lambda: var_ewma,
    "GARCH+FHS": lambda: GarchVaR(),
}


# --- rolling engine -----------------------------------------------------------

def rolling_backtest(port_returns, var_func, window=250, alpha=0.99):
    """For each day t, estimate VaR on the previous `window` returns only
    (no look-ahead) and flag an exception if the realized loss exceeds it."""
    r = pd.Series(port_returns).dropna()
    vals = r.values
    rec = []
    for t in range(window, len(r)):
        var_t = var_func(vals[t - window:t], alpha)
        realized = vals[t]
        rec.append((r.index[t], realized, var_t, realized < -var_t))
    out = pd.DataFrame(rec, columns=["date", "return", "VaR", "exception"])
    return out.set_index("date")


def exception_summary(bt, alpha=0.99):
    n, x = len(bt), int(bt["exception"].sum())
    return dict(n_obs=n, exceptions=x, expected=(1 - alpha) * n,
                exc_rate=x / n, target_rate=1 - alpha)


# --- coverage tests -----------------------------------------------------------

def _xlogp(k, p):
    """k*log(p) with the 0*log(0)=0 convention."""
    return k * np.log(p) if k > 0 else 0.0


def kupiec_pof(n, x, alpha=0.99):
    """Kupiec proportion-of-failures test (unconditional coverage).
    H0: the true exception rate is 1-alpha. LR ~ chi2(1)."""
    p, pi = 1 - alpha, x / n
    ll0 = _xlogp(x, p) + _xlogp(n - x, 1 - p)
    ll1 = _xlogp(x, pi) + _xlogp(n - x, 1 - pi)
    LR = -2 * (ll0 - ll1)
    return LR, stats.chi2.sf(LR, df=1)


def christoffersen_ind(exceptions):
    """Christoffersen independence test.
    H0: today's exception does not depend on yesterday's. LR ~ chi2(1)."""
    I = np.asarray(exceptions).astype(int)
    prev, cur = I[:-1], I[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    pi01 = n01 / (n00 + n01) if n00 + n01 > 0 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll0 = _xlogp(n00 + n10, 1 - pi) + _xlogp(n01 + n11, pi)
    ll1 = (_xlogp(n00, 1 - pi01) + _xlogp(n01, pi01)
           + _xlogp(n10, 1 - pi11) + _xlogp(n11, pi11))
    LR = -2 * (ll0 - ll1)
    counts = dict(n00=n00, n01=n01, n10=n10, n11=n11, pi01=pi01, pi11=pi11)
    return LR, stats.chi2.sf(LR, df=1), counts


def christoffersen_cc(n, x, exceptions, alpha=0.99):
    """Conditional coverage = Kupiec + independence, jointly. LR ~ chi2(2)."""
    LR_uc, _ = kupiec_pof(n, x, alpha)
    LR_ind, _, _ = christoffersen_ind(exceptions)
    LR = LR_uc + LR_ind
    return LR, stats.chi2.sf(LR, df=2)


# --- Basel traffic light --------------------------------------------------------

_YELLOW_ADDON = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def basel_zone(x):
    """Zone and capital multiplier for x exceptions over 250 days at 99%."""
    if x <= 4:
        return "green", 3.00
    if x <= 9:
        return "yellow", 3.00 + _YELLOW_ADDON[x]
    return "red", 4.00


def rolling_basel(exceptions, window=250):
    s = pd.Series(np.asarray(exceptions).astype(int))
    counts = s.rolling(window).sum().dropna()
    cur, worst = int(counts.iloc[-1]), int(counts.max())
    zc, mc = basel_zone(cur)
    zw, mw = basel_zone(worst)
    return dict(current=cur, current_zone=zc, current_mult=mc,
                worst=worst, worst_zone=zw, worst_mult=mw)


# --- validation report ----------------------------------------------------------

def validation_row(n, x, exceptions, alpha=0.99):
    """The three tests plus Basel zones in one row.

    The overall verdict is based on conditional coverage (the joint test).
    Kupiec and independence are still reported separately: a method can pass
    the joint test with a marginal independence component, and that should
    be visible rather than hidden behind a single pass/fail.
    """
    _, p_uc = kupiec_pof(n, x, alpha)
    _, p_ind, _ = christoffersen_ind(exceptions)
    _, p_cc = christoffersen_cc(n, x, exceptions, alpha)
    bz = rolling_basel(exceptions)
    return {
        "exceptions": int(x),
        "expected": (1 - alpha) * n,
        "exc_rate": x / n,
        "kupiec_p": p_uc,
        "chr_ind_p": p_ind,
        "cc_p": p_cc,
        "basel_current": bz["current_zone"],
        "basel_worst": bz["worst_zone"],
        "overall_pass": bool(p_cc > 0.05),
    }


def validation_report(results, alpha=0.99):
    """results: {method_name: (n, x, exceptions_array)}, order preserved.
    Returns (DataFrame, recommended method or None). Recommended = best
    calibrated (highest cc_p) among the methods that pass."""
    rows = [{"method": name, **validation_row(n, x, exc, alpha)}
            for name, (n, x, exc) in results.items()]
    df = pd.DataFrame(rows).set_index("method")
    passers = df[df["overall_pass"]]
    recommended = passers["cc_p"].astype(float).idxmax() if len(passers) else None
    return df, recommended
