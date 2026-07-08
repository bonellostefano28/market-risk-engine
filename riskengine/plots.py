"""Matplotlib figures used by the dashboard."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .methods import hist_es, hist_var


def plot_distribution(port_returns, alpha=0.99, bins=80):
    var = hist_var(port_returns, alpha)
    es = hist_es(port_returns, alpha)
    mu, sigma = port_returns.mean(), port_returns.std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.linspace(port_returns.min(), port_returns.max(), 500)

    ax1.hist(port_returns, bins=bins, density=True, alpha=0.6,
             color="steelblue", edgecolor="white", label="Empirical")
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), "k--", lw=2,
             label=f"Normal (sigma={sigma:.4f})")
    ax1.axvline(-var, color="orange", lw=2, label=f"VaR {alpha:.0%} = {var:.2%}")
    ax1.axvline(-es, color="red", lw=2, label=f"ES {alpha:.0%} = {es:.2%}")
    ax1.set_title("Portfolio return distribution")
    ax1.set_xlabel("Daily return")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)

    # left tail on a log scale: this is where the fat tails show up
    ax2.hist(port_returns, bins=bins, density=True, alpha=0.6,
             color="steelblue", edgecolor="white", label="Empirical")
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), "k--", lw=2, label="Normal")
    ax2.axvline(-var, color="orange", lw=2, label="VaR")
    ax2.axvline(-es, color="red", lw=2, label="ES")
    ax2.set_yscale("log")
    ax2.set_xlim(port_returns.min(), -var * 0.6)
    ax2.set_title("Left tail (log scale)")
    ax2.set_xlabel("Daily return")
    ax2.set_ylabel("Density (log)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_backtest(bt, ax, title=""):
    """Daily P&L against the forecast VaR line, exceptions highlighted.
    Clustered exceptions are visible at a glance."""
    ax.plot(bt.index, bt["return"], color="steelblue", lw=0.6, alpha=0.7,
            label="Daily P&L")
    ax.plot(bt.index, -bt["VaR"], color="black", lw=1.1, label="-VaR forecast")
    exc = bt[bt["exception"]]
    ax.scatter(exc.index, exc["return"], color="red", s=16, zorder=5,
               label=f"Exceptions ({len(exc)})")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(title)
    ax.set_ylabel("Return")
    ax.legend(fontsize=8, loc="lower left")


def plot_risk_decomposition(df, ax=None):
    """Side-by-side bars: weight vs VaR contribution per asset.
    df is the table from decomposition.risk_table()."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df["weight"] * 100, width,
           color="lightgray", edgecolor="grey", label="Weight %")
    colors = ["firebrick" if c >= 0 else "seagreen"
              for c in df["pct_contribution"]]
    ax.bar(x + width / 2, df["pct_contribution"] * 100, width,
           color=colors, label="VaR contribution %")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("%")
    ax.set_title("Weight vs risk contribution (component VaR)")
    ax.legend(fontsize=9)


def plot_stress_scenarios(labels, losses_pct, var, es, ax=None):
    """Horizontal bars of scenario losses with VaR and ES as reference."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(labels))
    ax.barh(y, np.array(losses_pct) * 100, color="firebrick", alpha=0.85)
    ax.axvline(-var * 100, color="orange", lw=2, label=f"VaR ({-var:.1%})")
    ax.axvline(-es * 100, color="black", lw=2, ls="--", label=f"ES ({-es:.1%})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Portfolio loss (%)")
    ax.set_title("Scenario losses vs VaR/ES")
    ax.legend(fontsize=9)
