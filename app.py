"""Market Risk Engine dashboard. Run with: streamlit run app.py"""
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from riskengine.backtest import (VAR_ESTIMATORS, christoffersen_cc,
                                 christoffersen_ind, exception_summary,
                                 kupiec_pof, rolling_backtest, rolling_basel,
                                 validation_report)
from riskengine.data import load_prices, log_returns
from riskengine.decomposition import risk_table
from riskengine.methods import (ewma_vol, garch_fhs, hist_es, hist_var,
                                mc_var_es, normal_es, normal_var,
                                portfolio_returns, portfolio_sigma,
                                simulate_normal, simulate_t)
from riskengine.plots import (plot_backtest, plot_distribution,
                              plot_risk_decomposition, plot_stress_scenarios)
from riskengine.stress import (historical_scenario, hypothetical_scenario,
                               worst_day_scenario)

st.set_page_config(page_title="Market Risk Engine", layout="wide")
st.title("Market Risk Engine")
st.caption("Multi-method VaR and Expected Shortfall on a multi-asset portfolio")

MIN_OBS_FOR_BACKTEST = 300  # 250-day window plus some test days

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Portfolio")
tickers_raw = st.sidebar.text_input("Tickers (comma-separated)",
                                    "SPY, EZU, EWI, TLT, GLD")
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
if not tickers:
    st.warning("Enter at least one ticker.")
    st.stop()

st.sidebar.subheader("Weights")
raw_w = [st.sidebar.number_input(t, min_value=0.0,
                                 value=round(1 / len(tickers), 2),
                                 step=0.05, key=f"w_{t}") for t in tickers]
w = np.array(raw_w, float)
if w.sum() <= 0:
    st.sidebar.error("Weights must sum to a positive value.")
    st.stop()
weights = w / w.sum()
if abs(w.sum() - 1.0) < 1e-6:
    st.sidebar.caption("Long-only, weights sum to 100%.")
else:
    st.sidebar.caption(f"Long-only, entered sum {w.sum():.2f}, rescaled to 100%.")

st.sidebar.header("Parameters")
conf = st.sidebar.slider("Confidence", 0.90, 0.99, 0.99, 0.005)
start = st.sidebar.date_input("Start date", datetime.date(2018, 1, 1))
notional = st.sidebar.number_input("Notional (EUR)", min_value=0,
                                   value=1_000_000, step=100_000)
headline_model = st.sidebar.selectbox(
    "Headline model (summary tiles)",
    ["Historical", "Parametric (flat)", "Parametric EWMA", "GARCH+FHS"],
    help="Drives the top tiles only; the tabs always compare all methods. "
         "These are single-shot estimates, so the summary stays instant.")

# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Downloading prices...")
def get_returns(tickers, start, end):
    return log_returns(load_prices(list(tickers), start, end))


try:
    rets = get_returns(tuple(tickers), str(start),
                       datetime.date.today().isoformat())
except Exception as e:
    st.error(f"Data download error: {e}")
    st.stop()

if rets.empty or rets.shape[1] < len(tickers):
    st.error("Some tickers have no valid data in this period. Check the symbols.")
    st.stop()

port = portfolio_returns(rets, weights)
cov = rets.cov().values
mu = rets.mean().values
var_ref = hist_var(port, conf)
es_ref = hist_es(port, conf)


@st.cache_data(show_spinner=False)
def run_backtest(method, port, conf):
    # VAR_ESTIMATORS holds factories, so each run gets a fresh estimator
    return rolling_backtest(port, VAR_ESTIMATORS[method](),
                            window=250, alpha=conf)


# ---------------------------------------------------------------- summary
# Everything up here must be fast: single-shot estimates only. The one tile
# that needs an exception series (Basel zone) reuses the cached historical
# backtest, never the GARCH one.
st.subheader("Current market risk summary")

sig_flat = portfolio_sigma(weights, cov)
sig_ewma = ewma_vol(port.values)[-1]

if headline_model == "Historical":
    h_var, h_es = hist_var(port, conf), hist_es(port, conf)
elif headline_model == "Parametric (flat)":
    h_var, h_es = normal_var(sig_flat, conf), normal_es(sig_flat, conf)
elif headline_model == "Parametric EWMA":
    h_var, h_es = normal_var(sig_ewma, conf), normal_es(sig_ewma, conf)
else:
    h_var, h_es, _ = garch_fhs(port, conf)

decomp, var_p0 = risk_table(weights, cov, conf, names=list(rets.columns))
div_ratio = decomp["standalone_var"].sum() / var_p0
top_asset = decomp["pct_contribution"].idxmax()
top_contrib = float(decomp["pct_contribution"].max())

vol_ratio = sig_ewma / sig_flat
regime = ("Elevated" if vol_ratio > 1.2
          else "Calm" if vol_ratio < 0.8 else "Normal")

basel_now = "n/a"
if len(rets) >= MIN_OBS_FOR_BACKTEST:
    bt_hist = run_backtest("Historical", port, conf)
    basel_now = rolling_basel(bt_hist["exception"].values)["current_zone"].capitalize()

r1 = st.columns(4)
r1[0].metric("Portfolio value", f"EUR {notional:,.0f}")
r1[1].metric(f"VaR 1-day ({headline_model})", f"{h_var:.2%}",
             f"EUR {h_var * notional:,.0f}", delta_color="off")
r1[2].metric("Expected Shortfall", f"{h_es:.2%}",
             f"EUR {h_es * notional:,.0f}", delta_color="off")
r1[3].metric("Volatility (EWMA, ann.)", f"{sig_ewma * np.sqrt(252):.1%}")

r2 = st.columns(4)
r2[0].metric("Volatility regime", regime,
             f"{vol_ratio - 1:+.0%} vs long-run", delta_color="inverse")
r2[1].metric("Diversification ratio", f"{div_ratio:.2f}x")
r2[2].metric("Top risk contributor", top_asset,
             f"{top_contrib:.1%} of VaR", delta_color="off")
r2[3].metric("Basel zone (current)", basel_now)

c1, c2 = st.columns(2)
c1.metric("Assets", rets.shape[1])
c2.metric("Observations", f"{len(rets):,}")
st.caption(f"Sample period: {rets.index[0]:%b %Y} to {rets.index[-1]:%b %Y}")
st.bar_chart(pd.DataFrame({"weight": weights}, index=rets.columns))

tab_m, tab_d, tab_b, tab_v, tab_r, tab_s = st.tabs(
    ["Methods", "Distribution", "Backtest", "Validation",
     "Risk decomposition", "Stress test"])

# ---------------------------------------------------------------- methods
with tab_m:
    vg, eg, _ = garch_fhs(port, conf)
    rows = [
        ("Historical", hist_var(port, conf), hist_es(port, conf)),
        ("Parametric (flat)", normal_var(sig_flat, conf), normal_es(sig_flat, conf)),
        ("Parametric EWMA", normal_var(sig_ewma, conf), normal_es(sig_ewma, conf)),
        ("Monte Carlo normal", *mc_var_es(simulate_normal(mu, cov), weights, conf)),
        ("Monte Carlo t (nu=5)", *mc_var_es(simulate_t(mu, cov, nu=5), weights, conf)),
        ("GARCH + FHS", vg, eg),
    ]
    dfm = pd.DataFrame(rows, columns=["Method", "VaR", "ES"])
    dfm["ES/VaR"] = dfm["ES"] / dfm["VaR"]

    st.subheader(f"VaR and Expected Shortfall at {conf:.1%}, 1-day horizon")
    disp = pd.DataFrame({
        "Method": dfm["Method"],
        "VaR %": dfm["VaR"] * 100,
        "ES %": dfm["ES"] * 100,
        "VaR EUR": dfm["VaR"] * notional,
        "ES EUR": dfm["ES"] * notional,
        "ES/VaR": dfm["ES/VaR"],
    })
    styled = (disp.style
              .format({"VaR %": "{:.2f}%", "ES %": "{:.2f}%",
                       "VaR EUR": "{:,.0f}", "ES EUR": "{:,.0f}",
                       "ES/VaR": "{:.2f}"})
              .background_gradient(subset=["ES/VaR"], cmap="Oranges"))
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("ES/VaR shaded: darker means fatter tail (the loss beyond VaR "
               "is larger relative to VaR itself). Normal is around 1.1, "
               "Student-t/GARCH around 1.3, historical around 1.6.")

# ---------------------------------------------------------------- distribution
with tab_d:
    st.subheader("Portfolio return distribution")
    fig = plot_distribution(port, conf)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------- backtest
with tab_b:
    st.subheader("Rolling out-of-sample backtest (250-day window)")
    if len(rets) < MIN_OBS_FOR_BACKTEST:
        st.warning("Not enough data for a 250-day rolling backtest.")
    else:
        method = st.selectbox("Method to backtest", list(VAR_ESTIMATORS))
        with st.spinner("Running backtest..."):
            bt = run_backtest(method, port, conf)
        s = exception_summary(bt, conf)
        exc = bt["exception"].values
        _, p_uc = kupiec_pof(s["n_obs"], s["exceptions"], conf)
        _, p_ind, _ = christoffersen_ind(exc)
        _, p_cc = christoffersen_cc(s["n_obs"], s["exceptions"], exc, conf)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Exceptions", f"{s['exceptions']} / {s['expected']:.0f} exp.")
        k2.metric("Kupiec p", f"{p_uc:.3f}")
        k3.metric("Christoffersen p", f"{p_ind:.3f}")
        k4.metric("Cond. coverage p", f"{p_cc:.3f}",
                  "PASS" if p_cc > 0.05 else "REJECT",
                  delta_color="normal" if p_cc > 0.05 else "inverse")

        fig, ax = plt.subplots(figsize=(11, 4))
        plot_backtest(bt, ax, f"{method}: P&L vs VaR")
        st.pyplot(fig)
        plt.close(fig)

        if len(bt) >= 250:
            b = rolling_basel(exc, window=250)
            st.caption(f"Basel traffic light, current 250d: **{b['current_zone']}** "
                       f"({b['current']} exc., k={b['current_mult']:.2f}); "
                       f"worst: **{b['worst_zone']}** ({b['worst']} exc., "
                       f"k={b['worst_mult']:.2f})")

# ---------------------------------------------------------------- validation
with tab_v:
    st.subheader("Validation report")
    if len(rets) < MIN_OBS_FOR_BACKTEST:
        st.warning("Not enough data for a 250-day rolling backtest.")
    else:
        # Streamlit reruns every tab body on each interaction, so the heavy
        # GARCH backtest sits behind a button; session_state keeps the result
        # visible across reruns and the cache makes later runs instant.
        st.caption("Backtests all methods over the full sample. The GARCH "
                   "run takes a little while the first time, then it's cached.")
        if st.button("Run validation report", type="primary"):
            st.session_state["run_validation"] = True

        if not st.session_state.get("run_validation"):
            st.info("Click the button to backtest all methods and get the "
                    "recommended model.")
        else:
            with st.spinner("Running all backtests..."):
                results = {}
                for m in VAR_ESTIMATORS:
                    bt_m = run_backtest(m, port, conf)
                    sm = exception_summary(bt_m, conf)
                    results[m] = (sm["n_obs"], sm["exceptions"],
                                  bt_m["exception"].values)
            rep, recommended = validation_report(results, conf)
            n_pass = int(rep["overall_pass"].sum())

            if recommended:
                tag = ("the only method that passes" if n_pass == 1
                       else "best calibrated of the methods that pass")
                st.success(f"**Recommended model: {recommended}**, {tag} the "
                           "joint conditional-coverage test.")
            else:
                st.warning("No method passes conditional coverage on this sample.")

            disp = pd.DataFrame({
                "Method": rep.index,
                "Exceptions": [f"{int(e)} / {x:.0f}"
                               for e, x in zip(rep["exceptions"], rep["expected"])],
                "Rate": rep["exc_rate"].values * 100,
                "Kupiec p": rep["kupiec_p"].values,
                "Christoffersen p": rep["chr_ind_p"].values,
                "Cond. coverage p": rep["cc_p"].values,
                "Basel (worst)": rep["basel_worst"].values,
                "Overall": ["PASS" if v else "FAIL"
                            for v in rep["overall_pass"].values],
            })

            GREEN = "background-color: #d4edda; color: #155724"
            RED = "background-color: #f8d7da; color: #721c24"

            def color_p(s):
                return [GREEN if v > 0.05 else RED for v in s]

            def color_overall(s):
                return [(GREEN if v == "PASS" else RED) + "; font-weight: 600"
                        for v in s]

            def bold_recommended(row):
                return ["font-weight: 700"
                        if (c == "Method" and row["Method"] == recommended)
                        else "" for c in row.index]

            styled = (disp.style
                      .format({"Rate": "{:.2f}%", "Kupiec p": "{:.3f}",
                               "Christoffersen p": "{:.3f}",
                               "Cond. coverage p": "{:.3f}"})
                      .apply(color_p, subset=["Kupiec p", "Christoffersen p",
                                              "Cond. coverage p"])
                      .apply(color_overall, subset=["Overall"])
                      .apply(bold_recommended, axis=1))
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption("Green: test not rejected (p > 0.05). Overall verdict is "
                       "conditional coverage, the joint test; Kupiec (count) and "
                       "Christoffersen (independence) are shown separately since "
                       "a method can pass the joint test with a marginal "
                       "independence component. Basel light is defined at "
                       "99% over 250 days.")

# ---------------------------------------------------------------- decomposition
with tab_r:
    st.subheader("Risk decomposition (parametric, variance-covariance)")
    dfd, var_p = risk_table(weights, cov, conf, names=list(rets.columns))
    disp = pd.DataFrame({
        "Asset": dfd.index,
        "Weight": dfd["weight"] * 100,
        "Standalone VaR": dfd["standalone_var"] * 100,
        "Beta": dfd["beta"],
        "Component VaR": dfd["component_var"] * 100,
        "Contribution": dfd["pct_contribution"] * 100,
    }).reset_index(drop=True)

    def color_contribution(row, tol=0.5):
        # red if the asset contributes more risk than its weight, green if less
        styles = [""] * len(row)
        j = list(row.index).index("Contribution")
        if row["Contribution"] > row["Weight"] + tol:
            styles[j] = "background-color: #f8d7da; color: #721c24"
        elif row["Contribution"] < row["Weight"] - tol:
            styles[j] = "background-color: #d4edda; color: #155724"
        return styles

    styled = (disp.style
              .format({"Weight": "{:.1f}%", "Standalone VaR": "{:.2f}%",
                       "Beta": "{:.2f}", "Component VaR": "{:.2f}%",
                       "Contribution": "{:.1f}%"})
              .apply(color_contribution, axis=1))
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(f"Portfolio VaR {var_p:.2%} equals the sum of component VaRs "
               "(Euler). Red: risk concentrator (contributes more than its "
               "weight); green: diversifier.")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_risk_decomposition(dfd, ax)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------- stress
with tab_s:
    st.subheader("Stress test: scenario losses vs VaR/ES")
    labels, losses = [], []

    for name, (a, b) in [("COVID crash 2020", ("2020-02-19", "2020-03-23")),
                         ("2022 selloff", ("2022-01-01", "2022-10-14"))]:
        if not rets.loc[a:b].empty:
            pr, _, _ = historical_scenario(rets, weights, a, b, notional)
            labels.append(name)
            losses.append(pr)

    day, pr, _, _ = worst_day_scenario(rets, weights, notional)
    labels.append(f"Worst day ({day.date()})")
    losses.append(pr)

    hypotheticals = [
        ("Equity -15%, TLT +3%, GLD +5%",
         {"SPY": -.15, "EZU": -.15, "EWI": -.15, "TLT": .03, "GLD": .05}),
        ("Rate shock: eq -8%, TLT -12%, GLD -3%",
         {"SPY": -.08, "EZU": -.08, "EWI": -.08, "TLT": -.12, "GLD": -.03}),
    ]
    for name, sh in hypotheticals:
        pr, _, _ = hypothetical_scenario(sh, list(rets.columns), weights, notional)
        labels.append(name)
        losses.append(pr)

    order = np.argsort(losses)
    labels = [labels[i] for i in order]
    losses = [losses[i] for i in order]

    disp = pd.DataFrame({
        "Scenario": labels,
        "Loss %": [l * 100 for l in losses],
        "P&L EUR": [l * notional for l in losses],
        "x VaR": [abs(l) / var_ref for l in losses],
    })
    styled = (disp.style
              .format({"Loss %": "{:.2f}%", "P&L EUR": "{:,.0f}",
                       "x VaR": "{:.1f}x"})
              .background_gradient(subset=["x VaR"], cmap="Reds"))
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("x VaR shaded: darker means the scenario loss is a larger "
               "multiple of the 1-day VaR. Stress losses typically exceed "
               "both VaR and ES.")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_stress_scenarios(labels, losses, var_ref, es_ref, ax)
    st.pyplot(fig)
    plt.close(fig)
