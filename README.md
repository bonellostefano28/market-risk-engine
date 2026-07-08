# Market Risk Engine

Multi-method Value-at-Risk and Expected Shortfall for a multi-asset portfolio, with a rolling out-of-sample backtest, risk decomposition and stress testing. Served as a Streamlit dashboard on live market data.

Live demo: https://market-risk-engine-stefano-bonello.streamlit.app/

![Dashboard](figures/dashboard.png)

## What it does

The dashboard opens with a summary strip (VaR, ES, EWMA volatility, volatility regime, diversification ratio, top risk contributor, current Basel zone), then goes deeper in six tabs:

- **Methods** — historical, parametric (variance-covariance), EWMA, Monte Carlo (normal and Student-t) and GARCH(1,1) + filtered historical simulation, compared on VaR, ES and the ES/VaR tail ratio.
- **Distribution** — portfolio returns against the fitted normal, with a log-scale zoom on the left tail.
- **Backtest** — 250-day rolling window, out-of-sample, no look-ahead. Kupiec, Christoffersen and the Basel traffic light on each method.
- **Validation** — one row per method with a pass/reject verdict on conditional coverage, and the recommended model.
- **Risk decomposition** — marginal and component VaR (Euler), so contributions sum exactly to the portfolio VaR.
- **Stress test** — historical replays (COVID crash, 2022 selloff, worst day in sample) and hypothetical factor shocks, benchmarked against VaR and ES.

Tickers, weights, confidence level, sample window and notional are all configurable from the sidebar.

## Backtest results

Rolling 250-day window at 99% confidence, ~1900 test days on a five-asset portfolio (SPY, EZU, EWI, TLT, GLD, equal weights). A method is rejected when the conditional-coverage p-value is below 0.05.

| Method          | Exceptions | Rate  | Kupiec p | Independence p | Cond. coverage p | Verdict  |
|-----------------|-----------:|------:|---------:|---------------:|-----------------:|----------|
| Historical      | 31         | 1.65% | 0.010    | 0.000          | 0.000            | rejected |
| Parametric      | 40         | 2.12% | 0.000    | 0.000          | 0.000            | rejected |
| Parametric EWMA | 39         | 2.07% | 0.000    | 0.053          | 0.000            | rejected |
| GARCH + FHS     | 21         | 1.12% | 0.623    | 0.021          | 0.061            | pass     |

The tests separate two failure modes: Kupiec catches the exception *count* (a tail-shape problem), Christoffersen catches exception *clustering* (a regime-reactivity problem). EWMA is the clean illustration — reactive enough to pass independence, but its normal tails fail coverage. Only GARCH+FHS handles both, and even there the independence component is marginal (a residue of the March 2020 shock), which the validation table shows rather than hides. Exact p-values move a little with the sample end date; the ordering doesn't.

At the worst point of the sample every static method reaches Basel's red zone; GARCH+FHS never leaves yellow.

## Project layout

```
riskengine/
  data.py            price download, parquet snapshot fallback, log-returns
  methods.py         all VaR/ES estimators (historical, parametric, EWMA, MC, GARCH+FHS)
  backtest.py        rolling engine, Kupiec/Christoffersen/Basel, validation report
  decomposition.py   marginal and component VaR
  stress.py          historical replay and hypothetical scenarios
  plots.py           matplotlib figures
app.py               Streamlit dashboard (UI only, no formulas)
scripts/             snapshot refresh
tests/               pytest suite for the engine
```

A couple of implementation notes:

- The GARCH backtest uses the closed-form FHS quantile (for a 1-day horizon the bootstrap adds nothing) and can optionally refit every k days instead of daily. It defaults to daily: with stale parameters the model reacts late after a shock, and the independence test notices.
- For fixed weights, EWMA volatility is computed on the portfolio series directly (equivalent to the full covariance recursion) via `scipy.signal.lfilter`, so there's no Python loop.
- Prices come from yfinance with a committed parquet snapshot as fallback, because Yahoo sometimes rate-limits Streamlit Cloud.

## Run locally

```bash
git clone https://github.com/bonellostefano28/market-risk-engine.git
cd market-risk-engine
python -m venv venv
source venv/bin/activate        # Windows Git Bash: source venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests:

```bash
pip install pytest
python -m pytest
```

## Limitations

- GARCH is fitted on the portfolio return series (univariate), which is correct for fixed weights. A DCC-GARCH extension would handle changing weights.
- The component-VaR decomposition uses the closed-form variance-covariance framework; a simulation-based version would be the natural next step.
- Log-returns aggregate exactly over time but only approximately across assets; at a daily horizon the error is negligible.

## License

MIT — see `LICENSE`.
