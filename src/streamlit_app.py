"""
Dashboard. Facade over the gold and silver layers only.

This file never imports from settings.py and never calls extractor,
processor, estimation, optimization, simulation, or heston directly.
It fetches finished parquet artifacts and renders them. No computation
happens here, matching the two-plane design used throughout this
project: compute runs on a schedule, serving stays thin.

Data comes from the repo's "latest" GitHub Release rather than files
committed to git — the pipeline overwrites those release assets every
run, so freshness no longer depends on a git push or a dashboard
redeploy. Local files under data/ are a fallback only, for offline
development.

Paths are defined locally rather than imported from settings.py,
since Settings() requires a live Alpha Vantage key to instantiate,
and this file never needs that key.
"""

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SILVER_DIR = ROOT_DIR / "data" / "02_silver"
GOLD_DIR = ROOT_DIR / "data" / "03_gold"

# The daily pipeline no longer commits data/ to git — it uploads to this
# release instead, overwriting the same assets in place every run. That
# keeps the repo's size and commit count flat forever, at any pipeline
# frequency. Local files under data/ are kept only as a fallback for
# offline development and the rare moment mid-refresh when the release
# is briefly being recreated.
RELEASE_BASE_URL = (
    "https://github.com/Nattawut30/MLOps-Portfolio-Optimization-Python"
    "/releases/download/latest"
)

ACCENT = "#c6a0f6"
SECOND = "#8aadf4"
BG = "#24273a"
GRID = "#363a4f"
TEXT = "#cad3f5"

PLOTLY_LAYOUT = {
    "paper_bgcolor": BG,
    "plot_bgcolor": BG,
    "font": {"color": TEXT, "family": "monospace", "size": 13},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    "xaxis": {"gridcolor": GRID, "zerolinecolor": GRID},
    "yaxis": {"gridcolor": GRID, "zerolinecolor": GRID},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
}


@st.cache_data(ttl=3600)
def safe_load(filename: str, local_path: Path) -> pd.DataFrame | None:
    try:
        response = requests.get(f"{RELEASE_BASE_URL}/{filename}", timeout=10)
        response.raise_for_status()
        return pd.read_parquet(io.BytesIO(response.content))
    except requests.RequestException:
        if local_path.exists():
            return pd.read_parquet(local_path)
        return None


st.set_page_config(page_title="Portfolio Optimization", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #1e2030;
        border: 1px solid #363a4f;
        border-radius: 6px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

weights = safe_load("weights.parquet", GOLD_DIR / "weights.parquet")
expected_returns = safe_load("expected_returns.parquet", SILVER_DIR / "expected_returns.parquet")
risk_metrics = safe_load("risk_metrics.parquet", GOLD_DIR / "risk_metrics.parquet")
terminal_values = safe_load("simulation_terminal_values.parquet", GOLD_DIR / "simulation_terminal_values.parquet")
hedge = safe_load("heston_hedge.parquet", GOLD_DIR / "heston_hedge.parquet")
summary = safe_load("run_summary.parquet", GOLD_DIR / "run_summary.parquet")

if all(x is None for x in [weights, expected_returns, risk_metrics, terminal_values, hedge, summary]):
    st.info("No pipeline output found. Run the pipeline first.")
    st.stop()

header_left, header_right = st.columns([3, 1])
with header_left:
    st.title("Portfolio Optimization")
    st.markdown(
        "Created by Nattawut Boonnoon &nbsp;&middot;&nbsp; "
        "[GitHub](https://github.com/Nattawut30) &nbsp;&middot;&nbsp; "
        "[LinkedIn](https://www.linkedin.com/in/nattawut-bn/)"
    )
with header_right:
    if summary is not None:
        run_date = pd.to_datetime(summary.iloc[0]["run_timestamp_utc"]).strftime("%Y-%m-%d")
        st.metric("Last run", run_date)

st.divider()

if summary is not None:
    row = summary.iloc[0]
    hedged_count = len(str(row["hedged_tickers"]).split(","))
    cols = st.columns(4)
    cols[0].metric("Hedged positions", str(hedged_count))
    cols[1].metric("GBM CVaR 95", f"{row['gbm_cvar_95']:.3f}")
    cols[2].metric("Heston CVaR 95", f"{row['heston_cvar_95']:.3f}")
    cols[3].metric("Avg hedge agreement", f"{row['hedge_mean_agreement_pct_diff']:.2f}%")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Portfolio Weights")
    if weights is not None:
        fig = go.Figure()
        fig.add_bar(x=weights.index, y=weights["max_sharpe"], name="Max Sharpe", marker_color=ACCENT)
        fig.add_bar(x=weights.index, y=weights["risk_parity"], name="Risk Parity", marker_color=SECOND)
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=340)
        st.plotly_chart(fig, width="stretch")
        st.latex(r"\max_{w}\ \frac{w^T\mu - r_f}{\sqrt{w^T\Sigma w}} \quad \text{s.t.}\ \sum_i w_i = 1,\ w_i \geq 0")
        st.latex(r"w_i(\Sigma w)_i = w_j(\Sigma w)_j \quad \forall\, i, j")
        st.caption(
            "Max Sharpe weights for return per unit of risk. Risk parity "
            "weights so each asset contributes the same amount of "
            "portfolio risk."
        )
    else:
        st.info("No weights available.")

with col_right:
    st.subheader("Expected Returns")
    if expected_returns is not None:
        fig = go.Figure()
        fig.add_bar(x=expected_returns.index, y=expected_returns["historical_mean"], name="Historical Mean", marker_color=SECOND)
        fig.add_bar(x=expected_returns.index, y=expected_returns["black_litterman"], name="Black-Litterman", marker_color=ACCENT)
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=340)
        st.plotly_chart(fig, width="stretch")
        st.latex(r"\pi = \delta \Sigma w_{mkt}")
        st.latex(r"E[R] = \left[(\tau\Sigma)^{-1} + P^T\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\pi + P^T\Omega^{-1}Q\right]")
        st.caption(
            "Black-Litterman starts from the return the market already "
            "implies, then blends in investor views. Historical mean is "
            "the plain average of past returns."
        )
    else:
        st.info("No expected returns available.")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Risk: VaR and CVaR")
    if risk_metrics is not None:
        fig = go.Figure()
        fig.add_bar(x=["VaR 95", "CVaR 95"], y=[risk_metrics.loc["gbm", "var_95"], risk_metrics.loc["gbm", "cvar_95"]], name="GBM", marker_color=SECOND)
        fig.add_bar(x=["VaR 95", "CVaR 95"], y=[risk_metrics.loc["heston", "var_95"], risk_metrics.loc["heston", "cvar_95"]], name="Heston", marker_color=ACCENT)
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=340)
        st.plotly_chart(fig, width="stretch")
        st.latex(r"\text{VaR}_\alpha = \inf\{l : P(L > l) \leq 1-\alpha\} \qquad \text{CVaR}_\alpha = E[L \mid L \geq \text{VaR}_\alpha]")
        st.caption(
            "VaR is the loss not expected to be exceeded at the given "
            "confidence level. CVaR is the average loss in the cases "
            "beyond that point."
        )
    else:
        st.info("No risk metrics available.")

with col_right:
    st.subheader("Terminal Value Distribution")
    if terminal_values is not None:
        fig = go.Figure()
        fig.add_histogram(x=terminal_values["gbm"], name="GBM", marker_color=SECOND, opacity=0.7, nbinsx=60)
        fig.add_histogram(x=terminal_values["heston"], name="Heston", marker_color=ACCENT, opacity=0.7, nbinsx=60)
        fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=340)
        st.plotly_chart(fig, width="stretch")
        st.latex(r"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t")
        st.latex(r"dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S \qquad dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v \qquad dW_t^S dW_t^v = \rho\,dt")
        st.caption(
            "GBM assumes constant volatility. Heston lets volatility move "
            "randomly over time, which produces a wider spread of "
            "outcomes."
        )
    else:
        st.info("No simulation output available.")

st.divider()

st.subheader("Tail Risk Hedge")
if hedge is not None:
    display = hedge[[
        "hedged_ticker", "hedged_weight", "spot", "strike",
        "fft_put_price", "mc_put_price", "fft_vs_mc_agreement_pct_diff",
    ]].rename(columns={
        "hedged_ticker": "Ticker",
        "hedged_weight": "Weight",
        "spot": "Spot",
        "strike": "Strike",
        "fft_put_price": "FFT Price",
        "mc_put_price": "MC Price",
        "fft_vs_mc_agreement_pct_diff": "Agreement %",
    }).set_index("Ticker")
    st.dataframe(display, width="stretch")
    st.latex(r"C(K) = \frac{e^{-\alpha k}}{\pi}\int_0^{\infty} e^{-ivk}\,\psi(v)\,dv")
    st.latex(r"P = C - S_0 + Ke^{-rT}")
    st.caption(
        "Each put is priced two independent ways, FFT and Monte Carlo, "
        "and cross-checked against each other. Put-call parity converts "
        "the FFT call price into a put price."
    )
    params = hedge[["hedged_ticker", "kappa", "theta", "v0", "vol_of_vol", "rho"]].rename(
        columns={"hedged_ticker": "Ticker"}
    ).set_index("Ticker")
    st.dataframe(params, width="stretch")
else:
    st.info("No hedge data available.")
