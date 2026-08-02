"""
Dashboard. Facade over the gold and silver layers only.

This file never imports from settings.py and never calls extractor,
processor, estimation, optimization, simulation, or heston directly.
It reads finished parquet artifacts and renders them. No computation
happens here, matching the two-plane design used throughout this
project: compute runs on a schedule, serving stays thin.

Paths are defined locally rather than imported from settings.py,
since Settings() requires a live Alpha Vantage key to instantiate,
and this file never needs that key.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SILVER_DIR = ROOT_DIR / "data" / "02_silver"
GOLD_DIR = ROOT_DIR / "data" / "03_gold"

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


def safe_load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


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

st.title("Portfolio Optimization")
st.markdown(
    "Created by Nattawut Boonnoon &nbsp;&middot;&nbsp; "
    "[GitHub](https://github.com/Nattawut30) &nbsp;&middot;&nbsp; "
    "[LinkedIn](https://www.linkedin.com/in/nattawut-bn/)"
)
st.divider()

weights = safe_load(GOLD_DIR / "weights.parquet")
expected_returns = safe_load(SILVER_DIR / "expected_returns.parquet")
risk_metrics = safe_load(GOLD_DIR / "risk_metrics.parquet")
terminal_values = safe_load(GOLD_DIR / "simulation_terminal_values.parquet")
hedge = safe_load(GOLD_DIR / "heston_hedge.parquet")
summary = safe_load(GOLD_DIR / "run_summary.parquet")

if all(x is None for x in [weights, expected_returns, risk_metrics, terminal_values, hedge, summary]):
    st.info("No pipeline output found. Run the pipeline first.")
    st.stop()

if summary is not None:
    row = summary.iloc[0]
    run_time = pd.to_datetime(row["run_timestamp_utc"]).strftime("%Y-%m-%d %H:%M UTC")
    cols = st.columns(5)
    cols[0].metric("Last run", run_time)
    cols[1].metric("Hedged position", str(row["hedged_ticker"]))
    cols[2].metric("GBM CVaR 95", f"{row['gbm_cvar_95']:.3f}")
    cols[3].metric("Heston CVaR 95", f"{row['heston_cvar_95']:.3f}")
    cols[4].metric("Hedge agreement", f"{row['hedge_agreement_pct_diff']:.2f}%")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Portfolio Weights")
    if weights is not None:
        fig = go.Figure()
        fig.add_bar(x=weights.index, y=weights["max_sharpe"], name="Max Sharpe", marker_color=ACCENT)
        fig.add_bar(x=weights.index, y=weights["risk_parity"], name="Risk Parity", marker_color=SECOND)
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=340)
        st.plotly_chart(fig, width='stretch')
        st.latex(r"\max_{w}\ \frac{w^T\mu - r_f}{\sqrt{w^T\Sigma w}} \quad \text{s.t.}\ \sum_i w_i = 1,\ w_i \geq 0")
        st.latex(r"w_i(\Sigma w)_i = w_j(\Sigma w)_j \quad \forall\, i, j")
    else:
        st.info("No weights available.")

with col_right:
    st.subheader("Expected Returns")
    if expected_returns is not None:
        fig = go.Figure()
        fig.add_bar(x=expected_returns.index, y=expected_returns["historical_mean"], name="Historical Mean", marker_color=SECOND)
        fig.add_bar(x=expected_returns.index, y=expected_returns["black_litterman"], name="Black-Litterman", marker_color=ACCENT)
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=340)
        st.plotly_chart(fig, width='stretch')
        st.latex(r"\pi = \delta \Sigma w_{mkt}")
        st.latex(r"E[R] = \left[(\tau\Sigma)^{-1} + P^T\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\pi + P^T\Omega^{-1}Q\right]")
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
        st.plotly_chart(fig, width='stretch')
        st.latex(r"\text{VaR}_\alpha = \inf\{l : P(L > l) \leq 1-\alpha\} \qquad \text{CVaR}_\alpha = E[L \mid L \geq \text{VaR}_\alpha]")
    else:
        st.info("No risk metrics available.")

with col_right:
    st.subheader("Terminal Value Distribution")
    if terminal_values is not None:
        fig = go.Figure()
        fig.add_histogram(x=terminal_values["gbm"], name="GBM", marker_color=SECOND, opacity=0.7, nbinsx=60)
        fig.add_histogram(x=terminal_values["heston"], name="Heston", marker_color=ACCENT, opacity=0.7, nbinsx=60)
        fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", height=340)
        st.plotly_chart(fig, width='stretch')
        st.latex(r"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t")
        st.latex(r"dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S \qquad dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v \qquad dW_t^S dW_t^v = \rho\,dt")
    else:
        st.info("No simulation output available.")

st.divider()

st.subheader("Tail Risk Hedge")
if hedge is not None:
    h = hedge.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Spot", f"{h['spot']:.2f}")
    cols[1].metric("Strike", f"{h['strike']:.2f}")
    cols[2].metric("FFT price", f"{h['fft_put_price']:.4f}")
    cols[3].metric("Monte Carlo price", f"{h['mc_put_price']:.4f}")
    st.latex(r"C(K) = \frac{e^{-\alpha k}}{\pi}\int_0^{\infty} e^{-ivk}\,\psi(v)\,dv")
    st.latex(r"P = C - S_0 + Ke^{-rT}")
    params = pd.DataFrame(
        {"value": [h["kappa"], h["theta"], h["v0"], h["vol_of_vol"], h["rho"]]},
        index=["kappa", "theta", "v0", "vol_of_vol", "rho"],
    )
    st.dataframe(params, width='stretch')
else:
    st.info("No hedge data available.")
