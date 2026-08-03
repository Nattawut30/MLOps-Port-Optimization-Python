"""
Combines the gold-layer outputs (weights, risk metrics, Heston hedges) into
one timestamped summary artifact, the single file streamlit_app.py reads,
keeping the dashboard a thin facade rather than something that assembles
several separate files itself.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.settings import GOLD_DIR
from src.storage import read_parquet, write_parquet


def finalize_summary() -> Path:
    weights = read_parquet(GOLD_DIR / "weights.parquet")
    risk_metrics = read_parquet(GOLD_DIR / "risk_metrics.parquet")
    hedge = read_parquet(GOLD_DIR / "heston_hedge.parquet")

    summary = pd.DataFrame([{
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **{f"weight_max_sharpe_{t}": float(w) for t, w in weights["max_sharpe"].items()},
        **{f"weight_risk_parity_{t}": float(w) for t, w in weights["risk_parity"].items()},
        "gbm_cvar_95": float(risk_metrics.loc["gbm", "cvar_95"]),
        "heston_cvar_95": float(risk_metrics.loc["heston", "cvar_95"]),
        "hedged_tickers": ",".join(hedge["hedged_ticker"]),
        "hedge_mean_agreement_pct_diff": float(hedge["fft_vs_mc_agreement_pct_diff"].mean()),
    }])

    return write_parquet(summary, GOLD_DIR / "run_summary.parquet")
