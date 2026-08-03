"""
Orchestrator: runs every stage in order and logs the run to MLflow.

Each stage function reads its own inputs from disk (bronze/silver/gold),
so this file doesn't pass data between them in memory — that's what keeps
every stage independently re-runnable and inspectable, the medallion
design established from the start of this project.
"""

import mlflow
import pandas as pd

from src import (
    estimation,
    extractor,
    heston,
    optimization,
    output,
    processor,
    simulation,
)
from src.settings import GOLD_DIR, settings


def _sharpe_of(weights_column: str) -> float:
    weights = pd.read_parquet(GOLD_DIR / "weights.parquet")[weights_column]
    expected_returns = pd.read_parquet("data/02_silver/expected_returns.parquet")["black_litterman"]
    covariance = pd.read_parquet("data/02_silver/covariance.parquet")
    port_return = float((weights * expected_returns.reindex(weights.index)).sum())
    port_vol = float((weights.values @ covariance.values @ weights.values) ** 0.5)
    return (port_return - settings.risk_free_rate) / port_vol


def run() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("portfolio-pipeline")

    with mlflow.start_run():
        mlflow.log_params({
            "tickers": ",".join(settings.tickers),
            "start_date": str(settings.start_date),
            "end_date": str(settings.end_date),
            "risk_free_rate": settings.risk_free_rate,
        })

        print("1/6 extracting prices...")
        extractor.extract()
        print("2/6 processing returns and covariance...")
        processor.process()
        print("3/6 estimating expected returns...")
        estimation.estimate()
        print("4/6 optimizing weights...")
        optimization.optimize()
        print("5/6 simulating risk...")
        simulation.simulate()
        print("6/6 pricing tail-risk hedge...")
        heston.run()

        summary_path = output.finalize_summary()
        summary = pd.read_parquet(summary_path).iloc[0]

        mlflow.log_metrics({
            "max_sharpe_sharpe_ratio": _sharpe_of("max_sharpe"),
            "gbm_cvar_95": summary["gbm_cvar_95"],
            "heston_cvar_95": summary["heston_cvar_95"],
            "hedge_fft_vs_mc_agreement_pct_diff": summary["hedge_mean_agreement_pct_diff"],
        })
        mlflow.log_artifact(str(summary_path))

        print(f"\nDone. Run summary: {summary_path}")


if __name__ == "__main__":
    run()
