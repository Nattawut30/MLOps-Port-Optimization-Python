"""
Estimates expected returns two ways: historical mean and Black-Litterman.

Black-Litterman anchors on market equilibrium (implied by the covariance
matrix and market weights) rather than trusting noisy historical averages
outright. Real market-cap weights would require another data source, so
this uses equal weights as a disclosed simplification. With no investor
views supplied, the formula has nothing to blend in and correctly returns
the equilibrium estimate directly — that's the intended zero-view behavior,
not a shortcut.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.settings import SILVER_DIR

TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_AVERSION = 2.5  # standard convention (He & Litterman, 1999)
DEFAULT_TAU = 0.05  # standard small-scalar convention for prior uncertainty


def historical_mean_returns(returns: pd.DataFrame) -> pd.Series:
    """Simple annualized mean of daily log returns."""
    return returns.mean() * TRADING_DAYS_PER_YEAR


def black_litterman_returns(
    covariance: pd.DataFrame,
    market_weights: pd.Series | None = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    tau: float = DEFAULT_TAU,
    views: dict[str, float] | None = None,
) -> pd.Series:
    """Black-Litterman expected returns.

    With no views, returns the market-implied equilibrium directly.
    With views, blends them in with the standard formula.
    """
    tickers = covariance.columns
    sigma = covariance.values

    if market_weights is None:
        # No market-cap data source in this project; equal weight is the
        # disclosed stand-in for a true market portfolio.
        market_weights = pd.Series(1.0 / len(tickers), index=tickers)
    w_mkt = market_weights.reindex(tickers).values

    equilibrium = risk_aversion * sigma @ w_mkt  # π = δ Σ w_mkt

    if not views:
        return pd.Series(equilibrium, index=tickers)

    # P: one row per view, 1.0 in that ticker's column (absolute views only)
    view_tickers = list(views.keys())
    p_matrix = np.zeros((len(view_tickers), len(tickers)))
    for i, ticker in enumerate(view_tickers):
        p_matrix[i, tickers.get_loc(ticker)] = 1.0
    q_vector = np.array([views[t] for t in view_tickers])

    tau_sigma = tau * sigma
    omega = np.diag(np.diag(p_matrix @ tau_sigma @ p_matrix.T))  # view uncertainty

    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(omega)

    posterior_cov = np.linalg.inv(
        tau_sigma_inv + p_matrix.T @ omega_inv @ p_matrix
    )
    posterior_mean = posterior_cov @ (
        tau_sigma_inv @ equilibrium + p_matrix.T @ omega_inv @ q_vector
    )
    return pd.Series(posterior_mean, index=tickers)


def estimate(
    returns: pd.DataFrame | None = None,
    covariance: pd.DataFrame | None = None,
) -> Path:
    """Compute both estimates and write them to the silver layer."""
    if returns is None:
        returns = pd.read_parquet(SILVER_DIR / "returns.parquet")
    if covariance is None:
        covariance = pd.read_parquet(SILVER_DIR / "covariance.parquet")

    expected_returns = pd.DataFrame(
        {
            "historical_mean": historical_mean_returns(returns),
            "black_litterman": black_litterman_returns(covariance),
        }
    )

    output_path = SILVER_DIR / "expected_returns.parquet"
    expected_returns.to_parquet(output_path)
    return output_path


if __name__ == "__main__":
    path = estimate()
    print(f"Wrote silver expected returns to {path}")
