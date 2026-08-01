"""
Builds portfolio weights two ways: max-Sharpe (Markowitz) and risk parity.

Max-Sharpe needs one expected-return number per asset. It defaults to the
black_litterman column rather than historical_mean, since the previous
step's own output showed historical mean swinging wildly with only ~100
days of history, while Black-Litterman stayed anchored and stable.
historical_mean remains available as an explicit override.

Both optimizers are long-only (weights bounded 0 to 1) — a disclosed
simplification, not the only valid choice.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.settings import GOLD_DIR, SILVER_DIR, settings

MIN_WEIGHT = 1e-6  # avoids divide-by-zero in risk parity's contribution calc
MAX_WEIGHT = 1.0


def max_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> pd.Series:
    """Long-only portfolio maximizing the Sharpe ratio (SLSQP)."""
    tickers = covariance.columns
    mu = expected_returns.reindex(tickers).values
    sigma = covariance.values
    n = len(tickers)

    def negative_sharpe(weights: np.ndarray) -> float:
        portfolio_return = weights @ mu
        portfolio_vol = np.sqrt(weights @ sigma @ weights)
        return -(portfolio_return - risk_free_rate) / portfolio_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, MAX_WEIGHT)] * n
    initial_guess = np.full(n, 1.0 / n)

    result = minimize(
        negative_sharpe,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")

    return pd.Series(result.x, index=tickers)


def risk_parity_weights(covariance: pd.DataFrame) -> pd.Series:
    """Long-only portfolio where every asset contributes equally to risk.

    Each asset's contribution to total portfolio variance is pushed toward
    an equal share (1/n of the total) rather than toward equal dollar
    weight — this is what lets it differ meaningfully from equal-weighting
    when assets have different volatilities or correlations.
    """
    tickers = covariance.columns
    sigma = covariance.values
    n = len(tickers)

    def risk_contribution_error(weights: np.ndarray) -> float:
        portfolio_variance = weights @ sigma @ weights
        marginal_contribution = sigma @ weights
        contribution = weights * marginal_contribution
        target = portfolio_variance / n
        return np.sum((contribution - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n
    # Start slightly off equal-weight — the equal-weight point sits exactly
    # on a near-flat gradient for this objective, which previously caused
    # SLSQP to mistake it for convergence after zero real iterations.
    rng = np.random.default_rng(seed=42)
    initial_guess = rng.dirichlet(np.ones(n))

    result = minimize(
        risk_contribution_error,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Risk parity optimization failed: {result.message}")

    return pd.Series(result.x, index=tickers)


def optimize(
    expected_returns: pd.DataFrame | None = None,
    covariance: pd.DataFrame | None = None,
    return_column: str = "black_litterman",
) -> Path:
    """Compute both weight sets and write them to the gold layer."""
    if expected_returns is None:
        expected_returns = pd.read_parquet(SILVER_DIR / "expected_returns.parquet")
    if covariance is None:
        covariance = pd.read_parquet(SILVER_DIR / "covariance.parquet")

    mu = expected_returns[return_column]

    weights = pd.DataFrame(
        {
            "max_sharpe": max_sharpe_weights(mu, covariance, settings.risk_free_rate),
            "risk_parity": risk_parity_weights(covariance),
        }
    )

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GOLD_DIR / "weights.parquet"
    weights.to_parquet(output_path)
    return output_path


if __name__ == "__main__":
    path = optimize()
    print(f"Wrote gold weights to {path}")
