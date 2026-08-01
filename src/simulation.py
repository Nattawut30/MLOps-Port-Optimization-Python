"""
Simulates forward portfolio value paths two ways: GBM and Heston.

Both use the SAME expected-return assumption (Black-Litterman), so the
comparison isolates one variable: constant volatility (GBM) vs stochastic
volatility (Heston). An earlier version let Heston recompute its own drift
from noisy historical data, which made GBM and Heston centered on different
average outcomes — an invalid comparison, since a higher center can mask a
genuinely fatter tail. Fixed here: theta, v0, and vol-of-vol are estimated
from historical data; the drift is not.

Both assume the portfolio is bought and held at today's max_sharpe weights
(no rebalancing over the horizon) — a disclosed simplification.

Heston is applied to the portfolio's own aggregate historical return series
as a single process, not per-asset — multi-asset Heston needs a much
heavier model (a Wishart-process extension), out of scope here.
kappa and rho are literature-convention defaults, since fitting them
properly needs a paid options feed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.settings import GOLD_DIR, SILVER_DIR

TRADING_DAYS_PER_YEAR = 252
N_PATHS = 10_000
HORIZON_DAYS = 252
RECENT_WINDOW = 20

HESTON_KAPPA = 3.0
HESTON_RHO = -0.7


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def simulate_gbm(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: pd.Series,
    n_paths: int = N_PATHS,
    horizon_days: int = HORIZON_DAYS,
    seed: int = 42,
) -> np.ndarray:
    """Correlated multi-asset GBM, aggregated into a buy-and-hold portfolio."""
    device = _get_device()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    tickers = covariance.columns
    mu_annual = torch.tensor(expected_returns.reindex(tickers).values, dtype=torch.float32)
    sigma_annual = torch.tensor(covariance.values, dtype=torch.float32)
    w = torch.tensor(weights.reindex(tickers).values, dtype=torch.float32)

    sigma_daily = sigma_annual / TRADING_DAYS_PER_YEAR
    drift_daily = (mu_annual - 0.5 * torch.diag(sigma_annual)) / TRADING_DAYS_PER_YEAR

    cholesky = torch.linalg.cholesky(sigma_daily)

    z = torch.randn(n_paths, horizon_days, len(tickers), generator=generator)
    correlated_shocks = z @ cholesky.T
    daily_log_returns = drift_daily + correlated_shocks

    cumulative_log_returns = torch.cumsum(daily_log_returns, dim=1)
    asset_relative_paths = torch.exp(cumulative_log_returns)

    portfolio_path = asset_relative_paths.to(device) @ w.to(device)
    return portfolio_path[:, -1].cpu().numpy()


def simulate_heston(
    mu: float,
    portfolio_returns: pd.Series,
    n_paths: int = N_PATHS,
    horizon_days: int = HORIZON_DAYS,
    seed: int = 42,
) -> np.ndarray:
    """Heston simulation of the portfolio as a single aggregate process.

    `mu` is passed in explicitly — the same Black-Litterman drift GBM uses —
    so this differs from simulate_gbm only in how volatility behaves, not
    in the average outcome. Only theta, v0, and vol-of-vol come from
    historical data. Uses a full-truncation Euler scheme so variance can
    never go negative, the standard fix for naive Heston discretization.
    """
    device = _get_device()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    theta = portfolio_returns.var() * TRADING_DAYS_PER_YEAR
    v0 = portfolio_returns.tail(RECENT_WINDOW).var() * TRADING_DAYS_PER_YEAR
    vol_of_vol = portfolio_returns.rolling(RECENT_WINDOW).var().dropna().diff().std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol_of_vol = max(vol_of_vol, 1e-4)

    s = torch.ones(n_paths, device=device)
    v = torch.full((n_paths,), float(v0), device=device)

    for _ in range(horizon_days):
        z1 = torch.randn(n_paths, generator=generator).to(device)
        z2 = torch.randn(n_paths, generator=generator).to(device)
        dw_s = z1 * np.sqrt(dt)
        dw_v = (HESTON_RHO * z1 + np.sqrt(1 - HESTON_RHO**2) * z2) * np.sqrt(dt)

        v_pos = torch.clamp(v, min=0.0)
        v = v + HESTON_KAPPA * (theta - v_pos) * dt + vol_of_vol * torch.sqrt(v_pos) * dw_v
        s = s * torch.exp((mu - 0.5 * v_pos) * dt + torch.sqrt(v_pos) * dw_s)

    return s.cpu().numpy()


def _risk_metrics(terminal_values: np.ndarray, confidence: float = 0.95) -> dict:
    losses = 1.0 - terminal_values
    var = np.percentile(losses, confidence * 100)
    cvar = losses[losses >= var].mean()
    return {
        "mean_terminal": float(terminal_values.mean()),
        "std_terminal": float(terminal_values.std()),
        f"var_{int(confidence * 100)}": float(var),
        f"cvar_{int(confidence * 100)}": float(cvar),
    }


def simulate(
    expected_returns: pd.DataFrame | None = None,
    covariance: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    if expected_returns is None:
        expected_returns = pd.read_parquet(SILVER_DIR / "expected_returns.parquet")
    if covariance is None:
        covariance = pd.read_parquet(SILVER_DIR / "covariance.parquet")
    if weights is None:
        weights = pd.read_parquet(GOLD_DIR / "weights.parquet")
    if returns is None:
        returns = pd.read_parquet(SILVER_DIR / "returns.parquet")

    w = weights["max_sharpe"]
    mu = expected_returns["black_litterman"]

    gbm_terminal = simulate_gbm(mu, covariance, w)

    portfolio_returns = returns @ w.reindex(returns.columns)
    # Same drift GBM used, aggregated to portfolio level — keeps the
    # comparison isolated to volatility behavior only.
    portfolio_mu = float((mu.reindex(w.index) * w).sum())
    heston_terminal = simulate_heston(portfolio_mu, portfolio_returns)

    terminal_values = pd.DataFrame({"gbm": gbm_terminal, "heston": heston_terminal})
    risk_metrics = pd.DataFrame(
        {
            "gbm": _risk_metrics(gbm_terminal),
            "heston": _risk_metrics(heston_terminal),
        }
    ).T

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    terminal_path = GOLD_DIR / "simulation_terminal_values.parquet"
    metrics_path = GOLD_DIR / "risk_metrics.parquet"
    terminal_values.to_parquet(terminal_path)
    risk_metrics.to_parquet(metrics_path)
    return terminal_path, metrics_path


if __name__ == "__main__":
    device = _get_device()
    print(f"Using device: {device}")
    t_path, m_path = simulate()
    print(f"Wrote terminal values to {t_path}")
    print(f"Wrote risk metrics to {m_path}")
