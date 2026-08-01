"""
Tests for src/simulation.py — including the shared-drift regression test.

An earlier version let simulate_heston recompute its own drift from noisy
historical data, which meant GBM and Heston were centered on different
average outcomes — an invalid comparison, since a higher center can mask
a genuinely fatter tail. test_gbm_and_heston_share_the_same_drift exists
specifically to catch that bug if it's ever reintroduced.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation import simulate_gbm, simulate_heston

N_PATHS = 5_000  # smaller than production for fast test runs
SEED = 42


@pytest.fixture
def two_asset_setup():
    tickers = ["A", "B"]
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.05]], index=tickers, columns=tickers
    )
    expected_returns = pd.Series({"A": 0.08, "B": 0.10})
    weights = pd.Series({"A": 0.5, "B": 0.5})
    return covariance, expected_returns, weights


def test_gbm_matches_analytic_expected_value(two_asset_setup):
    """GBM has a known closed-form expected terminal value: exp(mu * T)
    for each asset (T=1 year here). With enough paths, the simulated
    mean should land close to the analytic one — a direct check on the
    simulator itself, not a borrowed model."""
    covariance, expected_returns, weights = two_asset_setup

    terminal_values = simulate_gbm(
        expected_returns, covariance, weights, n_paths=20_000, seed=SEED
    )

    analytic_expected = float((weights * np.exp(expected_returns)).sum())
    simulated_mean = terminal_values.mean()

    # Monte Carlo noise means this needs a tolerance, not exact equality —
    # 5% is generous enough to be stable across seeds, tight enough to
    # catch a real drift-calculation bug.
    assert simulated_mean == pytest.approx(analytic_expected, rel=0.05)


def test_gbm_and_heston_share_the_same_drift(two_asset_setup):
    """Regression test for the shared-drift bug: GBM and Heston must be
    centered on the same average outcome. If Heston is ever changed to
    recompute its own drift from historical data again, this should fail."""
    covariance, expected_returns, weights = two_asset_setup

    gbm_terminal = simulate_gbm(expected_returns, covariance, weights, n_paths=N_PATHS, seed=SEED)

    portfolio_mu = float((expected_returns * weights).sum())
    portfolio_returns = pd.Series(np.random.default_rng(SEED).normal(0.0004, 0.01, 300))
    heston_terminal = simulate_heston(portfolio_mu, portfolio_returns, n_paths=N_PATHS, seed=SEED)

    # Same drift in, so means should land close — not identical, since
    # volatility behaves differently, but not wildly apart either.
    assert gbm_terminal.mean() == pytest.approx(heston_terminal.mean(), rel=0.15)


def test_heston_variance_never_goes_negative(two_asset_setup):
    """The full-truncation Euler scheme exists specifically to prevent
    negative variance, a well-known instability in naive Heston
    discretization. Terminal portfolio values should all stay positive —
    a negative or zero value would indicate the variance process broke."""
    portfolio_returns = pd.Series(np.random.default_rng(SEED).normal(0.0004, 0.01, 300))
    terminal_values = simulate_heston(0.08, portfolio_returns, n_paths=N_PATHS, seed=SEED)
    assert (terminal_values > 0).all()


def test_simulations_are_reproducible_with_same_seed(two_asset_setup):
    """Same seed should give identical results — reproducibility matters
    for an audit trail, one of the original system-design goals."""
    covariance, expected_returns, weights = two_asset_setup
    first_run = simulate_gbm(expected_returns, covariance, weights, n_paths=1000, seed=1)
    second_run = simulate_gbm(expected_returns, covariance, weights, n_paths=1000, seed=1)
    np.testing.assert_array_equal(first_run, second_run)
