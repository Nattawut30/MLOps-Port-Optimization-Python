"""
Tests for src/processor.py — log returns and Ledoit-Wolf covariance.
"""

import numpy as np
import pandas as pd
import pytest

from src.processor import MIN_OBSERVATIONS, _annualized_covariance, _log_returns


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Deterministic, realistic-scale price series — enough rows to clear
    MIN_OBSERVATIONS after the first-row diff is dropped."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    prices_a = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 60)))
    prices_b = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 60)))
    return pd.DataFrame({"A": prices_a, "B": prices_b}, index=dates)


def test_log_returns_drops_first_row(sample_prices):
    """The first row has no prior price to compare against, so it must
    be dropped, not filled with a fabricated value."""
    returns = _log_returns(sample_prices)
    assert len(returns) == len(sample_prices) - 1


def test_log_returns_raises_below_min_observations():
    """Confirms the safety guard actually fires on too little data.
    Built directly from MIN_OBSERVATIONS so this test stays correct
    if that threshold ever changes."""
    dates = pd.date_range("2024-01-01", periods=MIN_OBSERVATIONS)
    tiny_prices = pd.DataFrame({"A": range(100, 100 + MIN_OBSERVATIONS)}, index=dates)
    with pytest.raises(ValueError, match="at least"):
        _log_returns(tiny_prices)


def test_covariance_is_symmetric(sample_prices):
    """A real covariance matrix must be symmetric — the same property
    checked by hand on the first real run (AAPL-vs-MSFT had to equal
    MSFT-vs-AAPL). Shrinkage or annualizing breaking this would be a
    serious, silent bug."""
    returns = _log_returns(sample_prices)
    covariance = _annualized_covariance(returns)
    np.testing.assert_allclose(covariance.values, covariance.values.T, atol=1e-10)


def test_covariance_diagonal_is_positive(sample_prices):
    """Diagonal entries are variances — must be strictly positive for
    any real asset with actual price movement."""
    returns = _log_returns(sample_prices)
    covariance = _annualized_covariance(returns)
    assert (np.diag(covariance.values) > 0).all()


def test_covariance_index_matches_tickers(sample_prices):
    returns = _log_returns(sample_prices)
    covariance = _annualized_covariance(returns)
    assert list(covariance.index) == list(returns.columns)
    assert list(covariance.columns) == list(returns.columns)
