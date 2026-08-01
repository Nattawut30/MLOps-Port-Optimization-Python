"""
Tests for src/estimation.py — including the zero-views correctness check
that was described as "true by construction" when this file was built.
This test is what actually proves that, rather than just asserting it.
"""

import numpy as np
import pandas as pd
import pytest

from src.estimation import black_litterman_returns, historical_mean_returns


@pytest.fixture
def sample_covariance() -> pd.DataFrame:
    tickers = ["A", "B", "C"]
    data = [
        [0.04, 0.01, 0.01],
        [0.01, 0.09, 0.02],
        [0.01, 0.02, 0.09],
    ]
    return pd.DataFrame(data, index=tickers, columns=tickers)


def test_historical_mean_annualizes_correctly():
    """A constant daily return should annualize to daily_return * 252 —
    a direct, exact check on the annualization arithmetic itself."""
    returns = pd.DataFrame({"A": [0.001] * 100})
    result = historical_mean_returns(returns)
    assert result["A"] == pytest.approx(0.001 * 252, rel=1e-6)


def test_zero_views_returns_equilibrium_exactly(sample_covariance):
    """The real test. With no views, the formula has nothing to blend
    in and must return the market-implied equilibrium directly — proving
    the "true by construction" claim made when this file was written,
    rather than just asserting it in a docstring."""
    risk_aversion = 2.5
    equal_weights = pd.Series(1 / 3, index=sample_covariance.columns)
    expected_equilibrium = risk_aversion * sample_covariance.values @ equal_weights.values

    result = black_litterman_returns(sample_covariance, risk_aversion=risk_aversion)

    np.testing.assert_allclose(result.values, expected_equilibrium, atol=1e-10)


def test_a_view_moves_that_asset_toward_the_stated_value(sample_covariance):
    """Sanity check that supplying a view actually changes the output —
    confirms the blending math isn't silently ignoring the views
    argument. Ticker A's posterior should move meaningfully toward the
    strong 20% view stated for it."""
    no_view_result = black_litterman_returns(sample_covariance)
    with_view_result = black_litterman_returns(sample_covariance, views={"A": 0.20})

    assert with_view_result["A"] != pytest.approx(no_view_result["A"])
    # Should move toward the view, not away from it.
    assert abs(with_view_result["A"] - 0.20) < abs(no_view_result["A"] - 0.20)


def test_returns_series_indexed_by_all_tickers(sample_covariance):
    result = black_litterman_returns(sample_covariance)
    assert list(result.index) == list(sample_covariance.columns)
