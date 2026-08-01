"""
Tests for src/optimization.py — including the real convergence test that
caught SLSQP silently stalling at the equal-weight starting point.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization import max_sharpe_weights, risk_parity_weights


def test_risk_parity_weights_sum_to_one(sample_covariance):
    weights = risk_parity_weights(sample_covariance)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_risk_parity_weights_are_long_only(sample_covariance):
    weights = risk_parity_weights(sample_covariance)
    assert (weights >= 0).all()
    assert (weights <= 1).all()


def test_risk_parity_equalizes_risk_contribution(sample_covariance):
    """The real test. Equal weights alone don't prove correctness — this
    is what actually caught the bug: SLSQP reported success=True while
    never moving from its 1/n starting guess. Each asset's real
    contribution to portfolio variance must be equal, not just its weight."""
    weights = risk_parity_weights(sample_covariance)
    w = weights.values
    sigma = sample_covariance.values

    portfolio_variance = w @ sigma @ w
    contribution = w * (sigma @ w)
    contribution_share = contribution / portfolio_variance

    n = len(w)
    assert contribution_share == pytest.approx(np.full(n, 1.0 / n), abs=1e-3)


def test_risk_parity_favors_the_calmer_asset(sample_covariance):
    """Asset A has the lowest variance. A correct solution needs MORE
    dollars of a calm asset to reach an equal risk contribution, so its
    weight should come out higher than B's or C's."""
    weights = risk_parity_weights(sample_covariance)
    assert weights["A"] > weights["B"]
    assert weights["A"] > weights["C"]


def test_max_sharpe_weights_sum_to_one(sample_covariance):
    mu = pd.Series({"A": 0.08, "B": 0.12, "C": 0.10})
    weights = max_sharpe_weights(mu, sample_covariance, risk_free_rate=0.03)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_max_sharpe_weights_are_long_only(sample_covariance):
    mu = pd.Series({"A": 0.08, "B": 0.12, "C": 0.10})
    weights = max_sharpe_weights(mu, sample_covariance, risk_free_rate=0.03)
    assert (weights >= 0).all()
    assert (weights <= 1).all()
