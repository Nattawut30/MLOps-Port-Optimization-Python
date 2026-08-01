"""
Shared, deterministic test fixtures — no network calls, no real data.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_covariance() -> pd.DataFrame:
    """Small, deterministic, positive-definite covariance matrix.

    Asset A has clearly lower variance (0.04) than B and C (0.09 each).
    This asymmetry is deliberate: it's what lets test_optimization.py
    distinguish a genuinely converged risk-parity solution from the
    silently-stuck-at-equal-weight bug caught earlier by hand.
    """
    tickers = ["A", "B", "C"]
    data = [
        [0.04, 0.01, 0.01],
        [0.01, 0.09, 0.02],
        [0.01, 0.02, 0.09],
    ]
    return pd.DataFrame(data, index=tickers, columns=tickers)
