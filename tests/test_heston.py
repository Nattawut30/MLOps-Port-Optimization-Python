"""
Tests for src/heston.py — including the FFT-vs-Monte-Carlo self-consistency
check that was originally verified by hand (1.16% agreement) and is now
permanent.
"""

import numpy as np
import pandas as pd
import pytest

from src.heston import calibrate_heston, heston_fft_put_price, heston_mc_put_price

N_MC_PATHS = 20_000  # smaller than production default for fast test runs


@pytest.fixture
def sample_returns() -> pd.Series:
    """Deterministic, realistic-scale daily return series."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0004, 0.015, 300))


@pytest.fixture
def sample_params(sample_returns) -> dict:
    return calibrate_heston(sample_returns)


def test_calibration_produces_nonnegative_variance_params(sample_params):
    """theta and v0 are variances — a calibration bug that produces a
    negative one would be silently invalid input to the pricer."""
    assert sample_params["theta"] >= 0
    assert sample_params["v0"] >= 0
    assert sample_params["vol_of_vol"] > 0


def test_fft_and_mc_prices_agree(sample_params):
    """The real test — the self-consistency check originally verified by
    hand. Two structurally independent pricing methods, same parameters,
    should land close together. This is what proves the Heston
    implementation is mathematically sound, not a coincidence."""
    S0, K, T, r = 100.0, 100.0, 1.0, 0.045

    fft_price = heston_fft_put_price(S0, K, T, r, sample_params)
    mc_price = heston_mc_put_price(S0, K, T, r, sample_params, n_paths=N_MC_PATHS)

    percent_diff = abs(fft_price - mc_price) / fft_price * 100
    # 5% is a looser bound than the ~1.16% seen by hand, to keep this
    # test stable across Monte Carlo's inherent run-to-run randomness.
    assert percent_diff < 5.0


def test_put_price_is_nonnegative(sample_params):
    """An option can never have a negative price — a basic no-arbitrage
    sanity check independent of the Heston model itself."""
    fft_price = heston_fft_put_price(100.0, 100.0, 1.0, 0.045, sample_params)
    assert fft_price >= 0


def test_deep_out_of_the_money_put_is_cheap(sample_params):
    """A put struck far below the current spot price should be worth
    much less than an at-the-money put — a basic option-pricing sanity
    check that would catch a badly broken pricer even without a
    cross-check method."""
    atm_price = heston_fft_put_price(100.0, 100.0, 1.0, 0.045, sample_params)
    otm_price = heston_fft_put_price(100.0, 50.0, 1.0, 0.045, sample_params)
    assert otm_price < atm_price
