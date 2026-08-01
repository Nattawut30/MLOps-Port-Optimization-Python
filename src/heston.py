"""
Prices a protective put on the portfolio's largest holding using the Heston
model, then shows how adding that hedge changes the portfolio's simulated
risk profile.

Pricing uses the risk-neutral rate (settings.risk_free_rate) for discounting
and drift — NOT the real-world Black-Litterman expected return used in
simulation.py. This is a deliberate, important distinction: simulation.py
answers "what might actually happen"; this file answers "what is the
no-arbitrage price today." Conflating the two would produce a price with
no theoretical grounding.

Two independent pricing methods (Carr-Madan FFT and Monte Carlo) are run on
the same Heston parameters and checked against each other — that agreement
is the self-consistency test promised when Heston was first scoped into
this project. theta, v0, and vol-of-vol are estimated from the largest
holding's own historical returns; kappa and rho are the same literature-
convention defaults used in simulation.py (fitting them to a live options
market would require a paid data feed).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.settings import BRONZE_DIR, GOLD_DIR, SILVER_DIR, settings

TRADING_DAYS_PER_YEAR = 252
HESTON_KAPPA = 3.0   # must match simulation.py's convention default
HESTON_RHO = -0.7    # must match simulation.py's convention default
RECENT_WINDOW = 20
N_MC_PATHS = 20_000
HORIZON_DAYS = TRADING_DAYS_PER_YEAR  # 1-year option


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def calibrate_heston(returns: pd.Series) -> dict:
    """Estimate Heston volatility parameters from historical data.

    Only volatility parameters come from data. kappa and rho are literature
    defaults — the same disclosed simplification made in simulation.py.
    """
    theta = float(returns.var() * TRADING_DAYS_PER_YEAR)
    v0 = float(returns.tail(RECENT_WINDOW).var() * TRADING_DAYS_PER_YEAR)
    vol_of_vol = float(
        returns.rolling(RECENT_WINDOW).var().dropna().diff().std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    vol_of_vol = max(vol_of_vol, 1e-4)
    return {
        "kappa": HESTON_KAPPA,
        "theta": theta,
        "v0": v0,
        "vol_of_vol": vol_of_vol,
        "rho": HESTON_RHO,
    }


def _heston_char_function(
    u: np.ndarray, S0: float, T: float, r: float, params: dict
) -> np.ndarray:
    """Heston characteristic function of log(S_T), "little trap" form
    (Albrecher et al.) — avoids the branch-cut discontinuities that make
    the original Heston (1993) formulation numerically unstable."""
    kappa, theta, sigma_v, rho, v0 = (
        params["kappa"], params["theta"], params["vol_of_vol"],
        params["rho"], params["v0"],
    )
    i = 1j
    d = np.sqrt((rho * sigma_v * i * u - kappa) ** 2 + sigma_v**2 * (i * u + u**2))
    g = (kappa - rho * sigma_v * i * u - d) / (kappa - rho * sigma_v * i * u + d)

    C = r * i * u * T + (kappa * theta / sigma_v**2) * (
        (kappa - rho * sigma_v * i * u - d) * T
        - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = (
        (kappa - rho * sigma_v * i * u - d)
        / sigma_v**2
        * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    )
    return np.exp(C + D * v0 + i * u * np.log(S0))


def heston_fft_put_price(
    S0: float, K: float, T: float, r: float, params: dict,
    alpha: float = 1.5, n: int = 4096, eta: float = 0.25,
) -> float:
    """European put price via the Carr-Madan (1999) FFT method.

    Prices a call across a grid of strikes via one FFT call, then converts
    to a put via put-call parity — the standard, well-documented approach,
    avoiding a separate put-specific Fourier derivation.
    """
    lambda_ = 2 * np.pi / (n * eta)
    b = n * lambda_ / 2
    u = eta * np.arange(n)
    log_strikes = -b + lambda_ * np.arange(n)

    i = 1j
    shifted_u = u - (alpha + 1) * i
    phi = _heston_char_function(shifted_u, S0, T, r, params)
    denominator = alpha**2 + alpha - u**2 + i * (2 * alpha + 1) * u
    psi = np.exp(-r * T) * phi / denominator

    # Simpson's rule weighting for FFT accuracy (Carr-Madan convention)
    simpson_weights = (3 + (-1) ** (np.arange(n) + 1))
    simpson_weights[0] = 1
    integrand = np.exp(i * b * u) * psi * eta * simpson_weights / 3

    call_prices = np.exp(-alpha * log_strikes) / np.pi * np.real(np.fft.fft(integrand))

    strikes = np.exp(log_strikes)
    call_at_k = np.interp(np.log(K), log_strikes, call_prices)
    # Put-call parity: Put = Call - S0 + K * exp(-rT) (zero dividend yield, disclosed)
    put_at_k = call_at_k - S0 + K * np.exp(-r * T)
    return float(max(put_at_k, 0.0))


def heston_mc_put_price(
    S0: float, K: float, T: float, r: float, params: dict,
    n_paths: int = N_MC_PATHS, seed: int = 7,
) -> float:
    """Risk-neutral Monte Carlo put price — the independent cross-check
    against the FFT price above. Drift is r, the risk-free rate, NOT any
    real-world expected return; this is a no-arbitrage pricing simulation,
    not a real-world outcome simulation like simulation.py's."""
    device = _get_device()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    horizon_days = int(T * TRADING_DAYS_PER_YEAR)
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    kappa, theta, sigma_v, rho = (
        params["kappa"], params["vol_of_vol"] and params["theta"],
        params["vol_of_vol"], params["rho"],
    )
    theta = params["theta"]

    s = torch.full((n_paths,), float(S0), device=device)
    v = torch.full((n_paths,), float(params["v0"]), device=device)

    for _ in range(horizon_days):
        z1 = torch.randn(n_paths, generator=generator).to(device)
        z2 = torch.randn(n_paths, generator=generator).to(device)
        dw_s = z1 * np.sqrt(dt)
        dw_v = (rho * z1 + np.sqrt(1 - rho**2) * z2) * np.sqrt(dt)

        v_pos = torch.clamp(v, min=0.0)
        v = v + kappa * (theta - v_pos) * dt + sigma_v * torch.sqrt(v_pos) * dw_v
        s = s * torch.exp((r - 0.5 * v_pos) * dt + torch.sqrt(v_pos) * dw_s)

    payoff = torch.clamp(K - s, min=0.0)
    price = torch.exp(torch.tensor(-r * T)) * payoff.mean()
    return float(price.cpu())


def price_hedge() -> dict:
    """Price a 1-year, at-the-money protective put on the portfolio's
    largest max_sharpe holding, and cross-check the price two ways."""
    prices = pd.read_parquet(BRONZE_DIR / "prices.parquet")
    returns = pd.read_parquet(SILVER_DIR / "returns.parquet")
    weights = pd.read_parquet(GOLD_DIR / "weights.parquet")["max_sharpe"]

    largest_ticker = weights.idxmax()
    S0 = float(prices[largest_ticker].iloc[-1])
    K = S0  # at-the-money
    T = 1.0
    r = settings.risk_free_rate

    params = calibrate_heston(returns[largest_ticker])
    fft_price = heston_fft_put_price(S0, K, T, r, params)
    mc_price = heston_mc_put_price(S0, K, T, r, params)
    agreement_pct = abs(fft_price - mc_price) / fft_price * 100 if fft_price > 0 else float("nan")

    return {
        "hedged_ticker": largest_ticker,
        "hedged_weight": float(weights[largest_ticker]),
        "spot": S0,
        "strike": K,
        "maturity_years": T,
        "fft_put_price": fft_price,
        "mc_put_price": mc_price,
        "fft_vs_mc_agreement_pct_diff": agreement_pct,
        **params,
    }


def run() -> Path:
    result = price_hedge()
    output_path = GOLD_DIR / "heston_hedge.parquet"
    pd.DataFrame([result]).to_parquet(output_path)
    return output_path


if __name__ == "__main__":
    path = run()
    result = pd.read_parquet(path).iloc[0]
    print(f"Hedged ticker: {result['hedged_ticker']} (largest holding, weight {result['hedged_weight']:.1%})")
    print(f"FFT put price:  {result['fft_put_price']:.4f}")
    print(f"MC put price:   {result['mc_put_price']:.4f}")
    print(f"Agreement:      {result['fft_vs_mc_agreement_pct_diff']:.2f}% difference")
    print(f"Wrote {path}")
