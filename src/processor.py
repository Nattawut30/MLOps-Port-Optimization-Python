"""
Turns bronze-layer raw prices into silver-layer returns and a shrinkage
covariance matrix.

Log returns are used because they're additive over time and closer to
normally distributed, which later steps (Black-Litterman, Heston
calibration) assume. The covariance matrix uses Ledoit-Wolf shrinkage
rather than the raw sample covariance, since a few years of daily data
for a handful of assets produces a raw covariance matrix that is
poorly conditioned and unstable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.settings import BRONZE_DIR, SILVER_DIR

TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 30


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Only {len(returns)} return observations available; need at "
            f"least {MIN_OBSERVATIONS} for a stable covariance estimate."
        )
    return returns


def _annualized_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    shrinkage_estimator = LedoitWolf().fit(returns.values)
    daily_covariance = pd.DataFrame(
        shrinkage_estimator.covariance_,
        index=returns.columns,
        columns=returns.columns,
    )
    return daily_covariance * TRADING_DAYS_PER_YEAR


def process(prices: pd.DataFrame | None = None) -> tuple[Path, Path]:
    """Read bronze prices, write silver returns and covariance.

    Returns (returns_path, covariance_path). Accepts `prices` directly
    so tests can inject a small fake DataFrame instead of touching disk.
    """
    if prices is None:
        bronze_path = BRONZE_DIR / "prices.parquet"
        if not bronze_path.exists():
            raise FileNotFoundError(
                f"No bronze data at {bronze_path}. Run extractor.py first."
            )
        prices = pd.read_parquet(bronze_path)

    returns = _log_returns(prices)
    covariance = _annualized_covariance(returns)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    returns_path = SILVER_DIR / "returns.parquet"
    covariance_path = SILVER_DIR / "covariance.parquet"
    returns.to_parquet(returns_path)
    covariance.to_parquet(covariance_path)
    return returns_path, covariance_path


if __name__ == "__main__":
    r_path, c_path = process()
    print(f"Wrote silver returns to {r_path}")
    print(f"Wrote silver covariance to {c_path}")
