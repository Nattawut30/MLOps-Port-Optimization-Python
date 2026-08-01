"""
Pulls raw daily prices from Alpha Vantage and writes them to the bronze layer.

Deliberately isolated behind this one file: this is the second data source
this project has used (Stooq's page started requiring a browser JavaScript
check, which a script can't satisfy). Alpha Vantage is a real, documented
API meant for programmatic use, not a scraped page — a different category
of source, chosen specifically to avoid repeating that failure mode.
"""

import time
from io import StringIO
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

from src.settings import BRONZE_DIR, settings

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class PriceDataSource(Protocol):
    """Anything that can fetch daily close prices for a list of tickers."""

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...


class AlphaVantageDataSource:
    """Adapter around the Alpha Vantage TIME_SERIES_DAILY endpoint."""

    def __init__(self, max_retries: int = 3, retry_delay_seconds: float = 5.0) -> None:
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        frames: dict[str, pd.Series] = {}
        for i, ticker in enumerate(tickers):
            if i > 0:
                # Free-tier APIs like this throttle requests per minute;
                # a short pause between tickers avoids tripping that limit.
                time.sleep(13)
            frames[ticker] = self._fetch_one(ticker, start, end)
        prices = pd.DataFrame(frames).sort_index()
        prices = prices.loc[start:end]
        prices.index.name = "date"
        return prices

    def _fetch_one(self, ticker: str, start: str, end: str) -> pd.Series:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": settings.alpha_vantage_api_key,
            "datatype": "csv",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=15)
                response.raise_for_status()
                text = response.text

                first_line = text.splitlines()[0] if text.strip() else ""
                if "timestamp" not in first_line.lower():
                    # Alpha Vantage returns HTTP 200 even on errors (bad key,
                    # rate limit, bad symbol) — surface the real message.
                    raise ValueError(
                        f"Alpha Vantage did not return data for {ticker!r}. "
                        f"Response started with: {text[:200]!r}"
                    )

                data = pd.read_csv(
                    StringIO(text), parse_dates=["timestamp"], index_col="timestamp"
                )
                return data["close"].sort_index()
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"Failed to fetch {ticker} from Alpha Vantage after "
            f"{self.max_retries} attempts. Last error: {last_error}"
        ) from last_error


def extract(source: PriceDataSource | None = None) -> Path:
    """Fetch prices for the configured tickers and write them to the bronze layer."""
    source = source or AlphaVantageDataSource()
    prices = source.fetch(
        tickers=settings.tickers,
        start=settings.start_date.isoformat(),
        end=settings.end_date.isoformat(),
    )

    if prices.empty:
        raise ValueError("No price data returned; check tickers and date range.")

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BRONZE_DIR / "prices.parquet"
    prices.to_parquet(output_path)
    return output_path


if __name__ == "__main__":
    path = extract()
    print(f"Wrote bronze prices to {path}")
