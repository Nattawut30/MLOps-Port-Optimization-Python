"""
Pulls raw daily prices directly from Stooq's CSV endpoint and writes them
to the bronze layer.

Deliberately isolated behind this one file: if the data source ever needs
to change again, only this file changes. Nothing downstream imports Stooq
directly. This talks to Stooq's own CSV download endpoint rather than a
third-party wrapper, since that wrapper's Stooq support has a long history
of breaking independently of anything in this project.
"""

import time
from io import StringIO
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

from src.settings import BRONZE_DIR, settings

STOOQ_CSV_URL = "https://stooq.com/q/d/l/"


class PriceDataSource(Protocol):
    """Anything that can fetch daily close prices for a list of tickers."""

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...


class StooqDataSource:
    """Adapter around Stooq's public CSV endpoint.

    Retries on failure, since any external HTTP call can have a bad
    moment — better to retry once than fail an entire scheduled run.
    """

    def __init__(self, max_retries: int = 3, retry_delay_seconds: float = 2.0) -> None:
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        frames: dict[str, pd.Series] = {}
        for ticker in tickers:
            frames[ticker] = self._fetch_one(ticker, start, end)
        prices = pd.DataFrame(frames).sort_index()
        prices.index.name = "date"
        return prices

    def _fetch_one(self, ticker: str, start: str, end: str) -> pd.Series:
        symbol = f"{ticker.lower()}.us"
        params = {
            "s": symbol,
            "d1": start.replace("-", ""),
            "d2": end.replace("-", ""),
            "i": "d",  # daily
        }
        headers = {"User-Agent": "Mozilla/5.0"}

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    STOOQ_CSV_URL, params=params, headers=headers, timeout=10
                )
                response.raise_for_status()
                data = pd.read_csv(
                    StringIO(response.text), parse_dates=["Date"], index_col="Date"
                )
                if data.empty or "Close" not in data.columns:
                    raise ValueError(f"Stooq returned no usable data for {ticker}.")
                return data["Close"].sort_index()
            except Exception as error:  # network hiccups, throttling, bad ticker
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"Failed to fetch {ticker} from Stooq after {self.max_retries} attempts."
        ) from last_error


def extract(source: PriceDataSource | None = None) -> Path:
    """Fetch prices for the configured tickers and write them to the bronze layer.

    Returns the path to the written parquet file.
    """
    source = source or StooqDataSource()
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
