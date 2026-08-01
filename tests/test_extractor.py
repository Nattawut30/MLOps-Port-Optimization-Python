"""
Tests for src/extractor.py — using a fake data source for extract() tests
(proving the Adapter pattern actually decouples the pipeline from the
network) and monkeypatched HTTP responses for AlphaVantageDataSource tests
(proving the diagnostic-error behavior that already caught two real
provider-format problems by hand).
"""

import pandas as pd
import pytest

from src.extractor import AlphaVantageDataSource, extract


class FakeDataSource:
    """A minimal fake implementing the PriceDataSource protocol — no
    network, no API key needed. This is the payoff of building extract()
    against an interface instead of a concrete class."""

    def __init__(self, prices: pd.DataFrame):
        self._prices = prices

    def fetch(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        return self._prices


@pytest.fixture
def fake_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {"A": range(100, 110), "B": range(200, 210)}, index=dates
    )


def test_extract_writes_parquet_with_fake_source(monkeypatch, tmp_path, fake_prices):
    monkeypatch.setattr("src.extractor.BRONZE_DIR", tmp_path)
    source = FakeDataSource(fake_prices)

    output_path = extract(source=source)

    assert output_path.exists()
    written = pd.read_parquet(output_path)
    assert list(written.columns) == ["A", "B"]
    assert len(written) == 10


def test_extract_raises_on_empty_data(monkeypatch, tmp_path):
    monkeypatch.setattr("src.extractor.BRONZE_DIR", tmp_path)
    source = FakeDataSource(pd.DataFrame())

    with pytest.raises(ValueError, match="No price data returned"):
        extract(source=source)


def test_alpha_vantage_surfaces_diagnostic_on_bad_response(monkeypatch):
    """Regression guard: the exact bug that hit twice by hand — Stooq's
    JS-block page, then Alpha Vantage's 'premium feature' rejection.
    Both looked like valid HTTP 200 responses but weren't real CSV data.
    This confirms a non-CSV response raises a clear, readable error
    instead of a cryptic pandas parsing failure three layers deep."""

    class FakeResponse:
        status_code = 200
        text = '{"Information": "This is a premium endpoint."}'

        def raise_for_status(self):
            pass

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("src.extractor.requests.get", fake_get)

    source = AlphaVantageDataSource(max_retries=1, retry_delay_seconds=0)
    with pytest.raises(RuntimeError, match="Failed to fetch"):
        source.fetch(["AAPL"], "2024-01-01", "2024-01-10")


def test_alpha_vantage_retries_before_giving_up(monkeypatch):
    """Confirms the retry loop actually retries the configured number of
    times before raising — not zero times, not infinitely."""
    call_count = 0

    class FakeResponse:
        status_code = 200
        text = "not real data"

        def raise_for_status(self):
            pass

    def fake_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return FakeResponse()

    monkeypatch.setattr("src.extractor.requests.get", fake_get)

    source = AlphaVantageDataSource(max_retries=3, retry_delay_seconds=0)
    with pytest.raises(RuntimeError):
        source.fetch(["AAPL"], "2024-01-01", "2024-01-10")

    assert call_count == 3
