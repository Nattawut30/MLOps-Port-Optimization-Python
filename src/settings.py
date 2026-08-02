"""
Centralized, validated configuration for the project.

Every other module imports `settings` from here rather than reading
environment variables directly. This is what lets a bad value (a
malformed date, an empty ticker list, a risk-free rate typed as 4.5
instead of 0.045) fail loudly at startup instead of silently, deep
inside a scheduled pipeline run.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
BRONZE_DIR = DATA_DIR / "01_bronze"
SILVER_DIR = DATA_DIR / "02_silver"
GOLD_DIR = DATA_DIR / "03_gold"


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tickers: list[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    )
    start_date: date = date(2020, 1, 1)
    end_date: date = datetime.now(tz=timezone.utc).date()
    risk_free_rate: float = 0.045
    mlflow_tracking_uri: str = f"sqlite:///{ROOT_DIR / 'mlruns' / 'mlflow.db'}"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Free key from alphavantage.co — required, no default on purpose,
    # so a missing key fails loudly at startup instead of deep in extractor.py.
    alpha_vantage_api_key: str

    @field_validator("tickers", mode="before")
    @classmethod
    def _split_tickers(cls, value: object) -> object:
        if isinstance(value, str):
            tickers = [t.strip().upper() for t in value.split(",") if t.strip()]
            if not tickers:
                raise ValueError("TICKERS must contain at least one ticker.")
            return tickers
        return value

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, end_date: date, info) -> date:
        start_date = info.data.get("start_date")
        if start_date and end_date <= start_date:
            raise ValueError("END_DATE must be after START_DATE.")
        return end_date

    @field_validator("risk_free_rate")
    @classmethod
    def _sane_rate(cls, rate: float) -> float:
        if not -0.05 <= rate <= 0.25:
            raise ValueError(
                f"RISK_FREE_RATE={rate} looks implausible; expected a small "
                "decimal like 0.045 for 4.5%, not a percentage or a typo."
            )
        return rate

    @field_validator("alpha_vantage_api_key")
    @classmethod
    def _real_key(cls, key: str) -> str:
        if not key or key.strip().lower() in {"your_key_here", "changeme", ""}:
            raise ValueError(
                "ALPHA_VANTAGE_API_KEY is missing or a placeholder. "
                "Get a free key at alphavantage.co and set it in .env."
            )
        return key


settings = Settings()
