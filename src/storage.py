"""
Shared parquet read/write helpers.

Used by output.py and pipeline.py below. The six earlier modules
(extractor through heston) keep their own inline parquet calls — a
deliberate choice to avoid touching already-verified, working code for a
refactor with no functional benefit today. They can adopt this later.
"""

from pathlib import Path

import pandas as pd


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No data at {path}. Has the pipeline run yet?")
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path
