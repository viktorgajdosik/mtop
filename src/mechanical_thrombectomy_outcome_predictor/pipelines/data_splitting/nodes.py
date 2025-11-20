from __future__ import annotations

from typing import Any
import pandas as pd


def create_temporal_split(df: pd.DataFrame, train_end_year: int = 2022) -> pd.DataFrame:
    """
    Add a 'split' column based on onset_date year.

    - Train: onset_year <= train_end_year
    - Test:  onset_year > train_end_year

    Assumes 'onset_date' exists and is parseable as datetime.
    Rows with missing/unparseable onset_date are assigned to 'train'
    (conservative choice to avoid future leakage).
    """
    df = df.copy()

    # Parse onset_date safely
    onset_dt = pd.to_datetime(df["onset_date"], errors="coerce")
    onset_year = onset_dt.dt.year

    df["onset_year"] = onset_year

    # Default split: train
    df["split"] = "train"
    mask_test = onset_year > train_end_year
    df.loc[mask_test, "split"] = "test"

    # Make split a categorical for clarity
    df["split"] = df["split"].astype("category")

    return df
