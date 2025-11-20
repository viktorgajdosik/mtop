from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def _compute_smd(x_train: pd.Series, x_test: pd.Series) -> float:
    """
    Standardized mean difference.

    - Numeric: (mean1 - mean2) / pooled SD
    - Categorical: worst-case SMD across levels (based on proportions).
    """
    x_train = x_train.dropna()
    x_test = x_test.dropna()

    # If no data, SMD is undefined -> 0.0
    if x_train.empty and x_test.empty:
        return 0.0

    if x_train.dtype.kind in "bifc" and x_test.dtype.kind in "bifc":
        m1, m2 = x_train.mean(), x_test.mean()
        s1, s2 = x_train.std(ddof=1), x_test.std(ddof=1)
        denom = np.sqrt((s1**2 + s2**2) / 2) if (s1 > 0 or s2 > 0) else 0.0
        return float((m1 - m2) / denom) if denom != 0 else 0.0

    # Treat as categorical
    levels = sorted(
        list(
            set(x_train.dropna().unique()) | set(x_test.dropna().unique())
        )
    )
    if not levels:
        return 0.0

    smd_levels = []
    for lev in levels:
        p1 = (x_train == lev).mean() if len(x_train) else 0.0
        p2 = (x_test == lev).mean() if len(x_test) else 0.0
        denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2) if (p1 not in (0, 1) or p2 not in (0, 1)) else 0.0
        smd = (p1 - p2) / denom if denom != 0 else 0.0
        smd_levels.append(smd)

    # Return the level with the largest absolute imbalance
    return float(max(smd_levels, key=lambda x: abs(x)))


def make_baseline_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build baseline characteristics for Train vs Test:

    - Table 1: per variable:
        * numeric: mean, SD in each split + SMD
        * categorical: counts (per level) stored as dict (stringified in CSV)
    - SMD table: variable + SMD, sorted by |SMD| desc.
    """
    if "split" not in df.columns:
        raise ValueError("Expected a 'split' column with values 'train'/'test' in the input DataFrame.")

    df = df.copy()
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    rows = []
    smd_rows = []

    for col in df.columns:
        if col in {"split", "onset_year", "patient_id"}:
            continue

        x_train = train[col]
        x_test = test[col]
        smd = _compute_smd(x_train, x_test)

        if x_train.dtype.kind in "bifc" and x_test.dtype.kind in "bifc":
            row = {
                "variable": col,
                "type": "numeric",
                "train_n": int(x_train.notna().sum()),
                "train_mean": float(x_train.mean()) if x_train.notna().any() else None,
                "train_sd": float(x_train.std(ddof=1)) if x_train.notna().any() else None,
                "test_n": int(x_test.notna().sum()),
                "test_mean": float(x_test.mean()) if x_test.notna().any() else None,
                "test_sd": float(x_test.std(ddof=1)) if x_test.notna().any() else None,
                "smd": smd,
            }
        else:
            train_counts = x_train.value_counts(dropna=False)
            test_counts = x_test.value_counts(dropna=False)

            # Convert index to str for CSV friendliness
            train_counts_dict = {str(k): int(v) for k, v in train_counts.items()}
            test_counts_dict = {str(k): int(v) for k, v in test_counts.items()}

            row = {
                "variable": col,
                "type": "categorical",
                "train_counts": train_counts_dict,
                "test_counts": test_counts_dict,
                "smd": smd,
            }

        rows.append(row)
        smd_rows.append({"variable": col, "smd": smd})

    table1 = pd.DataFrame(rows)
    smd_table = pd.DataFrame(smd_rows).sort_values(
        "smd", key=lambda s: s.abs(), ascending=False
    )

    return table1, smd_table
