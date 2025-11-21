from __future__ import annotations
import re
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


RANDOM_STATE = 42  # reserved for future use


def _trim_and_fix_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize all string cells: strip spaces incl. non-breaking spaces
    def _clean_cell(x: Any) -> Any:
        if isinstance(x, str):
            x = x.replace("\u00A0", " ")  # nbsp -> space
            return x.strip()
        return x
    return df.map(_clean_cell)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip, collapse runs of spaces, lower, replace spaces/dashes with underscore
    cols = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    df.columns = cols
    return df


def _rename_columns(df: pd.DataFrame, column_map: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    original_cols = df.columns.tolist()
    # map works on ORIGINAL names; build reverse lookup by normalized original keys
    rename_dict = {}
    for k, v in column_map.items():
        rename_dict[k] = v

    # perform rename only for present keys
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

    # diagnostics
    expected = set(column_map.values())
    present = set(df.columns)
    missing_expected = sorted(list(expected - present))
    applied = {k: v for k, v in column_map.items() if k in original_cols}

    log = {
        "renamed_count": len(applied),
        "missing_expected_after_rename": missing_expected,
        "unmapped_original_columns": [c for c in original_cols if c not in column_map],
    }
    return df, log


def _apply_value_translations(df: pd.DataFrame, translate_map: Dict[str, str]) -> pd.DataFrame:
    if translate_map:
        df = df.replace(translate_map)
    return df


def _normalize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all generic missing markers to pandas NA.
    We keep clinical "No finding" as a real category and do NOT touch it.
    """
    # First, normalize Python None
    df = df.replace({None: pd.NA})

    # Then normalize common string tokens that should NOT be clinical categories
    # (case-insensitive: none, na, n/a, nan, ?, empty / whitespace)
    df = df.replace(
        to_replace=[
            r"^\s*$",          # empty / whitespace
            r"(?i)^none$",     # "none", "None", "NONE"
            r"(?i)^na$",       # "na", "NA"
            r"(?i)^n/?a$",     # "n/a", "N/A"
            r"(?i)^nan$",      # "nan", "NaN"
            r"^\?$",           # "?"
        ],
        value=pd.NA,
        regex=True,
    )

    return df



def _normalize_tici(series: pd.Series) -> pd.Series:
    def _norm_tici(x: Any) -> Any:
        if pd.isna(x):
            return pd.NA
        s = str(x).strip().lower()
        s = (s.replace(" ", "")
               .replace(",", ".")
               .replace("–", "-")
               .replace("_", "")
               .replace("ii", "2")   # tolerate roman numerals II, III
               .replace("iii", "3")
               )
        # allow forms like 2a., 2-a, 2_a
        s = s.replace("2a.", "2a").replace("2b.", "2b").replace("2c.", "2c")
        s = s.replace("-", "").replace(".", "")

        mapping = {
            "0": "0", "1": "1", "2": "2", "3": "3",
            "2a": "2a", "2b": "2b", "2c": "2c",
            "tii2a": "2a", "tii2b": "2b", "tii2c": "2c", "tii3": "3",
            "tici2a": "2a", "tici2b": "2b", "tici2c": "2c", "tici3": "3",
            "2b/3": "2b", "≥2b": "2b", "=>2b": "2b"
        }
        if s in mapping:
            return mapping[s]
        if s in {"0", "1", "2", "3"}:
            return s
        return pd.NA

    out = series.apply(_norm_tici).astype("object")
    dtype = CategoricalDtype(categories=["0", "1", "2a", "2b", "2c", "3"], ordered=True)
    return out.astype(dtype)


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    coerced_log: Dict[str, int] = {}
    for col in cols:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after = df[col].isna().sum()
            coerced_log[col] = max(0, after - before)  # newly coerced to NaN
    return df, coerced_log


def _to_binary(series: pd.Series, true_tokens: List[str], false_tokens: List[str], na_tokens: List[str]) -> pd.Series:
    def _map(v: Any) -> float | None:
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in true_tokens:
            return 1.0
        if s in false_tokens:
            return 0.0
        if s in na_tokens:
            return np.nan
        return np.nan
    return series.map(_map)


def clean_mt_patients(
    mt_patients_raw: pd.DataFrame,
    column_map: Dict[str, str],
    value_maps: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Deterministic, TRIPOD-compliant cleaning:
      - column normalization & renaming
      - semantic value translations (Czech->English)
      - whitespace/nbsp cleanup
      - missing markers normalization
      - TICI normalization to ordered categorical
      - numeric coercion with logging
      - binary normalization
      - IVT consistency rules
    Returns:
      cleaned_df, cleaning_log (dict)
    """
    log: Dict[str, Any] = {}

    df = mt_patients_raw.copy()
    df = _trim_and_fix_whitespace(df)

    # Keep a copy of original columns for diagnostics
    log["original_columns"] = df.columns.tolist()

    # --- apply semantic translations early (on original labels) ---
    translate_map = value_maps.get("translate_map", {}) if value_maps else {}
    df = _apply_value_translations(df, translate_map)

    # --- normalize headers (lower/underscore) before rename for safety ---
    df_norm_headers = df.copy()
    df_norm_headers = _normalize_columns(df_norm_headers)

    # --- rename using original column names map (from config) ---
    df_renamed, rename_log = _rename_columns(df, column_map)
    log["rename"] = rename_log

    # After renaming, normalize headers again to ensure snake_case
    df = _normalize_columns(df_renamed)

    # --- normalize missing markers ---
    df = _normalize_missing_markers(df)

    # --- specific fields normalization ---
    if "tici" in df.columns:
        df["tici"] = _normalize_tici(df["tici"])

    # --- numeric coercion ---
    numeric_cols = [
        "age", "aspects",
        "onset_to_ivt_min", "onset_to_puncture_min", "onset_to_recan_min",
        "procedure_duration",
        "nihss_admission", "nihss_24h", "nihss_7d",
        "mrs_before", "mrs90",
        "bmi", "systolic_bp", "diastolic_bp",
        "cholesterol", "glycemia",
    ]
    df, coerced_log = _coerce_numeric(df, numeric_cols)
    log["numeric_coercion_new_nans"] = coerced_log

    # --- binary normalization ---
    true_tokens = [t.lower() for t in value_maps.get("binary_true", ["yes","ano","1","true","y"])]
    false_tokens = [t.lower() for t in value_maps.get("binary_false", ["no","ne","0","false","n"])]
    na_tokens = [t.lower() for t in value_maps.get("binary_na", ["unknown","?","na","n/a","nan","none"])]

    binary_cols = [
        "ivt_given", "iat_given", "sich", "non_sich",
        "hypertension", "diabetes", "hyperlipidemia", "smoking", "alcohol_abuse",
        "arrhythmia", "tia_before", "cmp_before", "statins_before",
        "cerebral_edema", "decompression_surgery",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = _to_binary(df[col], true_tokens, false_tokens, na_tokens)


    # --- IVT consistency rules ---
    if "ivt_given" in df.columns:
        no_ivt_mask = df["ivt_given"].isin([0.0])  # normalized already
        if "onset_to_ivt_min" in df.columns:
            df.loc[no_ivt_mask, "onset_to_ivt_min"] = pd.NA

        if "ivt_different_hospital" in df.columns:
            df["ivt_different_hospital"] = df["ivt_different_hospital"].replace({None: pd.NA}).replace(r"^\s*$", pd.NA, regex=True)
            df.loc[no_ivt_mask, "ivt_different_hospital"] = "No finding"

        # QC: if ivt_given==1 then onset_to_ivt_min should usually exist (>90%)
        if "onset_to_ivt_min" in df.columns:
            with_ivt = df["ivt_given"].eq(1.0)
            prop_with_time = df.loc[with_ivt, "onset_to_ivt_min"].notna().mean() if with_ivt.any() else np.nan
            log["ivt_time_present_rate_when_ivt_given"] = float(prop_with_time)

    # --- final diagnostics ---
    log["final_columns"] = df.columns.tolist()
    log["row_count"] = int(df.shape[0])
    log["missing_fraction"] = df.isna().mean().round(3).to_dict()

    return df, log

# ----- ADD BELOW YOUR EXISTING CODE IN nodes.py -----
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt

def validate_and_fix(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    log: Dict[str, Any] = {"time_violations": {}, "range_violations": {}}

    # --- Monotonic time checks ---
    trio = ["onset_to_ivt_min", "onset_to_puncture_min", "onset_to_recan_min"]
    for c in trio:
        if c not in df.columns:
            log.setdefault("missing_columns", []).append(c)

    if all(c in df.columns for c in trio):
        v1_idx = df.index[
            df["onset_to_ivt_min"].notna()
            & df["onset_to_puncture_min"].notna()
            & (df["onset_to_ivt_min"] > df["onset_to_puncture_min"])
        ].tolist()
        v2_idx = df.index[
            df["onset_to_puncture_min"].notna()
            & df["onset_to_recan_min"].notna()
            & (df["onset_to_puncture_min"] > df["onset_to_recan_min"])
        ].tolist()

        # Fix strategy: blank the later step
        df.loc[v1_idx, "onset_to_puncture_min"] = pd.NA
        df.loc[v2_idx, "onset_to_recan_min"] = pd.NA

        log["time_violations"]["ivt_gt_puncture_rows"] = v1_idx
        log["time_violations"]["puncture_gt_recan_rows"] = v2_idx
        log["time_violations"]["ivt_gt_puncture_n"] = len(v1_idx)
        log["time_violations"]["puncture_gt_recan_n"] = len(v2_idx)
    
    # --- ASPECTS 0..10 ---
    if "aspects" in df.columns:
    # Only flag rows that are not missing AND outside [0, 10]
        mask_aspects_oor = df["aspects"].notna() & ~df["aspects"].between(0, 10, inclusive="both")
        bad_aspects = df.index[mask_aspects_oor].tolist()
        df.loc[mask_aspects_oor, "aspects"] = pd.NA
        log["range_violations"]["aspects_out_of_range_rows"] = bad_aspects
        log["range_violations"]["aspects_out_of_range_n"] = len(bad_aspects)

    # --- mRS (0..6), cast to Int64 ---
    for mrs_col in ["mrs_before", "mrs90"]:
        if mrs_col in df.columns:
            bad = df.index[df[mrs_col].notna() & ~df[mrs_col].isin([0, 1, 2, 3, 4, 5, 6])].tolist()
            df.loc[bad, mrs_col] = pd.NA
            df[mrs_col] = df[mrs_col].astype("Int64")
            log["range_violations"][f"{mrs_col}_out_of_range_rows"] = bad
            log["range_violations"][f"{mrs_col}_out_of_range_n"] = len(bad)

    # --- NIHSS 7d sanity: flag >42; blank absurd >99 ---
    if "nihss_7d" in df.columns:
        extreme = df.index[df["nihss_7d"] > 42].tolist()
        absurd = df.index[df["nihss_7d"] > 99].tolist()
        df.loc[absurd, "nihss_7d"] = pd.NA
        log["range_violations"]["nihss7d_flag_gt_42_rows"] = extreme
        log["range_violations"]["nihss7d_blanked_gt_99_rows"] = absurd

    # --- Binary casts to Int8 {0,1} where applicable (already normalized upstream) ---
    bin_cols = [
        "ivt_given","iat_given","sich","non_sich","hypertension","diabetes","hyperlipidemia",
        "smoking","alcohol_abuse","arrhythmia","tia_before","cmp_before","statins_before",
        "cerebral_edema","decompression_surgery",
    ]
    for c in bin_cols:
        if c in df.columns:
            df[c] = df[c].astype("Int8")

    return df, log

def make_lightgbm_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a LightGBM-ready table with:
      - no imputation (NaNs kept)
      - explicit missingness indicators for key predictors
      - a simple onset_year feature (from onset_date) if available

    This runs on mt_patients_validated (after exclusions & range checks).
    """
    df = df.copy()

    # --- Simple derived feature: onset_year (used already in your model) ---
    if "onset_date" in df.columns and "onset_year" not in df.columns:
        onset_dt = pd.to_datetime(df["onset_date"], errors="coerce")
        df["onset_year"] = onset_dt.dt.year.astype("Int64")

    # --- Numeric features where missingness is clinically relevant ---
    numeric_for_flags = [
        "age",
        "aspects",
        "onset_to_ivt_min",
        "onset_to_puncture_min",
        "onset_to_recan_min",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "cholesterol",
        "glycemia",
        # handle whichever NIHSS naming you actually have:
        "nihss_admission",
        "admission_nihss",
        "nihss_24h",
        "nihss_7d",
        "mrs_before",
        "mrs90",
    ]

    for col in numeric_for_flags:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype("Int8")

    # --- Categorical / binary features where missingness may carry signal ---
    categorical_for_flags = [
        "sex",
        "hemisphere",
        "occlusion_site",
        "etiology",
        "ivt_given",
        "thrombolytics",
        "ivt_different_hospital",
        "transfer_from_other_hospital",
        "iat_given",
        "antithrombotics_before",
        "statins_before",
        "hypertension",
        "diabetes",
        "hyperlipidemia",
        "smoking",
        "alcohol_abuse",
        "arrhythmia",
        "tia_before",
        "cmp_before",
        "heart_condition",
        "cerebral_edema",
        "decompression_surgery",
        "anesthesia",
    ]

    for col in categorical_for_flags:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype("Int8")

    return df

# ======== NEW NODE: create mt_patients_regression_ready with MICE + logs ========

from typing import Any, Dict, Tuple


def make_regression_ready(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Take mt_lightgbm_ready (already cleaned + with missingness flags)
    and produce mt_patients_regression_ready by:

      - Automatically detecting which variables to impute
      - MICE (IterativeImputer) for numeric columns
      - Mode imputation for categorical/binary columns
      - Keeping all *_missing flag columns unchanged
      - Returning a log dict with:
          * which columns were imputed
          * missing counts before/after
          * MICE iterations
          * success flag (True if all imputed cols have no NA)

    This is a single complete-case dataset suitable for
    multivariable logistic regression.
    """

    df = df.copy()
    log: Dict[str, Any] = {}

    # --- Protect some columns from imputation ---
    protected_cols = []
    for cand in ["mrs90", "good_mrs", "patient_id", "case_id", "row_id"]:
        if cand in df.columns:
            protected_cols.append(cand)

    # Missingness flags should NOT be imputed
    missing_flag_cols = [c for c in df.columns if c.endswith("_missing")]

    # --- Detect numeric columns to impute ---
    numeric_cols = [
        c
        for c in df.select_dtypes(include=["number"]).columns
        if c not in protected_cols and c not in missing_flag_cols
    ]

    # --- Detect categorical columns to impute ---
    categorical_cols: list[str] = []
    for c in df.columns:
        if c in protected_cols or c in missing_flag_cols:
            continue
        dtype_str = str(df[c].dtype)
        if dtype_str.startswith("object") or dtype_str.startswith("category"):
            categorical_cols.append(c)

    # Save what we decided to impute
    log["protected_cols"] = protected_cols
    log["missing_flag_cols"] = missing_flag_cols
    log["numeric_cols_imputed"] = numeric_cols
    log["categorical_cols_imputed"] = categorical_cols

    # Missing counts BEFORE imputation
    cols_for_missing = numeric_cols + categorical_cols
    log["missing_before"] = (
        df[cols_for_missing].isna().sum().astype(int).to_dict()
        if cols_for_missing
        else {}
    )

    # --- MICE (IterativeImputer) for numeric columns ---
    if numeric_cols:
        try:
            imp = IterativeImputer(
                random_state=random_state,
                sample_posterior=False,
                max_iter=20,
                initial_strategy="median",
            )
            numeric_array = imp.fit_transform(df[numeric_cols])
            df[numeric_cols] = numeric_array
            log["mice_n_iter"] = int(getattr(imp, "n_iter_", 0))
            log["mice_success"] = True
        except Exception as e:
            # Fallback: median imputation if MICE fails
            log["mice_success"] = False
            log["mice_error"] = str(e)
            for col in numeric_cols:
                median_val = df[col].median(skipna=True)
                df[col] = df[col].fillna(median_val)
    else:
        log["mice_n_iter"] = 0
        log["mice_success"] = True

    # --- Mode imputation for categorical/binary columns ---
    for col in categorical_cols:
        s = df[col]
        mode_val = s.dropna().mode()
        if not mode_val.empty:
            df[col] = s.fillna(mode_val.iloc[0])
        else:
            # Fallback: consistent "Unknown" category
            df[col] = s.fillna("Unknown")

    # Missing counts AFTER imputation
    log["missing_after"] = (
        df[cols_for_missing].isna().sum().astype(int).to_dict()
        if cols_for_missing
        else {}
    )

    # --- Overall success flag ---
    total_missing_after = sum(log["missing_after"].values()) if log["missing_after"] else 0
    log["success"] = (total_missing_after == 0) and bool(log.get("mice_success", True))

    return df, log


def merge_reports(eda_audit_report: Dict[str, Any], data_quality_violations: Dict[str, Any]) -> Dict[str, Any]:
    """Merge EDA audit and DQ violations into one JSON-able dict."""
    return {
        "eda_audit": eda_audit_report,
        "data_quality": data_quality_violations,
        "generated_from": "kedro pipeline: data_cleaning + eda_audit merge",
    }


def plot_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dict of matplotlib Figures for NIHSS_7d (hist), mRS90 (bar), TICI (bar)."""
    figs: Dict[str, Any] = {}

    # NIHSS 7d histogram
    if "nihss_7d" in df.columns:
        s = df["nihss_7d"].dropna()
        fig1 = plt.figure()
        s.plot(kind="hist", bins=30)
        plt.title("NIHSS at 7 days")
        plt.xlabel("NIHSS 7d")
        plt.ylabel("Count")
        figs["nihss_7d_hist"] = fig1

    # mRS90 bar
    if "mrs90" in df.columns:
        s = df["mrs90"].dropna()
        # cast to Int for nicer bars if possible
        try:
            s = s.astype("Int64").astype("float").astype(int)
        except Exception:
            pass
        vc = s.value_counts().sort_index()
        fig2 = plt.figure()
        vc.plot(kind="bar")
        plt.title("mRS at 90 days")
        plt.xlabel("mRS90")
        plt.ylabel("Count")
        figs["mrs90_bar"] = fig2

    # TICI bar
    if "tici" in df.columns:
        s = df["tici"].astype("string").dropna()
        vc = s.value_counts().sort_index()
        fig3 = plt.figure()
        vc.plot(kind="bar")
        plt.title("TICI grades")
        plt.xlabel("TICI")
        plt.ylabel("Count")
        figs["tici_bar"] = fig3

    return figs

def exclude_invalid_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Exclude records with physiologically or logically impossible values:
      - onset_to_ivt_min > onset_to_puncture_min
      - onset_to_puncture_min > onset_to_recan_min
      - ASPECTS > 10
      - any NIHSS_* > 42
    Returns: filtered_df, exclusion_summary
    """
    initial_count = len(df)
    removed = {}

    # --- Logical time-order violations ---
    if set(["onset_to_ivt_min", "onset_to_puncture_min"]).issubset(df.columns):
        mask_ivt_gt_puncture = (
            df["onset_to_ivt_min"].notna()
            & df["onset_to_puncture_min"].notna()
            & (df["onset_to_ivt_min"] > df["onset_to_puncture_min"])
        )
        removed["ivt_gt_puncture"] = int(mask_ivt_gt_puncture.sum())
        df = df.loc[~mask_ivt_gt_puncture]

    if set(["onset_to_puncture_min", "onset_to_recan_min"]).issubset(df.columns):
        mask_puncture_gt_recan = (
            df["onset_to_puncture_min"].notna()
            & df["onset_to_recan_min"].notna()
            & (df["onset_to_puncture_min"] > df["onset_to_recan_min"])
        )
        removed["puncture_gt_recan"] = int(mask_puncture_gt_recan.sum())
        df = df.loc[~mask_puncture_gt_recan]

    # --- Clinical plausibility ---
    if "aspects" in df.columns:
        mask_aspects = df["aspects"].notna() & (df["aspects"] > 10)
        removed["aspects_gt10"] = int(mask_aspects.sum())
        df = df.loc[~mask_aspects]

    for col in ["nihss_admission", "nihss_24h", "nihss_7d"]:
        if col in df.columns:
            mask_nihss = df[col].notna() & (df[col] > 42)
            removed[f"{col}_gt42"] = int(mask_nihss.sum())
            df = df.loc[~mask_nihss]

    final_count = len(df)
    removed["total_removed"] = initial_count - final_count
    removed["final_rows"] = final_count
    return df, removed

