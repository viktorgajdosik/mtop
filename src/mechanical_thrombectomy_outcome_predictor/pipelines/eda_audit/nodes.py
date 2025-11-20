from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any

def _to_py(v):
    """Convert NumPy / pandas scalars and NA to vanilla Python types."""
    if v is None:
        return None
    try:
        import pandas as pd  # local import ok
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

def _counts_as_str_keys(ser: pd.Series) -> Dict[str, int]:
    """value_counts with JSON-safe string keys."""
    vc = ser.value_counts(dropna=False)
    out: Dict[str, int] = {}
    for k, v in vc.items():
        key = "NaN" if (k is None or (isinstance(k, float) and pd.isna(k)) or (hasattr(pd, "isna") and pd.isna(k))) else str(_to_py(k))
        out[key] = int(_to_py(v))
    return out

def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    # --- basic
    report["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    report["columns"] = [str(c) for c in df.columns.tolist()]

    # --- missingness
    miss = df.isna().mean().sort_values(ascending=False)
    report["missingness_top20"] = {str(k): float(v) for k, v in miss.head(20).round(4).to_dict().items()}

    # --- outcome prevalence / ranges
    outcomes = ["sich", "cerebral_edema", "nihss_7d", "mrs90"]
    out_summary: Dict[str, Any] = {}
    for c in outcomes:
        if c in df.columns:
            ser = df[c]
            if ser.dropna().dtype.kind in "biufc":
                if ser.nunique(dropna=True) <= 10:
                    out_summary[c] = {
                        "type": "categorical_or_few_levels",
                        "counts": _counts_as_str_keys(ser),
                    }
                else:
                    out_summary[c] = {
                        "type": "numeric",
                        "min": float(ser.min(skipna=True)) if ser.notna().any() else None,
                        "median": float(ser.median(skipna=True)) if ser.notna().any() else None,
                        "max": float(ser.max(skipna=True)) if ser.notna().any() else None,
                        "n_missing": int(ser.isna().sum()),
                    }
            else:
                out_summary[c] = {
                    "type": "object",
                    "counts": _counts_as_str_keys(ser.astype("string")),
                }
    report["outcomes"] = out_summary

    # --- logic constraints (still informative on validated data)
    violations: Dict[str, int] = {}

    if "ivt_given" in df.columns and "onset_to_ivt_min" in df.columns:
        no_ivt = df["ivt_given"].isin([0, 0.0]).fillna(False)
        v = int((no_ivt & df["onset_to_ivt_min"].notna()).sum())
        violations["ivt_time_present_despite_no_ivt"] = v

    trio = ["onset_to_ivt_min", "onset_to_puncture_min", "onset_to_recan_min"]
    if all(c in df.columns for c in trio):
        t = df[trio].copy()
        v1 = int((t["onset_to_ivt_min"].notna() & t["onset_to_puncture_min"].notna() &
                  (t["onset_to_ivt_min"] > t["onset_to_puncture_min"])).sum())
        v2 = int((t["onset_to_puncture_min"].notna() & t["onset_to_recan_min"].notna() &
                  (t["onset_to_puncture_min"] > t["onset_to_recan_min"])).sum())
        violations["ivt_gt_puncture"] = v1
        violations["puncture_gt_recan"] = v2

    if "patient_id" in df.columns:
        dup = int(df["patient_id"].duplicated().sum())
        violations["duplicate_patient_id"] = dup

    report["violations"] = {str(k): int(v) for k, v in violations.items()}
    return report
