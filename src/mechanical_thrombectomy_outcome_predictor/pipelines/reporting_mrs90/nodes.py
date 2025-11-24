# src/mechanical_thrombectomy_outcome_predictor/pipelines/reporting_mrs90/nodes.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

from mechanical_thrombectomy_outcome_predictor.pipelines.baseline_characteristics.nodes import (
    _compute_smd,
)
from mechanical_thrombectomy_outcome_predictor.pipelines.modeling_mrs90.config import (
    get_monotone_sign_map,
    build_monotone_constraints,
)
from mechanical_thrombectomy_outcome_predictor.pipelines.modeling_mrs90.calibration_utils import (
    apply_logistic_calibration,
)


# ============================================================
# A. Cohort flow & outcome availability
# ============================================================

def build_cohort_flow(
    mt_patients_master: pd.DataFrame,
    mt_patients_clean: pd.DataFrame,
    mt_patients_valid: pd.DataFrame,
    mt_patients_validated: pd.DataFrame,
    mt_patients_split: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a simple cohort flow table: raw -> clean -> valid -> validated -> split -> mRS90 observed.

    Uses row counts from each stage. This is meant for reporting / CONSORT-like figure.
    """
    rows: List[Dict[str, Any]] = []

    def add_row(step: int, name: str, description: str, n: int) -> None:
        rows.append(
            {
                "step": step,
                "name": name,
                "description": description,
                "n_patients": int(n),
            }
        )

    # Step 1: raw registry
    add_row(
        1,
        "raw_registry",
        "Patients in raw thrombectomy registry (before cleaning)",
        len(mt_patients_master),
    )

    # Step 2: after deterministic cleaning
    add_row(
        2,
        "cleaned",
        "Records after deterministic cleaning (header normalization, value translations, missing markers)",
        len(mt_patients_clean),
    )

    # Step 3: after plausibility exclusions
    add_row(
        3,
        "valid",
        "Records after exclusion of physiologically impossible values and logical time-order violations",
        len(mt_patients_valid),
    )

    # Step 4: after validate_and_fix
    add_row(
        4,
        "validated",
        "Validated cohort after range checks and type casting",
        len(mt_patients_validated),
    )

    # Step 5: split-annotated dataset
    add_row(
        5,
        "split_annotated",
        "Validated cohort with temporal split labels (train vs test)",
        len(mt_patients_split),
    )

    # Step 6: cases with observed mRS90
    if "mrs90" in mt_patients_validated.columns:
        n_mrs90_obs = int(mt_patients_validated["mrs90"].notna().sum())
        add_row(
            6,
            "mrs90_observed",
            "Patients with observed 90-day mRS (included in outcome modeling)",
            n_mrs90_obs,
        )

    flow_df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return flow_df


def plot_cohort_flow_diagram(
    cohort_flow_table: pd.DataFrame,
) -> plt.Figure:
    """
    Simple vertical cohort flow diagram based on the cohort_flow_table.

    This is not a full-blown CONSORT diagram, but good enough for an appendix figure.
    """
    df = cohort_flow_table.sort_values("step").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6, 1.2 * len(df) + 1))
    ax.axis("off")

    y_positions = np.linspace(1.0, 0.1, len(df))

    for (idx, row), y in zip(df.iterrows(), y_positions):
        label = f"{row['name']}: n={row['n_patients']}\n{row['description']}"
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white"),
            fontsize=9,
            transform=ax.transAxes,
        )

    # arrows between boxes
    for i in range(len(y_positions) - 1):
        y1 = y_positions[i] - 0.05
        y2 = y_positions[i + 1] + 0.05
        ax.annotate(
            "",
            xy=(0.5, y2),
            xytext=(0.5, y1),
            arrowprops=dict(arrowstyle="->", lw=1),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
        )

    fig.suptitle("Cohort flow – thrombectomy registry to mRS90 modeling set", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def mrs90_outcome_availability(
    mt_patients_validated: pd.DataFrame,
    mt_patients_split: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize availability of 90-day mRS overall and by temporal split.

    Returns a table with rows: overall, train, test.
    """
    if "patient_id" not in mt_patients_validated.columns:
        raise ValueError("mt_patients_validated must contain 'patient_id'.")

    if "patient_id" not in mt_patients_split.columns or "split" not in mt_patients_split.columns:
        raise ValueError("mt_patients_split must contain 'patient_id' and 'split'.")

    df = mt_patients_validated[["patient_id", "mrs90"]].merge(
        mt_patients_split[["patient_id", "split"]],
        on="patient_id",
        how="left",
    )

    rows: List[Dict[str, Any]] = []

    def add_row(label: str, mask: pd.Series) -> None:
        subset = df[mask]
        n = len(subset)
        n_avail = int(subset["mrs90"].notna().sum())
        n_missing = int(subset["mrs90"].isna().sum())
        rows.append(
            {
                "subset": label,
                "n_patients": n,
                "mrs90_available_n": n_avail,
                "mrs90_available_pct": float(n_avail / n) if n > 0 else np.nan,
                "mrs90_missing_n": n_missing,
                "mrs90_missing_pct": float(n_missing / n) if n > 0 else np.nan,
            }
        )

    add_row("overall", pd.Series(True, index=df.index))
    add_row("train", df["split"] == "train")
    add_row("test", df["split"] == "test")

    out = pd.DataFrame(rows)
    return out


# ============================================================
# A3. Missingness overview for LightGBM predictors
# ============================================================

def mrs90_missingness_table(
    mt_lightgbm_ready: pd.DataFrame,
    mt_patients_split: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a missingness table for all predictors in mt_lightgbm_ready.

    For each column:
      - overall % missing
      - % missing in train and in test (based on temporal split)
      - whether it is a *_missing indicator
      - whether a missing flag exists for its base variable
      - a simple handling_strategy label
    """
    if "patient_id" not in mt_lightgbm_ready.columns:
        raise ValueError("mt_lightgbm_ready must contain 'patient_id'.")

    if "patient_id" not in mt_patients_split.columns or "split" not in mt_patients_split.columns:
        raise ValueError("mt_patients_split must contain 'patient_id' and 'split'.")

    df = mt_lightgbm_ready.merge(
        mt_patients_split[["patient_id", "split"]],
        on="patient_id",
        how="left",
    )

    all_cols = list(mt_lightgbm_ready.columns)
    rows: List[Dict[str, Any]] = []

    for col in all_cols:
        if col == "patient_id":
            continue

        is_missing_indicator = col.endswith("_missing")
        base_name = col[:-8] if is_missing_indicator else col
        flag_name = f"{base_name}_missing"

        has_missing_flag = flag_name in all_cols and not is_missing_indicator

        # compute missing fractions
        overall = df[col].isna().mean()
        train = df.loc[df["split"] == "train", col].isna().mean()
        test = df.loc[df["split"] == "test", col].isna().mean()

        # crude handling strategy label
        if is_missing_indicator:
            handling = "missing_indicator"
        elif has_missing_flag:
            handling = "modeled_with_missing_flag"
        else:
            handling = "direct_input_with_NaNs"

        rows.append(
            {
                "variable": col,
                "base_variable": base_name,
                "is_missing_indicator": bool(is_missing_indicator),
                "has_missing_flag": bool(has_missing_flag),
                "missing_overall_pct": float(overall),
                "missing_train_pct": float(train),
                "missing_test_pct": float(test),
                "handling_strategy": handling,
                "dtype": str(mt_lightgbm_ready[col].dtype),
            }
        )

    out = pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)
    return out


# ============================================================
# B. Baseline by outcome (train set)
# ============================================================

def make_baseline_by_outcome(
    mrs90_X_train: pd.DataFrame,
    mrs90_y_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Baseline pre-procedural characteristics by 90-day mRS outcome in TRAIN.

    For each predictor:
      - numeric: mean ± SD and n in mRS 0–2 vs 3–6 + SMD
      - categorical: counts (dict) and SMD
    """
    y = mrs90_y_train.iloc[:, 0]
    if not set(np.unique(y.dropna())).issubset({0, 1}):
        raise ValueError("mrs90_y_train must be binary coded 0/1 for mRS 3–6 / 0–2.")

    df = mrs90_X_train.copy()
    df["_outcome"] = y

    good = df[df["_outcome"] == 1]
    poor = df[df["_outcome"] == 0]

    rows: List[Dict[str, Any]] = []

    for col in mrs90_X_train.columns:
        x_good = good[col]
        x_poor = poor[col]
        smd = _compute_smd(x_good, x_poor)

        if x_good.dtype.kind in "bifc" and x_poor.dtype.kind in "bifc":
            row = {
                "variable": col,
                "type": "numeric",
                "good_n": int(x_good.notna().sum()),
                "good_mean": float(x_good.mean()) if x_good.notna().any() else None,
                "good_sd": float(x_good.std(ddof=1)) if x_good.notna().any() else None,
                "poor_n": int(x_poor.notna().sum()),
                "poor_mean": float(x_poor.mean()) if x_poor.notna().any() else None,
                "poor_sd": float(x_poor.std(ddof=1)) if x_poor.notna().any() else None,
                "smd": float(smd),
            }
        else:
            vc_good = x_good.value_counts(dropna=False)
            vc_poor = x_poor.value_counts(dropna=False)

            good_counts = {str(k): int(v) for k, v in vc_good.items()}
            poor_counts = {str(k): int(v) for k, v in vc_poor.items()}

            row = {
                "variable": col,
                "type": "categorical",
                "good_counts": good_counts,
                "poor_counts": poor_counts,
                "smd": float(smd),
            }

        rows.append(row)

    table = pd.DataFrame(rows)
    return table


# ============================================================
# B5. Predictor dictionary for TRIPOD
# ============================================================

def build_predictor_dictionary(
    mt_lightgbm_ready: pd.DataFrame,
    mrs90_feature_list: List[str],
    mrs90_lgbm_model: LGBMClassifier,
) -> pd.DataFrame:
    """
    Build a predictor 'data dictionary' for TRIPOD / appendix.

    For each column in mt_lightgbm_ready:
      - type, dtype
      - monotone prior sign (if defined)
      - whether it has *_missing flag
      - whether it is actually used by the final model
      - whether it is part of an interaction
    """
    all_cols = list(mt_lightgbm_ready.columns)
    sign_map = get_monotone_sign_map()

    model_feats: List[str]
    if hasattr(mrs90_lgbm_model, "feature_name_") and mrs90_lgbm_model.feature_name_ is not None:
        model_feats = list(mrs90_lgbm_model.feature_name_)
    else:
        model_feats = list(mrs90_feature_list)

    used_set = set(model_feats)
    feature_list_set = set(mrs90_feature_list)

    rows: List[Dict[str, Any]] = []

    for col in all_cols:
        if col == "patient_id":
            continue

        dtype = mt_lightgbm_ready[col].dtype
        is_missing_indicator = col.endswith("_missing")
        base_name = col[:-8] if is_missing_indicator else col
        flag_name = f"{base_name}_missing"
        has_missing_flag = flag_name in all_cols and not is_missing_indicator

        if str(dtype).startswith(("int", "float", "bool")):
            var_type = "numeric"
        else:
            var_type = "categorical_or_text"

        # monotone prior: try column name, then base name, else 0
        mono_sign = sign_map.get(col, sign_map.get(base_name, 0))

        interaction_component = ("_x_" in col) or (col in {"age_x_nihss", "onset_to_puncture_min_x_nihss", "age_x_onset_to_puncture_min"})

        rows.append(
            {
                "variable": col,
                "base_variable": base_name,
                "included_in_candidate_feature_list": bool(col in feature_list_set),
                "used_in_final_model": bool(col in used_set),
                "type": var_type,
                "dtype": str(dtype),
                "is_missing_indicator": bool(is_missing_indicator),
                "has_missing_flag": bool(has_missing_flag),
                "monotone_sign": int(mono_sign),
                "interaction_component": bool(interaction_component),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["base_variable", "is_missing_indicator", "variable"]
    ).reset_index(drop=True)

    return df


# ============================================================
# C. CV summary & performance tables
# ============================================================

def build_cv_summary_table(
    mrs90_lgbm_cv_metrics_raw_10x20: Dict[str, Any],
) -> pd.DataFrame:
    """
    Summarize cross-validation metrics from the Optuna tuning JSON.

    Expected structure: the 'cv_metrics' dict produced in optuna_tune_mrs90_lgbm.
    """
    cv = mrs90_lgbm_cv_metrics_raw_10x20
    summary = cv.get("summary", {})
    n_splits = cv.get("n_splits", None)
    n_repeats = cv.get("n_repeats", 1)
    optuna_info = cv.get("optuna", {})

    rows: List[Dict[str, Any]] = []

    for metric_name in ["auc", "brier", "log_loss"]:
        m = summary.get(metric_name, {})
        rows.append(
            {
                "metric": metric_name,
                "mean": float(m.get("mean", np.nan)),
                "std": float(m.get("std", np.nan)),
                "n_splits": int(n_splits) if n_splits is not None else None,
                "n_repeats": int(n_repeats) if n_repeats is not None else None,
                "optuna_n_trials": int(optuna_info.get("n_trials", 0)),
                "optuna_best_value": float(optuna_info.get("best_value", np.nan)),
                "optuna_best_trial_number": int(
                    optuna_info.get("best_trial_number", -1)
                ),
            }
        )

    return pd.DataFrame(rows)


def build_performance_summary(
    mrs90_calibration_metrics_raw: Dict[str, Any],
) -> pd.DataFrame:
    """
    Turn calibration_mrs90_analysis metrics JSON into a flat table.

    Expected keys: 'train', 'test', 'train_calibrated', 'test_calibrated'.
    """
    metrics = mrs90_calibration_metrics_raw
    rows: List[Dict[str, Any]] = []

    for key in ["train", "test", "train_calibrated", "test_calibrated"]:
        if key not in metrics:
            continue
        m = metrics[key]
        rows.append(
            {
                "dataset": key,
                "n": int(m.get("n", 0)),
                "auc": float(m.get("auc", np.nan)),
                "brier": float(m.get("brier", np.nan)),
                "log_loss": float(m.get("log_loss", np.nan)),
                "cal_intercept": float(m.get("cal_intercept", np.nan)),
                "cal_slope": float(m.get("cal_slope", np.nan)),
            }
        )

    return pd.DataFrame(rows)


def build_threshold_performance(
    mrs90_test_predictions: pd.DataFrame,
    mrs90_logistic_calibration_params: Dict[str, Any],
    thresholds: List[float] | None = None,
) -> pd.DataFrame:
    """
    Threshold-based performance on TEST using CALIBRATED probabilities.

    Inputs:
      - mrs90_test_predictions: from evaluate_mrs90_lgbm()
        must have columns 'y_true' and 'y_pred_proba' (RAW LightGBM probs).
      - mrs90_logistic_calibration_params: JSON with 'a' and 'b'.
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    if "y_true" not in mrs90_test_predictions.columns or "y_pred_proba" not in mrs90_test_predictions.columns:
        raise ValueError("mrs90_test_predictions must contain 'y_true' and 'y_pred_proba'.")

    y_true = mrs90_test_predictions["y_true"].astype(int).values
    p_raw = mrs90_test_predictions["y_pred_proba"].values

    a = float(mrs90_logistic_calibration_params.get("a", 0.0))
    b = float(mrs90_logistic_calibration_params.get("b", 1.0))

    p_cal = apply_logistic_calibration(p_raw, a, b)

    N = len(y_true)
    rows: List[Dict[str, Any]] = []

    for t in thresholds:
        pred_pos = p_cal >= t

        TP = int(((pred_pos == 1) & (y_true == 1)).sum())
        FP = int(((pred_pos == 1) & (y_true == 0)).sum())
        TN = int(((pred_pos == 0) & (y_true == 0)).sum())
        FN = int(((pred_pos == 0) & (y_true == 1)).sum())

        def safe_div(num, den):
            return float(num / den) if den > 0 else np.nan

        sens = safe_div(TP, TP + FN)
        spec = safe_div(TN, TN + FP)
        ppv = safe_div(TP, TP + FP)
        npv = safe_div(TN, TN + FN)
        acc = safe_div(TP + TN, N)
        prop_high = float(pred_pos.mean())
        if pred_pos.any():
            obs_event_high = float(y_true[pred_pos].mean())
        else:
            obs_event_high = np.nan

        rows.append(
            {
                "threshold": float(t),
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "sensitivity": sens,
                "specificity": spec,
                "PPV": ppv,
                "NPV": npv,
                "accuracy": acc,
                "proportion_high_risk": prop_high,
                "observed_event_rate_high_risk": obs_event_high,
                "n": N,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# D. Model specification JSON
# ============================================================

def build_model_spec(
    mrs90_lgbm_model: LGBMClassifier,
    mrs90_lgbm_optuna_best_params: Dict[str, Any],
    mrs90_logistic_calibration_params: Dict[str, Any],
    mrs90_feature_list: List[str],
) -> Dict[str, Any]:
    """
    Build a JSON-serializable specification of the final model:

      - ordered feature list actually used by model
      - LightGBM parameters
      - monotone constraints for that feature list
      - Optuna best trial parameters (incl. interaction flags)
      - logistic calibration parameters (a, b)
    """
    if hasattr(mrs90_lgbm_model, "feature_name_") and mrs90_lgbm_model.feature_name_ is not None:
        feature_names = list(mrs90_lgbm_model.feature_name_)
    else:
        feature_names = list(mrs90_feature_list)

    monotone_constraints = build_monotone_constraints(feature_names)
    lgbm_params = mrs90_lgbm_model.get_params()

    spec: Dict[str, Any] = {
        "note": (
            "Specification of pre-procedural LightGBM model for predicting mRS 0-2 at 90 days. "
            "To obtain calibrated probabilities, first compute p_raw from LightGBM, then apply "
            "logistic calibration: p_cal = logistic(a + b * logit(p_raw))."
        ),
        "feature_list_used": feature_names,
        "monotone_constraints": monotone_constraints,
        "lightgbm_params": lgbm_params,
        "optuna_best_params": mrs90_lgbm_optuna_best_params,
        "logistic_calibration": {
            "a": float(mrs90_logistic_calibration_params.get("a", 0.0)),
            "b": float(mrs90_logistic_calibration_params.get("b", 1.0)),
            "n_train": int(mrs90_logistic_calibration_params.get("n_train", 0)),
            "n_splits_cv": int(mrs90_logistic_calibration_params.get("n_splits_cv", 0)),
            "note": mrs90_logistic_calibration_params.get("note", ""),
        },
    }

    return spec


# ============================================================
# D10. SHAP top 30
# ============================================================

def build_shap_top30(
    mrs90_shap_summary_test: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Truncate global test-set SHAP summary to the top N features.

    Expects columns ['feature', 'mean_abs_shap'] sorted descending
    (as produced by compute_mrs90_shap).
    """
    if "feature" not in mrs90_shap_summary_test.columns or "mean_abs_shap" not in mrs90_shap_summary_test.columns:
        raise ValueError("mrs90_shap_summary_test must have 'feature' and 'mean_abs_shap' columns.")

    df = mrs90_shap_summary_test.sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    df_top = df.head(top_n).copy()
    df_top["rank"] = np.arange(1, len(df_top) + 1)
    return df_top[["rank", "feature", "mean_abs_shap"]]


# ============================================================
# E12. DCA numeric summary
# ============================================================

def build_dca_summary(
    mrs90_decision_curve_calibrated: pd.DataFrame,
    thresholds_of_interest: List[float] | None = None,
) -> pd.DataFrame:
    """
    Numeric summary of DCA at a few prespecified thresholds.

    Inputs:
      - mrs90_decision_curve_calibrated: CSV produced by decision_curve_mrs90.py
        with columns ['threshold', 'net_benefit_model', 'net_benefit_all', 'net_benefit_none'].
    """
    if thresholds_of_interest is None:
        thresholds_of_interest = [0.20, 0.30, 0.40, 0.50]

    df = mrs90_decision_curve_calibrated.copy()
    # robust float comparison
    df["threshold_rounded"] = df["threshold"].round(3)
    toi = [round(t, 3) for t in thresholds_of_interest]

    subset = df[df["threshold_rounded"].isin(toi)].copy()
    subset = subset.drop(columns=["threshold_rounded"])

    subset = subset.sort_values("threshold").reset_index(drop=True)
    return subset

def build_tripod_excel(
    mt_cohort_flow_table: pd.DataFrame,
    mrs90_outcome_availability: pd.DataFrame,
    mrs90_missingness_table: pd.DataFrame,
    mrs90_predictor_dictionary: pd.DataFrame,
    mrs90_model_spec: Dict[str, Any],
    mrs90_cv_summary_table: pd.DataFrame,
    mrs90_performance_summary: pd.DataFrame,
    mrs90_threshold_performance: pd.DataFrame,
    mrs90_shap_top30: pd.DataFrame,
    mrs90_dca_summary: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Bundle all TRIPOD-related outputs into a dict of DataFrames
    keyed by sheet name. The Excel dataset will write each sheet.
    """

    # Flatten model spec into a key/value table so it fits nicely in Excel
    def _flatten_dict(prefix: str, obj: Any, rows: List[Dict[str, Any]]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                _flatten_dict(new_prefix, v, rows)
        else:
            if isinstance(obj, (dict, list)):
                v_str = json.dumps(obj)
            else:
                v_str = obj
            rows.append({"key": prefix, "value": v_str})

    model_spec_rows: List[Dict[str, Any]] = []
    _flatten_dict("", mrs90_model_spec, model_spec_rows)
    model_spec_df = pd.DataFrame(model_spec_rows) if model_spec_rows else pd.DataFrame(
        columns=["key", "value"]
    )

    # Return dict of sheets -> DataFrames
    tripod_book: Dict[str, pd.DataFrame] = {
        "1_cohort_flow": mt_cohort_flow_table,
        "2_outcome_availability": mrs90_outcome_availability,
        "3_missingness": mrs90_missingness_table,
        "4_predictor_dictionary": mrs90_predictor_dictionary,
        "5_model_spec": model_spec_df,
        "6_cv_summary": mrs90_cv_summary_table,
        "7_performance_summary": mrs90_performance_summary,
        "8_threshold_performance": mrs90_threshold_performance,
        "9_shap_top30": mrs90_shap_top30,
        "10_dca_summary": mrs90_dca_summary,
    }

    return tripod_book
