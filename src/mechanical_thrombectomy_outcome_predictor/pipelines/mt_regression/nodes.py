from __future__ import annotations

import os
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

import shap
from pandas.api.types import is_numeric_dtype
from patsy import bs  # for age splines


# ============================================================
# CONFIG
# ============================================================

# Base output folder for all figures & Excel (Kedro will still store tables separately)
BASE_OUT = "data/08_reporting/mt_regression"

# Cross-validation settings
CV_FOLDS = 10
CV_REPEATS = 20
CV_RANDOM_SEED = 123


# ============================================================
# HELPERS: dirs, data prep, encoding
# ============================================================

def _ensure_dirs() -> None:
    subdirs = [
        "figures/pre",
        "figures/post",
        "figures/age",
        "figures/shap",
        "figures/cv",
        "tables",
    ]
    for sd in subdirs:
        os.makedirs(os.path.join(BASE_OUT, sd), exist_ok=True)


def _typical_row(data: pd.DataFrame, predictors: List[str]) -> pd.Series:
    """
    Build a 'typical' patient row:

    - numeric columns  -> median
    - non-numeric cols -> most frequent category (mode)

    Used for PDP and heatmaps to avoid crashing on string categories
    like 'CAD', 'No finding', etc.
    """
    values: Dict[str, Any] = {}

    for col in predictors:
        s = data[col]
        if is_numeric_dtype(s):
            values[col] = s.median()
        else:
            m = s.mode(dropna=True)
            values[col] = m.iloc[0] if not m.empty else np.nan

    return pd.Series(values, index=predictors)


def _encoded_numeric_matrix(df: pd.DataFrame, predictors: List[str]) -> np.ndarray:
    """
    Return a numeric matrix (numpy) for sklearn models:

    - Non-numeric columns are converted to categorical codes
      (-1 -> NaN) and cast to float.
    """
    X = df[predictors].copy()
    for col in X.columns:
        if not is_numeric_dtype(X[col]):
            cat = X[col].astype("category")
            codes = cat.cat.codes.replace(-1, np.nan)
            X[col] = codes.astype(float)
    return X.values


def build_pre_post_datasets(
    mt_patients_regression_ready: pd.DataFrame,
    mt_patients_split: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    List[str], List[str]
]:
    """
    Merge regression-ready table with temporal split info and construct
    PRE and POST datasets for logistic regression, using a temporal
    train/test split (TRIPOD-friendly).

    Inputs
    -------
      mt_patients_regression_ready: output of make_regression_ready()
      mt_patients_split: must contain ['patient_id', 'split']

    Returns
    -------
      pre_train_df, pre_test_df,
      post_train_df, post_test_df,
      pre_predictors, post_predictors
    (all DataFrames include 'good_mRS' as outcome)
    """
    df_ready = mt_patients_regression_ready.copy()
    df_split = mt_patients_split.copy()

    if "patient_id" not in df_ready.columns:
        raise ValueError("mt_patients_regression_ready must contain 'patient_id'.")
    if "patient_id" not in df_split.columns:
        raise ValueError("mt_patients_split must contain 'patient_id'.")
    if "split" not in df_split.columns:
        raise ValueError("mt_patients_split must contain 'split' column.")

    # Merge
    df = df_ready.merge(
        df_split[["patient_id", "split"]],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    # Outcome: good_mRS = 1 if mrs90 <= 2
    if "mrs90" not in df.columns:
        raise ValueError("mt_patients_regression_ready must contain 'mrs90'.")

    df = df[df["mrs90"].notna()].copy()
    df["mrs90"] = df["mrs90"].astype("Int64")
    df["good_mRS"] = (df["mrs90"] <= 2).astype("Int8")

    # TICI → tici_2b3 (if not already present)
    if "tici_2b3" not in df.columns and "tici" in df.columns:
        tici = df["tici"].astype(str).str.strip().str.lower()
        tici_2b3 = pd.Series(np.nan, index=df.index, dtype="float")
        tici_2b3.loc[tici.isin(["2", "2b", "2c", "3"])] = 1.0
        tici_2b3.loc[tici.isin(["0", "1", "2a"])] = 0.0
        df["tici_2b3"] = tici_2b3

    # puncture_to_recan_min (if missing)
    if "puncture_to_recan_min" not in df.columns:
        if {"onset_to_recan_min", "onset_to_puncture_min"}.issubset(df.columns):
            df["puncture_to_recan_min"] = (
                df["onset_to_recan_min"] - df["onset_to_puncture_min"]
            )
        else:
            df["puncture_to_recan_min"] = np.nan

    # --- predictor sets (as in your script) ---
    pre_predictors = [
         "age",
  "sex",
  "aspects",
  "ivt_given",
  "onset_to_ivt_min",
  "onset_to_puncture_min",
  "admission_nihss",
  "mrs_before",
  "statins_before",
  "hypertension",
  "diabetes",
  "hyperlipidemia",
  "smoking",
  "alcohol_abuse",
  "bmi",
  "arrhythmia",
  "tia_before",
  "cmp_before",
  "heart_condition",
  "systolic_bp",
  "diastolic_bp",
  "glycemia",
  "cholesterol",
    ]
    post_extra = [
        "puncture_to_recan_min",
        "tici_2b3",
        "procedure_duration",
        "cerebral_edema",
        "sich",
        # nihss_24h intentionally removed in your current modeling
    ]

    # keep only available columns
    pre_predictors = [p for p in pre_predictors if p in df.columns]
    post_predictors = pre_predictors + [p for p in post_extra if p in df.columns]

    # Temporal split
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    # Drop zero-variance predictors based on TRAIN ONLY
    def drop_zero_variance(df_train_sub: pd.DataFrame,
                           predictors: List[str],
                           label: str) -> List[str]:
        nunique = df_train_sub[predictors].nunique()
        zero_var = nunique[nunique <= 1].index.tolist()
        if zero_var:
            print(f"[{label}] Dropping zero-variance predictors on TRAIN:", zero_var)
        clean_preds = [p for p in predictors if p not in zero_var]
        return clean_preds

    pre_predictors_final = drop_zero_variance(df_train, pre_predictors, "PRE")
    post_predictors_final = drop_zero_variance(df_train, post_predictors, "POST")

    # Build final train / test DataFrames (no dropna: imputed dataset)
    pre_train_df = df_train[["good_mRS"] + pre_predictors_final].copy()
    pre_test_df = df_test[["good_mRS"] + pre_predictors_final].copy()

    post_train_df = df_train[["good_mRS"] + post_predictors_final].copy()
    post_test_df = df_test[["good_mRS"] + post_predictors_final].copy()

    return (
        pre_train_df,
        pre_test_df,
        post_train_df,
        post_test_df,
        pre_predictors_final,
        post_predictors_final,
    )


# ============================================================
# 2) STATS: FIT MODELS & COEFFICIENT TABLES
# ============================================================

def fit_logit_models(
    pre_train_df: pd.DataFrame,
    post_train_df: pd.DataFrame,
    pre_predictors: List[str],
    post_predictors: List[str],
):
    """
    Fit PRE and POST statsmodels Logit on TRAIN data only (age linear).
    """
    formula_pre = "good_mRS ~ " + " + ".join(pre_predictors)
    formula_post = "good_mRS ~ " + " + ".join(post_predictors)

    model_pre = smf.logit(formula_pre, data=pre_train_df).fit(disp=False)
    model_post = smf.logit(formula_post, data=post_train_df).fit(disp=False)

    return model_pre, model_post


def coef_table(model: sm.discrete.discrete_model.BinaryResults) -> pd.DataFrame:
    ci = model.conf_int()
    ci.columns = ["ci_low", "ci_high"]
    df_coef = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "z": model.tvalues,
        "p_value": model.pvalues,
    })
    df_coef = pd.concat([df_coef, ci], axis=1)
    return df_coef


# ============================================================
# 2b) AGE GROUP ORs (decades)
# ============================================================

def age_group_or_table(
    model: sm.discrete.discrete_model.BinaryResults,
    model_label: str,
    age_breaks: List[Tuple[int, int]],
    ref_group: Tuple[int, int],
) -> pd.DataFrame:
    """
    Compute ORs for age groups (decades) vs a reference decade
    using the linear age coefficient:

      OR(age_mid vs ref_mid) = exp(beta_age * (age_mid - ref_mid))

    95% CI derived from the CI of beta_age.
    """
    params = model.params
    ci = model.conf_int()

    if "age" not in params.index:
        # age not in model (e.g. dropped) -> return empty
        return pd.DataFrame()

    beta_age = float(params["age"])
    ci_low_age = float(ci.loc["age", 0])
    ci_high_age = float(ci.loc["age", 1])

    ref_low, ref_high = ref_group
    ref_mid = 0.5 * (ref_low + ref_high)

    rows: List[Dict[str, Any]] = []

    for low, high in age_breaks:
        mid = 0.5 * (low + high)
        delta = mid - ref_mid

        log_or = beta_age * delta
        or_val = float(np.exp(log_or))
        or_low = float(np.exp(ci_low_age * delta))
        or_high = float(np.exp(ci_high_age * delta))

        rows.append({
            "model_label": model_label,
            "age_group": f"{low}-{high}",
            "age_low": low,
            "age_high": high,
            "age_mid": mid,
            "ref_age_low": ref_low,
            "ref_age_high": ref_high,
            "ref_age_mid": ref_mid,
            "OR": or_val,
            "CI_low": or_low,
            "CI_high": or_high,
        })

    return pd.DataFrame(rows)


# ============================================================
# 3) ROC, CALIBRATION, DCA, METRICS
# ============================================================

def plot_roc(y_true, y_pred, label, out_path):
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_pred)

    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC — {label}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    thr_padded = np.append(thr, np.nan) if len(thr) < len(fpr) else thr
    roc_df = pd.DataFrame({
        "model": label,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thr_padded,
        "auc": auc_val,
    })
    return roc_df, auc_val


def plot_calibration(y_true, y_pred, label, out_path, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy="quantile"
    )
    plt.figure(figsize=(4, 4))
    plt.plot(prob_pred, prob_true, "o-", label=label)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed proportion good mRS")
    plt.title(f"Calibration — {label}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    calib_df = pd.DataFrame({
        "model": label,
        "bin": np.arange(1, len(prob_true) + 1),
        "mean_predicted": prob_pred,
        "fraction_of_positives": prob_true,
    })
    return calib_df


def dca_curve(y_true, y_pred, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)
    N = len(y_true)
    prevalence = y_true.mean()

    rows = []
    for pt in thresholds:
        pred_pos = (y_pred >= pt)
        TP = np.sum((pred_pos == 1) & (y_true == 1))
        FP = np.sum((pred_pos == 1) & (y_true == 0))
        nb_model = (TP / N) - (FP / N) * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({
            "threshold": pt,
            "nb_model": nb_model,
            "nb_treat_all": nb_all,
            "nb_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def plot_dca(df_dca, label, out_path):
    plt.figure(figsize=(4, 4))
    plt.plot(df_dca["threshold"], df_dca["nb_model"], label=label)
    plt.plot(df_dca["threshold"], df_dca["nb_treat_all"], "--", label="Treat all")
    plt.plot(df_dca["threshold"], df_dca["nb_treat_none"], ":", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"DCA — {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# 4) PDP, AGE SPLINES & HEATMAP
# ============================================================

def age_pdp(model, data, predictors, label, out_path):
    if "age" not in data.columns:
        return None, None

    age = data["age"].dropna()
    if age.empty:
        return None, None

    a_min, a_max = np.percentile(age, [5, 95])
    grid = np.linspace(a_min, a_max, 80)

    med = _typical_row(data, predictors)
    preds = []
    for val in grid:
        row = med.copy()
        row["age"] = val
        df_row = pd.DataFrame([row])[predictors]
        preds.append(model.predict(df_row)[0])

    plt.figure(figsize=(6, 4))
    plt.plot(grid, preds)
    plt.xlabel("Age")
    plt.ylabel("Predicted Pr(good mRS)")
    plt.title(f"PDP age — {label} (linear age)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return grid, np.array(preds)


def feature_pdp(model, data, predictors, feature, label, out_path):
    if feature not in predictors or feature not in data.columns:
        return

    ser = data[feature].dropna()
    if ser.empty:
        return

    # Build grid depending on type
    if is_numeric_dtype(ser):
        f_min, f_max = np.percentile(ser, [5, 95])
        if f_min == f_max:
            return
        grid = np.linspace(f_min, f_max, 80)
    else:
        grid = np.sort(ser.unique())

    med = _typical_row(data, predictors)
    preds = []
    for val in grid:
        row = med.copy()
        row[feature] = val
        df_row = pd.DataFrame([row])[predictors]
        preds.append(model.predict(df_row)[0])

    plt.figure(figsize=(6, 4))
    plt.plot(grid, preds)
    plt.xlabel(feature)
    plt.ylabel("Predicted Pr(good mRS)")
    plt.title(f"PDP — {feature} ({label})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fit_age_spline_model(
    train_df: pd.DataFrame,
    predictors: List[str],
    df_spline: int = 4,
):
    """
    Fit a logistic model where age is modeled as a restricted cubic spline
    and all other predictors stay linear. Used only for age curves
    (does not change the main PRE/POST models).
    """
    if "age" not in predictors or "age" not in train_df.columns:
        return None

    other_preds = [p for p in predictors if p != "age"]
    rhs_terms = [f"bs(age, df={df_spline}, include_intercept=False)"] + other_preds
    formula = "good_mRS ~ " + " + ".join(rhs_terms)
    model = smf.logit(formula, data=train_df).fit(disp=False)
    return model


def age_spline_curve(model, data, predictors, label, out_path):
    """
    Generate age curve from a spline-based age model, using the same
    'typical' patient pattern as PDP.
    """
    if model is None or "age" not in data.columns:
        return None, None

    age = data["age"].dropna()
    if age.empty:
        return None, None

    a_min, a_max = np.percentile(age, [5, 95])
    grid = np.linspace(a_min, a_max, 80)

    med = _typical_row(data, predictors)
    rows = []
    for val in grid:
        row = med.copy()
        row["age"] = val
        rows.append(row)
    df_grid = pd.DataFrame(rows)[predictors]

    preds = model.predict(df_grid)

    plt.figure(figsize=(6, 4))
    plt.plot(grid, preds)
    plt.xlabel("Age")
    plt.ylabel("Predicted Pr(good mRS)")
    plt.title(f"Spline age curve — {label}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return grid, np.array(preds)


def post_age_nihss_heatmap(model_post, post_df, post_predictors, out_path):
    if "age" not in post_df.columns or "admission_nihss" not in post_df.columns:
        return

    ages = np.linspace(
        np.percentile(post_df["age"], 5),
        np.percentile(post_df["age"], 95),
        50,
    )
    nihss = np.linspace(
        np.percentile(post_df["admission_nihss"], 5),
        np.percentile(post_df["admission_nihss"], 95),
        50,
    )

    med = _typical_row(post_df, post_predictors)
    Z = np.zeros((len(nihss), len(ages)))
    for i, nh in enumerate(nihss):
        for j, a in enumerate(ages):
            row = med.copy()
            row["age"] = a
            row["admission_nihss"] = nh
            df_row = pd.DataFrame([row])[post_predictors]
            Z[i, j] = model_post.predict(df_row)[0]

    plt.figure(figsize=(6, 4))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[ages[0], ages[-1], nihss[0], nihss[-1]],
        cmap="viridis",
    )
    plt.colorbar(im, label="Pr(good mRS)")
    plt.xlabel("Age")
    plt.ylabel("Admission NIHSS")
    plt.title("POST full model: Age × NIHSS heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# 5) FOREST PLOTS
# ============================================================

def forest_plot(coef_df: pd.DataFrame, title: str, out_path: str) -> None:
    df = coef_df.copy()
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "term"}, inplace=True)
    else:
        df["term"] = df.index

    df = df[df["term"] != "Intercept"]
    df["OR"] = np.exp(df["coef"])
    df["CI_low_OR"] = np.exp(df["ci_low"])
    df["CI_high_OR"] = np.exp(df["ci_high"])
    df = df.sort_values("OR", ascending=False)

    y = np.arange(len(df))
    plt.figure(figsize=(6, 0.5 * len(df) + 2))
    plt.errorbar(
        df["OR"], y,
        xerr=[df["OR"] - df["CI_low_OR"], df["CI_high_OR"] - df["OR"]],  # type: ignore[arg-type]
        fmt="o", capsize=4,
    )
    plt.yticks(y, df["term"])
    plt.axvline(1, color="gray", linestyle="--")
    plt.xlabel("Odds Ratio (log scale)")
    plt.xscale("log")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# 6) SHAP-LIKE GLOBAL IMPORTANCE
# ============================================================

def shap_like_importance(
    model,
    data: pd.DataFrame,
    predictors: List[str],
    csv_path: str,
    fig_path: str,
    label: str,
) -> pd.DataFrame:
    params = model.params
    used_preds = [p for p in predictors if p in params.index]
    X = data[used_preds].copy()
    X_centered = X - X.mean(axis=0)
    betas = params[used_preds].values
    phi = X_centered.values * betas
    mean_abs = np.mean(np.abs(phi), axis=0)
    imp_df = pd.DataFrame({
        "feature": used_preds,
        "mean_abs_shap_like": mean_abs,
    }).sort_values("mean_abs_shap_like", ascending=False)

    imp_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.barh(imp_df["feature"], imp_df["mean_abs_shap_like"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP-like contribution| (logit scale)")
    plt.title(f"Global importance (SHAP-like) — {label}")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return imp_df


# ============================================================
# 7) TRUE SHAP (sklearn LogisticRegression)
# ============================================================

def run_real_shap(
    pre_train_df: pd.DataFrame,
    post_train_df: pd.DataFrame,
    pre_predictors: List[str],
    post_predictors: List[str],
):
    # Encode categoricals for sklearn
    X_pre = _encoded_numeric_matrix(pre_train_df, pre_predictors)
    y_pre = pre_train_df["good_mRS"].values

    clf_pre = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
    )
    clf_pre.fit(X_pre, y_pre)

    X_post = _encoded_numeric_matrix(post_train_df, post_predictors)
    y_post = post_train_df["good_mRS"].values
    clf_post = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
    )
    clf_post.fit(X_post, y_post)

    def shap_for_model(
        clf,
        X,
        feature_names,
        label_prefix,
        max_background=300,
        max_eval=1000,
        color_feature=None,
        out_dir="figures/shap",
    ):
        if X.shape[0] > max_eval:
            idx_eval = np.random.choice(X.shape[0], size=max_eval, replace=False)
            X_eval = X[idx_eval]
        else:
            X_eval = X

        if X.shape[0] > max_background:
            idx_bg = np.random.choice(X.shape[0], size=max_background, replace=False)
            X_bg = X[idx_bg]
        else:
            X_bg = X

        try:
            explainer = shap.LinearExplainer(
                clf, X_bg, feature_perturbation="interventional"
            )
            shap_values = explainer.shap_values(X_eval)
        except Exception:
            explainer = shap.Explainer(clf, X_bg)
            shap_values = explainer(X_eval).values

        if isinstance(shap_values, list):
            shap_arr = np.array(shap_values[-1])
        else:
            shap_arr = np.array(shap_values)

        base = label_prefix.lower()

        mean_abs = np.mean(np.abs(shap_arr), axis=0)
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)
        csv_path = os.path.join(BASE_OUT, "tables", f"shap_importance_{base}.csv")
        imp_df.to_csv(csv_path, index=False)

        shap.summary_plot(shap_arr, X_eval, feature_names=feature_names, show=False)
        plt.title(f"SHAP summary — {label_prefix}")
        sum_png = os.path.join(BASE_OUT, out_dir, f"shap_{base}_summary.png")
        plt.gcf().savefig(sum_png, bbox_inches="tight")
        plt.close()

        shap.summary_plot(
            shap_arr, X_eval, feature_names=feature_names,
            plot_type="bar", show=False
        )
        plt.title(f"SHAP global importance — {label_prefix}")
        bar_png = os.path.join(BASE_OUT, out_dir, f"shap_{base}_bar.png")
        plt.gcf().savefig(bar_png, bbox_inches="tight")
        plt.close()

        if "age" in feature_names:
            if color_feature is not None and color_feature in feature_names:
                shap.dependence_plot(
                    "age", shap_arr, X_eval,
                    interaction_index=feature_names.index(color_feature),
                    feature_names=feature_names, show=False,
                )
            else:
                shap.dependence_plot(
                    "age", shap_arr, X_eval,
                    feature_names=feature_names, show=False,
                )
            plt.title(f"SHAP dependence — age ({label_prefix})")
            dep_png = os.path.join(
                BASE_OUT, out_dir, f"shap_age_dependence_{base}.png"
            )
            plt.gcf().savefig(dep_png, bbox_inches="tight")
            plt.close()
        else:
            dep_png = None

        return {
            "importance_csv": csv_path,
            "summary_png": sum_png,
            "bar_png": bar_png,
            "age_dependence_png": dep_png,
            "shap_values": shap_arr,
            "X_eval": X_eval,
            "feature_names": feature_names,
        }

    pre_info = shap_for_model(
        clf_pre, X_pre, pre_predictors,
        label_prefix="PRE_full",
        color_feature="heart_condition",
        out_dir="figures/shap",
    )
    post_info = shap_for_model(
        clf_post, X_post, post_predictors,
        label_prefix="POST_full",
        color_feature="onset_to_puncture_min",
        out_dir="figures/shap",
    )
    return pre_info, post_info


def build_shap_age_profile(
    shap_info: Dict[str, Any],
    label: str,
    out_csv: str,
    n_bins: int = 8,
) -> pd.DataFrame:
    """
    Collapse SHAP(age) into a smooth age profile by binning age and
    taking mean/median SHAP per bin (log-odds scale).
    """
    shap_vals = shap_info.get("shap_values")
    X_eval = shap_info.get("X_eval")
    feature_names = shap_info.get("feature_names")

    if shap_vals is None or X_eval is None or feature_names is None:
        return pd.DataFrame()

    if "age" not in feature_names:
        return pd.DataFrame()

    age_idx = feature_names.index("age")
    age_vals = X_eval[:, age_idx]
    shap_age = shap_vals[:, age_idx]

    df = pd.DataFrame({"age": age_vals, "shap": shap_age}).dropna()
    if df.empty:
        return df

    q_low, q_high = df["age"].quantile([0.05, 0.95])
    df = df[(df["age"] >= q_low) & (df["age"] <= q_high)]
    if df.empty:
        return df

    bins = np.linspace(q_low, q_high, n_bins + 1)
    df["age_bin"] = pd.cut(df["age"], bins=bins, include_lowest=True)
    grouped = df.groupby("age_bin").agg(
        age_mid=("age", "median"),
        shap_mean=("shap", "mean"),
        shap_median=("shap", "median"),
    ).reset_index(drop=True)
    grouped["model_label"] = label
    grouped.to_csv(out_csv, index=False)
    return grouped


def plot_combined_age_effect(
    label: str,
    age_grid_pdp: np.ndarray | None,
    age_pdp_vals: np.ndarray | None,
    age_grid_spline: np.ndarray | None,
    age_spline_vals: np.ndarray | None,
    shap_age_df: pd.DataFrame,
    out_path: str,
):
    """
    Single figure with 2 panels:
      - PDP (linear age) + spline PDP (risk vs age)
      - SHAP median (log-odds) vs age
    """
    if age_grid_pdp is None and age_grid_spline is None:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # Panel 1: predicted probability
    if age_grid_pdp is not None and age_pdp_vals is not None:
        ax1.plot(age_grid_pdp, age_pdp_vals, label="PDP (linear age)")
    if age_grid_spline is not None and age_spline_vals is not None:
        ax1.plot(age_grid_spline, age_spline_vals, label="Spline age PDP")
    ax1.set_ylabel("Pr(good mRS)")
    ax1.set_title(f"Age effect — predicted risk ({label})")
    ax1.legend()
    ax1.grid(True)

    # Panel 2: SHAP median
    if shap_age_df is not None and not shap_age_df.empty:
        ax2.plot(
            shap_age_df["age_mid"],
            shap_age_df["shap_median"],
            marker="o",
            label="SHAP median (log-odds)",
        )
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Age (years)")
    ax2.set_ylabel("SHAP (log-odds)")
    ax2.set_title(f"Age effect — SHAP contributions ({label})")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_age_effect_summary(
    age_or_df: pd.DataFrame,
    shap_age_pre: pd.DataFrame,
    shap_age_post: pd.DataFrame,
    age_grid_pre_pdp: np.ndarray | None,
    age_vals_pre_pdp: np.ndarray | None,
    age_grid_post_pdp: np.ndarray | None,
    age_vals_post_pdp: np.ndarray | None,
) -> pd.DataFrame:
    """
    Table combining:
      - ORs by age group (from linear model)
      - PDP-based predicted risk at age midpoints
      - SHAP median at age midpoints
    for PRE and POST models.
    """
    rows: List[Dict[str, Any]] = []

    def _interp(x_grid, y_vals, x0):
        if x_grid is None or y_vals is None:
            return np.nan
        if len(x_grid) == 0:
            return np.nan
        return float(np.interp(x0, x_grid, y_vals))

    for model_label, shap_df, age_grid, age_pdp_vals in [
        ("PRE full", shap_age_pre, age_grid_pre_pdp, age_vals_pre_pdp),
        ("POST full", shap_age_post, age_grid_post_pdp, age_vals_post_pdp),
    ]:
        if age_or_df.empty:
            continue
        sub_or = age_or_df[age_or_df["model_label"] == model_label]
        if sub_or.empty:
            continue

        for _, r in sub_or.iterrows():
            age_mid = float(r["age_mid"])
            pdp_prob = _interp(age_grid, age_pdp_vals, age_mid)

            if shap_df is not None and not shap_df.empty:
                shap_med = float(np.interp(
                    age_mid,
                    shap_df["age_mid"].values,
                    shap_df["shap_median"].values,
                ))
            else:
                shap_med = np.nan

            rows.append({
                "model_label": model_label,
                "age_group": r["age_group"],
                "age_low": r["age_low"],
                "age_high": r["age_high"],
                "age_mid": age_mid,
                "OR_vs_ref": r["OR"],
                "OR_CI_low": r["CI_low"],
                "OR_CI_high": r["CI_high"],
                "PDP_prob": pdp_prob,
                "SHAP_median": shap_med,
            })

    return pd.DataFrame(rows)


# ============================================================
# 8) CROSS-VALIDATION (on TRAIN only)
# ============================================================

def run_cv(
    pre_train_df: pd.DataFrame,
    post_train_df: pd.DataFrame,
    pre_predictors: List[str],
    post_predictors: List[str],
):
    rows: List[Dict[str, Any]] = []

    def cv_for_model(X, y, key, label):
        for r in range(CV_REPEATS):
            skf = StratifiedKFold(
                n_splits=CV_FOLDS,
                shuffle=True,
                random_state=CV_RANDOM_SEED + r,
            )
            fold = 0
            for train_idx, val_idx in skf.split(X, y):
                fold += 1
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                clf = LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                )
                clf.fit(X_train, y_train)
                p_val = clf.predict_proba(X_val)[:, 1]
                p_clip = np.clip(p_val, 1e-15, 1 - 1e-15)

                auc = roc_auc_score(y_val, p_clip)
                brier = brier_score_loss(y_val, p_clip)
                ll = log_loss(y_val, p_clip)

                logit_p = np.log(p_clip / (1 - p_clip))
                X_cal = sm.add_constant(logit_p)
                try:
                    cal_model = sm.Logit(y_val, X_cal).fit(disp=False)
                    cal_intercept = cal_model.params.iloc[0]
                    cal_slope = (
                        cal_model.params.iloc[1]
                        if len(cal_model.params) > 1 else np.nan
                    )
                except Exception:
                    cal_intercept = np.nan
                    cal_slope = np.nan

                rows.append({
                    "model_key": key,
                    "model_label": label,
                    "repeat": r + 1,
                    "fold": fold,
                    "n_val": len(y_val),
                    "auc": auc,
                    "brier": brier,
                    "log_loss": ll,
                    "cal_intercept": cal_intercept,
                    "cal_slope": cal_slope,
                })

    # Encode categoricals before feeding into sklearn
    X_pre = _encoded_numeric_matrix(pre_train_df, pre_predictors)
    y_pre = pre_train_df["good_mRS"].values
    cv_for_model(X_pre, y_pre, "pre_full", "PRE full")

    X_post = _encoded_numeric_matrix(post_train_df, post_predictors)
    y_post = post_train_df["good_mRS"].values
    cv_for_model(X_post, y_post, "post_full", "POST full")

    cv_df = pd.DataFrame(rows)

    summary_rows = []
    for key, label in [("pre_full", "PRE full"), ("post_full", "POST full")]:
        sub = cv_df[cv_df["model_key"] == key]
        if sub.empty:
            continue
        for metric in ["auc", "brier", "log_loss", "cal_intercept", "cal_slope"]:
            summary_rows.append({
                "model_key": key,
                "model_label": label,
                "metric": metric,
                "mean": sub[metric].mean(),
                "std": sub[metric].std(),
            })
    cv_summary_df = pd.DataFrame(summary_rows)

    plt.figure(figsize=(7, 5))
    for key, label in [("pre_full", "PRE full"), ("post_full", "POST full")]:
        sub = cv_df[cv_df["model_key"] == key]
        if sub.empty:
            continue
        plt.hist(sub["auc"], bins=15, alpha=0.5, label=label)
    plt.xlabel("AUC (CV folds)")
    plt.ylabel("Count")
    plt.title(f"Cross-validated AUC (k={CV_FOLDS} x r={CV_REPEATS})")
    plt.legend()
    plt.tight_layout()
    auc_hist_path = os.path.join(
        BASE_OUT, "figures/cv", "cv_auc_distribution_pre_post_full.png"
    )
    plt.savefig(auc_hist_path)
    plt.close()

    return cv_df, cv_summary_df


# ============================================================
# 9) VIF & NRI/IDI
# ============================================================

def compute_vif(
    df: pd.DataFrame,
    predictors: List[str],
    label: str,
    out_csv: str,
) -> pd.DataFrame:
    """
    Compute VIF for each predictor.

    - Any non-numeric columns (e.g. 'CAD', 'AF', etc.) are converted
      to categorical codes before VIF calculation.
    """

    X = df[predictors].copy()

    # Convert non-numeric columns to categorical codes
    for col in X.columns:
        if not is_numeric_dtype(X[col]):
            cat = X[col].astype("category")
            codes = cat.cat.codes.replace(-1, np.nan)
            X[col] = codes.astype(float)

    # Now everything should be numeric
    X = sm.add_constant(X, has_constant="add")
    cols = X.columns

    vif_rows = []
    # skip intercept (index 0)
    for i in range(1, X.shape[1]):
        v = variance_inflation_factor(X.values, i)
        vif_rows.append({"feature": cols[i], "VIF": v})

    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(out_csv, index=False)
    print(f"[{label}] VIF saved to {out_csv}")
    return vif_df


def nri_idi(y, p_old, p_new) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)

    events = (y == 1)
    nonevents = (y == 0)

    discr_old = p_old[events].mean() - p_old[nonevents].mean()
    discr_new = p_new[events].mean() - p_new[nonevents].mean()
    idi = discr_new - discr_old

    up_e = np.mean(p_new[events] > p_old[events])
    down_e = np.mean(p_new[events] < p_old[events])
    nri_events = up_e - down_e

    down_ne = np.mean(p_new[nonevents] < p_old[nonevents])
    up_ne = np.mean(p_new[nonevents] > p_old[nonevents])
    nri_nonevents = down_ne - up_ne

    nri_total = nri_events + nri_nonevents

    return {
        "discrimination_old": float(discr_old),
        "discrimination_new": float(discr_new),
        "IDI": float(idi),
        "NRI_events": float(nri_events),
        "NRI_nonevents": float(nri_nonevents),
        "NRI_total": float(nri_total),
    }


# ============================================================
# 10) MASTER NODE FOR KEDRO
# ============================================================

def run_mt_stroke_regression_pipeline(
    mt_patients_regression_ready: pd.DataFrame,
    mt_patients_split: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,  # coef_pre
    pd.DataFrame,  # coef_post
    pd.DataFrame,  # fit_stats_df
    pd.DataFrame,  # roc_values
    pd.DataFrame,  # calib_values
    pd.DataFrame,  # dca_values
    pd.DataFrame,  # shap_like_pre
    pd.DataFrame,  # shap_like_post
    pd.DataFrame,  # cv_df
    pd.DataFrame,  # cv_summary_df
    pd.DataFrame,  # vif_pre
    pd.DataFrame,  # vif_post
    pd.DataFrame,  # nri_idi_df
]:
    """
    Kedro node:
      - merges regression_ready + split
      - builds PRE + POST datasets with temporal train/test split
      - fits models on TRAIN only & generates all diagnostics
      - writes figures + Excel to BASE_OUT
      - returns main tables as DataFrames for catalog
    """
    np.random.seed(42)
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    _ensure_dirs()

    # 1) Build datasets (TRIPOD-friendly)
    (
        pre_train_df,
        pre_test_df,
        post_train_df,
        post_test_df,
        pre_predictors,
        post_predictors,
    ) = build_pre_post_datasets(
        mt_patients_regression_ready, mt_patients_split
    )

    # 2) Fit statsmodels Logit on TRAIN ONLY (linear age)
    model_pre, model_post = fit_logit_models(
        pre_train_df, post_train_df, pre_predictors, post_predictors
    )
    coef_pre = coef_table(model_pre)
    coef_post = coef_table(model_post)

    # 2b) Age group ORs (decades) from linear age term
    age_breaks = [(40, 49), (50, 59), (60, 69), (70, 79), (80, 89)]
    ref_group = (60, 69)
    age_or_pre = age_group_or_table(model_pre, "PRE full", age_breaks, ref_group)
    age_or_post = age_group_or_table(model_post, "POST full", age_breaks, ref_group)
    if not age_or_pre.empty or not age_or_post.empty:
        age_or_df = pd.concat([age_or_pre, age_or_post], ignore_index=True)
    else:
        age_or_df = pd.DataFrame()
    if not age_or_df.empty:
        age_or_csv = os.path.join(BASE_OUT, "tables", "age_group_ORs_pre_post.csv")
        age_or_df.to_csv(age_or_csv, index=False)

    # 3) VIF (on TRAIN)
    vif_pre = compute_vif(
        pre_train_df, pre_predictors,
        label="PRE_train",
        out_csv=os.path.join(BASE_OUT, "tables", "vif_pre_full_train.csv"),
    )
    vif_post = compute_vif(
        post_train_df, post_predictors,
        label="POST_train",
        out_csv=os.path.join(BASE_OUT, "tables", "vif_post_full_train.csv"),
    )

    thresholds = np.linspace(0.05, 0.75, 15)

    metrics_rows: List[Dict[str, Any]] = []
    roc_values_list: List[pd.DataFrame] = []
    calib_values_list: List[pd.DataFrame] = []
    dca_values_list: List[pd.DataFrame] = []

    # 3a) TRAIN performance
    for name, model, data, fig_subdir, tag in [
        ("PRE full", model_pre, pre_train_df, "pre", "train"),
        ("POST full", model_post, post_train_df, "post", "train"),
    ]:
        y = data["good_mRS"].values
        p = model.predict(data)
        label = f"{name} ({tag})"

        # ROC
        roc_path = os.path.join(
            BASE_OUT, f"figures/{fig_subdir}", f"roc_{fig_subdir}_{tag}.png"
        )
        roc_df, auc_val = plot_roc(y, p, label, roc_path)
        roc_df["dataset"] = tag
        roc_values_list.append(roc_df)

        # Calibration
        cal_path = os.path.join(
            BASE_OUT, f"figures/{fig_subdir}", f"calibration_{fig_subdir}_{tag}.png"
        )
        calib_df = plot_calibration(y, p, label, cal_path)
        calib_df["dataset"] = tag
        calib_values_list.append(calib_df)

        # DCA
        dca_df = dca_curve(y, p, thresholds)
        dca_df["model"] = name
        dca_df["dataset"] = tag
        dca_values_list.append(dca_df)
        dca_path = os.path.join(
            BASE_OUT, f"figures/{fig_subdir}", f"dca_{fig_subdir}_{tag}.png"
        )
        plot_dca(dca_df, label, dca_path)

        # Metrics
        p_clip = np.clip(p, 1e-15, 1 - 1e-15)
        brier = brier_score_loss(y, p_clip)
        ll = log_loss(y, p_clip)
        logit_p = np.log(p_clip / (1 - p_clip))
        X_cal = sm.add_constant(logit_p)
        try:
            cal_model = sm.Logit(y, X_cal).fit(disp=False)
            cal_intercept = cal_model.params.iloc[0]
            cal_slope = (
                cal_model.params.iloc[1]
                if len(cal_model.params) > 1 else np.nan
            )
        except Exception:
            cal_intercept = np.nan
            cal_slope = np.nan

        metrics_rows.append({
            "model": name,
            "dataset": tag,
            "n": len(y),
            "auc": auc_val,
            "brier": brier,
            "log_loss": ll,
            "cal_intercept": cal_intercept,
            "cal_slope": cal_slope,
        })

    # 3b) TEST performance (temporal validation)
    for name, model, data, fig_subdir, tag in [
        ("PRE full", model_pre, pre_test_df, "pre", "test"),
        ("POST full", model_post, post_test_df, "post", "test"),
    ]:
        if data.empty:
            continue

        y = data["good_mRS"].values
        p = model.predict(data)
        label = f"{name} ({tag})"

        # ROC
        roc_path = os.path.join(
            BASE_OUT, f"figures/{fig_subdir}", f"roc_{fig_subdir}_{tag}.png"
        )
        roc_df, auc_val = plot_roc(y, p, label, roc_path)
        roc_df["dataset"] = tag
        roc_values_list.append(roc_df)

        # Calibration
        cal_path = os.path.join(
            BASE_OUT, f"figures/{fig_subdir}", f"calibration_{fig_subdir}_{tag}.png"
        )
        calib_df = plot_calibration(y, p, label, cal_path)
        calib_df["dataset"] = tag
        calib_values_list.append(calib_df)

        # DCA
        dca_df = dca_curve(y, p, thresholds)
        dca_df["model"] = name
        dca_df["dataset"] = tag
        dca_values_list.append(dca_df)
        dca_path = os.path.join(
            BASE_OUT, f"figures/{fig_subdir}", f"dca_{fig_subdir}_{tag}.png"
        )
        plot_dca(dca_df, label, dca_path)

        # Metrics
        p_clip = np.clip(p, 1e-15, 1 - 1e-15)
        brier = brier_score_loss(y, p_clip)
        ll = log_loss(y, p_clip)
        logit_p = np.log(p_clip / (1 - p_clip))
        X_cal = sm.add_constant(logit_p)
        try:
            cal_model = sm.Logit(y, X_cal).fit(disp=False)
            cal_intercept = cal_model.params.iloc[0]
            cal_slope = (
                cal_model.params.iloc[1]
                if len(cal_model.params) > 1 else np.nan
            )
        except Exception:
            cal_intercept = np.nan
            cal_slope = np.nan

        metrics_rows.append({
            "model": name,
            "dataset": tag,
            "n": len(y),
            "auc": auc_val,
            "brier": brier,
            "log_loss": ll,
            "cal_intercept": cal_intercept,
            "cal_slope": cal_slope,
        })

    fit_stats_df = pd.DataFrame(metrics_rows)
    roc_values = pd.concat(roc_values_list, ignore_index=True)
    calib_values = pd.concat(calib_values_list, ignore_index=True)
    dca_values = pd.concat(dca_values_list, ignore_index=True)

    # 4) NRI/IDI: POST vs PRE on TEST dataset (common patients, TRIPOD-style)
    if not post_test_df.empty:
        y_common = post_test_df["good_mRS"].values
        p_pre_on_test = model_pre.predict(post_test_df[pre_predictors])
        p_post_on_test = model_post.predict(post_test_df[post_predictors])

        nri_idi_res = nri_idi(y_common, p_pre_on_test, p_post_on_test)
        nri_idi_df = pd.DataFrame([nri_idi_res])
        nri_idi_csv = os.path.join(
            BASE_OUT, "tables", "nri_idi_pre_vs_post_test_only.csv"
        )
        nri_idi_df.to_csv(nri_idi_csv, index=False)
    else:
        nri_idi_df = pd.DataFrame(columns=[
            "discrimination_old", "discrimination_new", "IDI",
            "NRI_events", "NRI_nonevents", "NRI_total"
        ])

    # 5) PDPs (on TRAIN) + spline models + spline curves
    age_grid_pre_pdp, age_vals_pre_pdp = age_pdp(
        model_pre, pre_train_df, pre_predictors,
        "PRE full (train)",
        os.path.join(BASE_OUT, "figures/age", "pdp_age_pre_full_train.png"),
    )
    age_grid_post_pdp, age_vals_post_pdp = age_pdp(
        model_post, post_train_df, post_predictors,
        "POST full (train)",
        os.path.join(BASE_OUT, "figures/age", "pdp_age_post_full_train.png"),
    )

    age_spline_model_pre = fit_age_spline_model(pre_train_df, pre_predictors)
    age_spline_model_post = fit_age_spline_model(post_train_df, post_predictors)

    age_grid_pre_spline, age_vals_pre_spline = age_spline_curve(
        age_spline_model_pre,
        pre_train_df,
        pre_predictors,
        "PRE full (train)",
        os.path.join(BASE_OUT, "figures/age", "spline_age_pre_full_train.png"),
    )
    age_grid_post_spline, age_vals_post_spline = age_spline_curve(
        age_spline_model_post,
        post_train_df,
        post_predictors,
        "POST full (train)",
        os.path.join(BASE_OUT, "figures/age", "spline_age_post_full_train.png"),
    )

    for feat in pre_predictors:
        out_path = os.path.join(
            BASE_OUT, "figures/pre", f"pdp_pre_{feat}_train.png"
        )
        feature_pdp(
            model_pre, pre_train_df, pre_predictors,
            feature=feat,
            label="PRE full (train)",
            out_path=out_path,
        )

    for feat in post_predictors:
        out_path = os.path.join(
            BASE_OUT, "figures/post", f"pdp_post_{feat}_train.png"
        )
        feature_pdp(
            model_post, post_train_df, post_predictors,
            feature=feat,
            label="POST full (train)",
            out_path=out_path,
        )

    post_age_nihss_heatmap(
        model_post, post_train_df, post_predictors,
        os.path.join(BASE_OUT, "figures/post", "heatmap_age_nihss_post_train.png"),
    )

    # 6) Forest plots
    forest_plot(
        coef_pre,
        "Forest plot — PRE full model",
        os.path.join(BASE_OUT, "figures/pre", "forest_pre_full.png"),
    )
    forest_plot(
        coef_post,
        "Forest plot — POST full model",
        os.path.join(BASE_OUT, "figures/post", "forest_post_full.png"),
    )

    # 7) SHAP-like importance (on TRAIN)
    shap_like_pre = shap_like_importance(
        model_pre, pre_train_df, pre_predictors,
        csv_path=os.path.join(
            BASE_OUT, "tables", "shap_like_importance_pre_full_train.csv"
        ),
        fig_path=os.path.join(
            BASE_OUT, "figures/age", "shap_like_importance_pre_full_train.png"
        ),
        label="PRE full (train)",
    )
    shap_like_post = shap_like_importance(
        model_post, post_train_df, post_predictors,
        csv_path=os.path.join(
            BASE_OUT, "tables", "shap_like_importance_post_full_train.csv"
        ),
        fig_path=os.path.join(
            BASE_OUT, "figures/age", "shap_like_importance_post_full_train.png"
        ),
        label="POST full (train)",
    )

    # 8) True SHAP (TRAIN only, with encoding)
    pre_shap_info, post_shap_info = run_real_shap(
        pre_train_df, post_train_df, pre_predictors, post_predictors
    )

    # Build age SHAP profiles
    shap_age_pre = build_shap_age_profile(
        pre_shap_info,
        "PRE full (train)",
        os.path.join(BASE_OUT, "tables", "shap_age_profile_pre_full_train.csv"),
    )
    shap_age_post = build_shap_age_profile(
        post_shap_info,
        "POST full (train)",
        os.path.join(BASE_OUT, "tables", "shap_age_profile_post_full_train.csv"),
    )

    # Combined age-effect figures (PDP + spline + SHAP)
    plot_combined_age_effect(
        "PRE full (train)",
        age_grid_pre_pdp,
        age_vals_pre_pdp,
        age_grid_pre_spline,
        age_vals_pre_spline,
        shap_age_pre,
        os.path.join(BASE_OUT, "figures/age", "combined_age_effect_pre_full_train.png"),
    )
    plot_combined_age_effect(
        "POST full (train)",
        age_grid_post_pdp,
        age_vals_post_pdp,
        age_grid_post_spline,
        age_vals_post_spline,
        shap_age_post,
        os.path.join(BASE_OUT, "figures/age", "combined_age_effect_post_full_train.png"),
    )

    # 9) Cross-validation (TRAIN only, with encoding)
    cv_df, cv_summary_df = run_cv(
        pre_train_df, post_train_df, pre_predictors, post_predictors
    )

    # 10) Age effect summary table (OR + PDP + SHAP)
    if "age_or_df" in locals() and not age_or_df.empty:
        age_effect_summary_df = build_age_effect_summary(
            age_or_df,
            shap_age_pre,
            shap_age_post,
            age_grid_pre_pdp,
            age_vals_pre_pdp,
            age_grid_post_pdp,
            age_vals_post_pdp,
        )
        if not age_effect_summary_df.empty:
            age_effect_summary_df.to_csv(
                os.path.join(
                    BASE_OUT, "tables", "age_effect_summary_pre_post.csv"
                ),
                index=False,
            )
    else:
        age_effect_summary_df = pd.DataFrame()

    # 11) Excel summary
    excel_path = os.path.join(
        BASE_OUT, "tables", "mt_final_models_summary_temporal_split.xlsx"
    )
    import xlsxwriter  # noqa: F401
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        coef_pre.to_excel(writer, sheet_name="Coefficients_pre_full")
        coef_post.to_excel(writer, sheet_name="Coefficients_post_full")
        fit_stats_df.to_excel(writer, sheet_name="Model_fit_stats", index=False)
        roc_values.to_excel(writer, sheet_name="ROC_values", index=False)
        calib_values.to_excel(writer, sheet_name="Calibration_values", index=False)
        dca_values.to_excel(writer, sheet_name="DCA_values", index=False)
        shap_like_pre.to_excel(
            writer, sheet_name="SHAP_like_pre_full_train", index=False
        )
        shap_like_post.to_excel(
            writer, sheet_name="SHAP_like_post_full_train", index=False
        )
        cv_df.to_excel(writer, sheet_name="CV_results", index=False)
        cv_summary_df.to_excel(writer, sheet_name="CV_summary", index=False)
        vif_pre.to_excel(writer, sheet_name="VIF_pre_full_train", index=False)
        vif_post.to_excel(writer, sheet_name="VIF_post_full_train", index=False)
        nri_idi_df.to_excel(
            writer, sheet_name="NRI_IDI_PRE_vs_POST_test_only", index=False
        )
        if "age_or_df" in locals() and not age_or_df.empty:
            age_or_df.to_excel(
                writer, sheet_name="Age_group_ORs", index=False
            )
        if age_effect_summary_df is not None and not age_effect_summary_df.empty:
            age_effect_summary_df.to_excel(
                writer, sheet_name="Age_effect_summary", index=False
            )

    print(f"[mt_regression] All done. Excel summary at: {excel_path}")

    return (
        coef_pre,
        coef_post,
        fit_stats_df,
        roc_values,
        calib_values,
        dca_values,
        shap_like_pre,
        shap_like_post,
        cv_df,
        cv_summary_df,
        vif_pre,
        vif_post,
        nri_idi_df,
    )
