from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import shap
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    log_loss,
)

# ============================================================
# GLOBAL SWITCHES FOR EXPERIMENTS
# ============================================================

# Interactions
ADD_INTERACTION_AGE_X_NIHSS = False          # current best: keep True
ADD_INTERACTION_ONSET_TO_PUNCTURE_MIN_X_NIHSS = False    # set True to test
ADD_INTERACTION_AGE_X_ONSET_TO_PUNCTURE_MIN = False    # set True to test

# Monotonic constraint on glycemia
RELAX_GLYCEMIA_CONSTRAINT = True           # set True to make glycemia unconstrained (0)

# ============================================================


def build_mrs90_dataset(
    df_ready: pd.DataFrame,
    df_split: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,  # X_train
    pd.DataFrame,  # y_train
    pd.DataFrame,  # X_test
    pd.DataFrame,  # y_test
    pd.DataFrame,  # train_ids
    pd.DataFrame,  # test_ids
    List[str],     # feature_list
]:
    """
    Merge LightGBM-ready dataset with split labels and
    build a pre-procedural dataset for mRS 0–2 (good) vs 3–6 (poor) at 90 days.
    """

    # -------------------------
    # 1) Merge on patient_id
    # -------------------------
    if "patient_id" not in df_ready.columns:
        raise ValueError("mt_lightgbm_ready must contain 'patient_id'.")

    if "patient_id" not in df_split.columns:
        raise ValueError("mt_patients_split must contain 'patient_id'.")

    if "split" not in df_split.columns:
        raise ValueError("mt_patients_split must contain 'split' column.")

    df = df_ready.merge(df_split[["patient_id", "split"]], on="patient_id", how="inner")

    # -------------------------
    # 2) Restrict to cases with observed mRS90
    # -------------------------
    df = df[df["mrs90"].notna()].copy()

    # -------------------------
    # 3) Outcome = binary mRS 0–2 (1) vs 3–6 (0)
    # -------------------------
    mrs_int = df["mrs90"].astype("Int64")
    y_bin = (mrs_int <= 2).astype("Int8")

    # -------------------------
    # 3b) Add interaction(s)
    # -------------------------
    if ADD_INTERACTION_AGE_X_NIHSS and {"age", "admission_nihss"}.issubset(df.columns):
        df["age_x_nihss"] = df["age"] * df["admission_nihss"]

    if ADD_INTERACTION_ONSET_TO_PUNCTURE_MIN_X_NIHSS and {"onset_to_puncture_min", "admission_nihss"}.issubset(df.columns):
        df["onset_to_puncture_min_x_nihss"] = df["onset_to_puncture_min"] * df["admission_nihss"]
        
    if ADD_INTERACTION_AGE_X_ONSET_TO_PUNCTURE_MIN and {"age", "onset_to_puncture_min"}.issubset(df.columns):
        df["age_x_onset_to_puncture_min"] = df["age"] * df["onset_to_puncture_min"]
    

    # -------------------------
    # 4) Drop non-predictor columns
    # -------------------------
    drop_cols = [
        
        # identifiers / split / raw time stamps
        "patient_id",
        "split",
        "onset_time",
        "onset_date",
        "onset_year",

        # outcomes / post-outcome variables
        "mrs90",
        "mrs90_missing",
        "nihss_7d",
        "nihss_24h",
        "nihss_24h_missing",
        "nihss_7d_missing",
        "sich",
        "non_sich",
        "cerebral_edema",
        "cerebral_edema_missing",

        # procedural / post-treatment features
        "procedure",
        "stent_combo",
        "extraction_system",
        "procedure_duration",
        "tici",
        "tici_success",
        "iat_given",
        "iat_given_missing",
        "onset_to_recan_min",
        "onset_to_recan_min_missing",
        "decompression_surgery",
        "decompression_surgery_missing",
        "anesthesia",
        "anesthesia_missing",
    ]
    
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # -------------------------
    # 5) Encode remaining categorical features as integer codes
    # -------------------------
    cat_like = [
        c
        for c in feature_cols
        if (df[c].dtype == "object")
        or pd.api.types.is_string_dtype(df[c])
        or pd.api.types.is_categorical_dtype(df[c])
    ]

    for c in cat_like:
        df[c] = df[c].astype("category")

    X = df[feature_cols]

    # -------------------------
    # 6) Split into train / test
    # -------------------------
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)

    y_train_df = y_bin.loc[train_mask].reset_index(drop=True).to_frame("mrs90_good")
    y_test_df = y_bin.loc[test_mask].reset_index(drop=True).to_frame("mrs90_good")

    train_ids_df = df.loc[train_mask, "patient_id"].reset_index(drop=True).to_frame("patient_id")
    test_ids_df = df.loc[test_mask, "patient_id"].reset_index(drop=True).to_frame("patient_id")

    return X_train, y_train_df, X_test, y_test_df, train_ids_df, test_ids_df, feature_cols


def train_mrs90_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,  # 1-col DF
) -> Tuple[LGBMClassifier, Dict[str, float]]:
    """
    Train a LightGBM classifier on pre-procedural features
    for mRS 0–2 vs 3–6, with clinically motivated monotone constraints.
    """

    # 1) Target as 1D array
    y = y_train.iloc[:, 0].values

    # 2) Build monotone constraints from column names
    feature_list: List[str] = list(X_train.columns)

    gly_sign = 0 if RELAX_GLYCEMIA_CONSTRAINT else -1

    # Effect is on probability of GOOD outcome (mrs90_good = 1)
    sign_map: Dict[str, int] = {
        # core clinical monotone effects
        "age": -1,                    # older -> lower chance of good outcome
        "admission_nihss": -1,        # more severe stroke -> worse
        "mrs_before": 0,             # worse baseline disability -> worse
        "onset_to_ivt_min": 0,       # longer delay -> worse
        "onset_to_puncture_min": -1,  # longer delay -> worse
        "aspects": 0,                # higher ASPECTS -> better
        "glycemia": gly_sign,         # experiment: constrain (-1) or relax (0)

        # interactions (if present)
        "age_x_nihss": 0,
        "onset_to_puncture_min_x_nihss": 0,
        "age_x_onset_to_puncture_min": 0,
    }

    monotone_constraints: List[int] = [sign_map.get(f, 0) for f in feature_list]

    if len(monotone_constraints) != X_train.shape[1]:
        raise ValueError(
            f"monotone_constraints length {len(monotone_constraints)} "
            f"does not match n_features {X_train.shape[1]}"
        )

    # Tuned hyperparameters (best_params)
    model = LGBMClassifier(
       n_estimators=400,
        learning_rate=0.01,
        num_leaves=31,
        min_child_samples=50,
        subsample=1.0,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=0.0,
        objective="binary",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        monotone_constraints=monotone_constraints,
        enable_categorical=True,
    )

    model.fit(X_train, y)

    proba = model.predict_proba(X_train)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics: Dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "log_loss": float(log_loss(y, proba)),
    }
    return model, metrics


def evaluate_mrs90_lgbm(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,    # 1-col DF
    test_ids: pd.DataFrame,  # 1-col DF
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate the trained LightGBM on the held-out temporal test set.
    Returns metrics + a prediction table.
    """
    y = y_test.iloc[:, 0].values
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics: Dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "log_loss": float(log_loss(y, proba)),
    }

    preds_df = pd.DataFrame(
        {
            "patient_id": test_ids.iloc[:, 0].reset_index(drop=True),
            "y_true": y_test.iloc[:, 0].reset_index(drop=True),
            "y_pred_proba": proba,
            "y_pred_label": pred,
        }
    )

    return metrics, preds_df


def mrs90_shap_importance_by_age_groups(
    shap_values_test: np.ndarray,
    X_test: pd.DataFrame,
    feature_list: List[str],
    thresholds: List[int] = [50, 75, 80, 85],
    age_col: str = "age",
) -> pd.DataFrame:
    """
    Compute SHAP mean(|value|) per feature within age subgroups on TEST set.

    For each threshold T in thresholds, we compute feature importance separately in:
      - age < T
      - age >= T

    Returns a long-format DataFrame with columns:
      ['threshold', 'group', 'n_patients', 'feature', 'mean_abs_shap']
    where group is e.g. '<50' or '≥50'.
    """

    if age_col not in X_test.columns:
        raise ValueError(f"X_test must contain '{age_col}' column for age-stratified SHAP.")

    age = X_test[age_col]
    records: List[Dict[str, Any]] = []

    # Ensure shapes align
    if shap_values_test.shape[0] != X_test.shape[0]:
        raise ValueError(
            f"shap_values_test has {shap_values_test.shape[0]} rows, "
            f"but X_test has {X_test.shape[0]} rows."
        )

    for thr in thresholds:
        mask_lt = (age < thr) & age.notna()
        mask_ge = (age >= thr) & age.notna()

        for label, mask in [("lt", mask_lt), ("ge", mask_ge)]:
            n_group = int(mask.sum())
            if n_group == 0:
                # no patients in this age group -> skip
                continue

            group_name = f"<{thr}" if label == "lt" else f"≥{thr}"
            shap_group = shap_values_test[mask.values, :]  # (n_group, n_features)

            mean_abs = np.mean(np.abs(shap_group), axis=0)

            for feat, val in zip(feature_list, mean_abs):
                records.append(
                    {
                        "threshold": int(thr),
                        "group": group_name,
                        "n_patients": n_group,
                        "feature": feat,
                        "mean_abs_shap": float(val),
                    }
                )

    df = pd.DataFrame(records)

    if not df.empty:
        df = df.sort_values(
            ["threshold", "group", "mean_abs_shap"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    return df


def compute_mrs90_shap(
    model: LGBMClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_list: List[str],
    nsamples: int = 300,
) -> Dict[str, Any]:
    """
    Compute SHAP values for both TRAIN (optional subsample) and TEST sets.
    TRIPOD-friendly: TEST set SHAP is never mixed with train data.

    Additionally computes an age-stratified SHAP summary on the TEST set
    for age cutpoints: <50 / ≥50, <75 / ≥75, <80 / ≥80, <85 / ≥85.

    Returns dictionaries matching the Kedro pipeline outputs.
    """

    # -----------------------------
    # Subsample train for speed
    # -----------------------------
    if (nsamples is not None) and (len(X_train) > nsamples):
        X_train_eval = X_train.sample(n=nsamples, random_state=42)
    else:
        X_train_eval = X_train

    X_test_eval = X_test  # full test set SHAP

    # -----------------------------
    # Create SHAP explainer
    # -----------------------------
    explainer = shap.TreeExplainer(model)

    # -----------------------------
    # TRAIN SHAP
    # -----------------------------
    shap_raw_tr = explainer.shap_values(X_train_eval, check_additivity=False)
    expected_raw = explainer.expected_value

    if isinstance(shap_raw_tr, list):
        # Binary classifier: take SHAP for class 1 (good outcome)
        shap_train = np.array(shap_raw_tr[1])
        if isinstance(expected_raw, (list, np.ndarray)):
            expected_value = float(expected_raw[1])
        else:
            expected_value = float(expected_raw)
    else:
        shap_train = np.array(shap_raw_tr)
        if isinstance(expected_raw, (list, np.ndarray)):
            expected_value = float(expected_raw[-1])
        else:
            expected_value = float(expected_raw)

    # Summary for TRAIN
    mean_abs_train = np.mean(np.abs(shap_train), axis=0)
    shap_summary_train = (
        pd.DataFrame(
            {"feature": feature_list, "mean_abs_shap": mean_abs_train}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # -----------------------------
    # TEST SHAP
    # -----------------------------
    shap_raw_te = explainer.shap_values(X_test_eval, check_additivity=False)

    if isinstance(shap_raw_te, list):
        shap_test = np.array(shap_raw_te[1])  # class 1
    else:
        shap_test = np.array(shap_raw_te)

    # Summary for TEST (overall)
    mean_abs_test = np.mean(np.abs(shap_test), axis=0)
    shap_summary_test = (
        pd.DataFrame(
            {"feature": feature_list, "mean_abs_shap": mean_abs_test}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # -----------------------------
    # Age-stratified TEST SHAP
    # -----------------------------
    shap_summary_test_by_age = mrs90_shap_importance_by_age_groups(
        shap_values_test=shap_test,
        X_test=X_test_eval,
        feature_list=feature_list,
        thresholds=[50, 75, 80, 85],
        age_col="age",
    )

    # -----------------------------
    # Return format required by Kedro
    # -----------------------------
    return {
        "mrs90_shap_train_values": shap_train,
        "mrs90_shap_test_values": shap_test,
        "mrs90_shap_expected_value": expected_value,
        "mrs90_shap_summary_train": shap_summary_train,
        "mrs90_shap_summary_test": shap_summary_test,
        # NEW: age-stratified test set summary
        "mrs90_shap_summary_test_by_age": shap_summary_test_by_age,
    }

def plot_mrs90_shap_age_groups(
    shap_values_test: np.ndarray,
    X_test: pd.DataFrame,
    feature_list: List[str],
    thresholds: List[int] = [50, 75, 80, 85],
    age_col: str = "age",
    top_n: int = 15,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    """
    Create age-stratified SHAP feature-importance figures on TEST set.

    For each threshold T we plot:
      - left panel: SHAP mean(|value|) for age < T (ranked within this group)
      - right panel: SHAP mean(|value|) for age ≥ T (ranked within this group)

    Each PNG shows the top_n features for each group separately.
    Returns 4 figures (for thresholds 50, 75, 80, 85), to be written by Kedro.
    """

    if age_col not in X_test.columns:
        raise ValueError(f"X_test must contain '{age_col}' for age-stratified plots.")

    age = X_test[age_col]
    n_samples, n_features = shap_values_test.shape

    if n_samples != X_test.shape[0]:
        raise ValueError(
            f"shap_values_test has {n_samples} rows, "
            f"but X_test has {X_test.shape[0]} rows."
        )

    figs: List[plt.Figure] = []
    feature_arr = np.array(feature_list)

    for thr in thresholds:
        mask_lt = (age < thr) & age.notna()
        mask_ge = (age >= thr) & age.notna()

        n_lt = int(mask_lt.sum())
        n_ge = int(mask_ge.sum())

        # handle case with no patients at all (paranoid)
        if n_lt == 0 and n_ge == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(
                0.5,
                0.5,
                f"No patients for threshold {thr}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            figs.append(fig)
            continue

        # SHAP mean |value| in each group
        mean_abs_lt = np.zeros(n_features)
        mean_abs_ge = np.zeros(n_features)

        if n_lt > 0:
            shap_lt = shap_values_test[mask_lt.values, :]
            mean_abs_lt = np.mean(np.abs(shap_lt), axis=0)
        if n_ge > 0:
            shap_ge = shap_values_test[mask_ge.values, :]
            mean_abs_ge = np.mean(np.abs(shap_ge), axis=0)

        # Decide number of rows to help set figure height
        max_len = 0
        if n_lt > 0:
            max_len = max(max_len, min(top_n, np.count_nonzero(mean_abs_lt)))
        if n_ge > 0:
            max_len = max(max_len, min(top_n, np.count_nonzero(mean_abs_ge)))
        max_len = max(max_len, 5)  # avoid tiny figures

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 0.4 * max_len + 2),
            sharey=False,
        )
        ax_left, ax_right = axes

        # ----- left: age < thr, ranked within this group -----
        if n_lt > 0 and mean_abs_lt.sum() > 0:
            order_lt = np.argsort(-mean_abs_lt)[:top_n]
            feat_lt = feature_arr[order_lt]
            vals_lt = mean_abs_lt[order_lt]

            y_pos_lt = np.arange(len(feat_lt))
            ax_left.barh(y_pos_lt, vals_lt)
            ax_left.set_yticks(y_pos_lt)
            ax_left.set_yticklabels(feat_lt)
            ax_left.set_title(f"Age < {thr} (n={n_lt})")
            ax_left.set_xlabel("Mean |SHAP|")
            ax_left.invert_yaxis()
            ax_left.grid(axis="x", alpha=0.3)
        else:
            ax_left.text(
                0.5,
                0.5,
                f"No patients\nAge < {thr}",
                ha="center",
                va="center",
                transform=ax_left.transAxes,
            )
            ax_left.axis("off")

        # ----- right: age ≥ thr, ranked within this group -----
        if n_ge > 0 and mean_abs_ge.sum() > 0:
            order_ge = np.argsort(-mean_abs_ge)[:top_n]
            feat_ge = feature_arr[order_ge]
            vals_ge = mean_abs_ge[order_ge]

            y_pos_ge = np.arange(len(feat_ge))
            ax_right.barh(y_pos_ge, vals_ge)
            ax_right.set_yticks(y_pos_ge)
            ax_right.set_yticklabels(feat_ge)
            ax_right.set_title(f"Age ≥ {thr} (n={n_ge})")
            ax_right.set_xlabel("Mean |SHAP|")
            ax_right.invert_yaxis()
            ax_right.grid(axis="x", alpha=0.3)
        else:
            ax_right.text(
                0.5,
                0.5,
                f"No patients\nAge ≥ {thr}",
                ha="center",
                va="center",
                transform=ax_right.transAxes,
            )
            ax_right.axis("off")

        fig.suptitle(
            f"SHAP feature importance by age groups (threshold {thr} years)",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        figs.append(fig)

    # We expect thresholds = [50, 75, 80, 85]
    return figs[0], figs[1], figs[2], figs[3]


def plot_mrs90_shap(
    shap_values_test: np.ndarray,
    X_test: pd.DataFrame,
    feature_list: List[str],
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:

    # 1) Barplot
    mean_abs = np.mean(np.abs(shap_values_test), axis=0)
    order = np.argsort(-mean_abs)

    fig_bar, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        np.array(feature_list)[order],
        mean_abs[order],
    )
    ax.set_title("SHAP Feature Importance (Test Set)")
    ax.invert_yaxis()

    # 2) Beeswarm
    fig_bee = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_test,
        X_test,
        feature_names=feature_list,
        show=False,
    )

    # 3) Dependence plot (top feature)
    top_feature = feature_list[order[0]]

    fig_dep = plt.figure(figsize=(6, 5))
    shap.dependence_plot(
        top_feature,
        shap_values_test,
        X_test,
        feature_names=feature_list,
        show=False,
        interaction_index=None,
    )

    return fig_bar, fig_bee, fig_dep


def mrs90_feature_importance(
    model: LGBMClassifier,
    feature_list: List[str],
) -> pd.DataFrame:
    """
    Extract global feature importance from the fitted LightGBM model.
    """
    importances = model.feature_importances_
    fi = (
        pd.DataFrame({"feature": feature_list, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return fi
