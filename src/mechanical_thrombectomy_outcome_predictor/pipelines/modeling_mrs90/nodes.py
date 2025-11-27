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
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import optuna
from optuna.samplers import TPESampler

# Central config: monotone constraints
from .config import build_monotone_constraints


# ============================================================
# DATASET BUILDING
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

    NOTE:
      - Interaction columns (age_x_nihss, onset_to_puncture_min_x_nihss,
        age_x_onset_to_puncture_min) are ALWAYS created if their parents exist.
      - Optuna later decides whether to USE or DROP each interaction per trial.
        That keeps this node deterministic and TRIPOD-clean.
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

    df = df_ready.merge(
        df_split[["patient_id", "split"]],
        on="patient_id",
        how="inner",
    )

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
    # 3b) Create interaction columns (always)
    #      Optuna will later decide whether to use them.
    # -------------------------
    if {"age", "admission_nihss"}.issubset(df.columns):
        df["age_x_nihss"] = df["age"] * df["admission_nihss"]

    if {"onset_to_puncture_min", "admission_nihss"}.issubset(df.columns):
        df["onset_to_puncture_min_x_nihss"] = (
            df["onset_to_puncture_min"] * df["admission_nihss"]
        )

    if {"age", "onset_to_puncture_min"}.issubset(df.columns):
        df["age_x_onset_to_puncture_min"] = (
            df["age"] * df["onset_to_puncture_min"]
        )

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
        "sich_missing",
        "non_sich_missing",
        "procedure_missing",
        "tici_missing",
        "stent_combo_missing",
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

    y_train_df = (
        y_bin.loc[train_mask].reset_index(drop=True).to_frame("mrs90_good")
    )
    y_test_df = (
        y_bin.loc[test_mask].reset_index(drop=True).to_frame("mrs90_good")
    )

    train_ids_df = (
        df.loc[train_mask, "patient_id"]
        .reset_index(drop=True)
        .to_frame("patient_id")
    )
    test_ids_df = (
        df.loc[test_mask, "patient_id"]
        .reset_index(drop=True)
        .to_frame("patient_id")
    )

    return (
        X_train,
        y_train_df,
        X_test,
        y_test_df,
        train_ids_df,
        test_ids_df,
        feature_cols,
    )


# ============================================================
# MONOTONE CONSTRAINTS (thin wrapper)
# ============================================================

def _get_monotone_constraints(feature_list: List[str]) -> List[int]:
    """
    Thin wrapper around the shared config function.
    All monotone logic (including glycemia) stays centralized in config.py.
    """
    return build_monotone_constraints(feature_list)


# ============================================================
# FINAL MODEL TRAINING WITH OPTUNA HYPERPARAMETERS
# ============================================================

def train_mrs90_lgbm_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,           # 1-col DF
    best_params: Dict[str, Any],     # from Optuna (includes interaction flags)
) -> Tuple[LGBMClassifier, Dict[str, float]]:
    """
    Train the final LightGBM model using Optuna-tuned hyperparameters.

    best_params is expected to contain:
      - LightGBM params (n_estimators, learning_rate, ...)
      - interaction flags:
            use_age_x_nihss: bool
            use_onset_to_puncture_min_x_nihss: bool
            use_age_x_onset_to_puncture_min: bool
    """
    y = y_train.iloc[:, 0].values

    # --------------------------------------------------
    # 1) Extract interaction flags from best_params
    # --------------------------------------------------
    use_age_x_nihss = best_params.pop("use_age_x_nihss", True)
    use_onset_x_nihss = best_params.pop("use_onset_to_puncture_min_x_nihss", True)
    use_age_x_onset = best_params.pop("use_age_x_onset_to_puncture_min", True)

    # Start from all columns, then drop interactions if flag=False
    active_features: List[str] = list(X_train.columns)

    if not use_age_x_nihss and "age_x_nihss" in active_features:
        active_features.remove("age_x_nihss")
    if not use_onset_x_nihss and "onset_to_puncture_min_x_nihss" in active_features:
        active_features.remove("onset_to_puncture_min_x_nihss")
    if not use_age_x_onset and "age_x_onset_to_puncture_min" in active_features:
        active_features.remove("age_x_onset_to_puncture_min")

    X_train_active = X_train[active_features]
    monotone_constraints = _get_monotone_constraints(active_features)

    # --------------------------------------------------
    # 2) Base params NOT tuned by Optuna
    # --------------------------------------------------
    base_params: Dict[str, Any] = {
        "objective": "binary",
        "random_state": 42,
        "n_jobs": -1,
        "monotone_constraints": monotone_constraints,
        "enable_categorical": True,
    }

    # Merge Optuna's best hyperparameters (now without flags)
    base_params.update(best_params)

    model = LGBMClassifier(**base_params)
    model.fit(X_train_active, y)

    proba = model.predict_proba(X_train_active)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics: Dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "log_loss": float(log_loss(y, proba)),
    }

    return model, metrics


# ============================================================
# OPTUNA TUNING (AUC + BRIER, with interaction auto-selection)
# ============================================================

def optuna_tune_mrs90_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,  # 1-col DF
    n_trials: int = 50,
    n_splits: int = 5,      # used for tuning only
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, plt.Figure]:
    """
    Run Optuna hyperparameter tuning for the mRS90 LightGBM model.

    - Uses TRAIN only, with StratifiedKFold CV for *tuning*.
    - Objective combines:
        * mean AUC (to maximize)
        * mean Brier score (to minimize)
      via: objective = mean_auc - LAMBDA_BRIER * mean_brier

    - Also tunes:
        use_age_x_nihss
        use_onset_to_puncture_min_x_nihss
        use_age_x_onset_to_puncture_min

    - After tuning, performs a *separate* 10×20 repeated stratified CV with the
      best hyperparameters to get a robust distribution of AUC, Brier and log-loss.

    Returns:
        best_params   : dict (Optuna best params incl. interaction flags)
        cv_metrics    : dict (JSON-able, includes per-fold metrics)
        cv_summary_df : 1-row DataFrame for CSV summary (mrs90_vs_summary_table)
        fig_auc_dist  : Matplotlib Figure with AUC distribution histogram
                        (mrs_lgbm_cv_auc_distribution_raw)
    """
    y = y_train.iloc[:, 0].values
    all_features: List[str] = list(X_train.columns)

    # ---------- 1) CV for OPTUNA TUNING ----------
    skf_tune = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Weight for Brier penalty in the Optuna objective
    LAMBDA_BRIER = 0.75

    def objective(trial: optuna.Trial) -> float:
        # --------------------------------------------------
        # 1) Interaction flags (trial-level feature selection)
        # --------------------------------------------------
        use_age_x_nihss = trial.suggest_categorical(
            "use_age_x_nihss", [True, False]
        )
        use_onset_x_nihss = trial.suggest_categorical(
            "use_onset_to_puncture_min_x_nihss", [True, False]
        )
        use_age_x_onset = trial.suggest_categorical(
            "use_age_x_onset_to_puncture_min", [True, False]
        )

        active_features = list(all_features)
        if not use_age_x_nihss and "age_x_nihss" in active_features:
            active_features.remove("age_x_nihss")
        if not use_onset_x_nihss and "onset_to_puncture_min_x_nihss" in active_features:
            active_features.remove("onset_to_puncture_min_x_nihss")
        if not use_age_x_onset and "age_x_onset_to_puncture_min" in active_features:
            active_features.remove("age_x_onset_to_puncture_min")

        # Monotone constraints for the ACTIVE feature set
        monotone_constraints = _get_monotone_constraints(active_features)

        # --------------------------------------------------
        # 2) LightGBM hyperparameter search space
        # --------------------------------------------------
        lgbm_params: Dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 45),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }

        aucs: List[float] = []
        briers: List[float] = []

        for train_idx, val_idx in skf_tune.split(X_train, y):
            X_tr = X_train.iloc[train_idx][active_features]
            X_val = X_train.iloc[val_idx][active_features]
            y_tr, y_val = y[train_idx], y[val_idx]

            base_params: Dict[str, Any] = {
                "objective": "binary",
                "random_state": random_state,
                "n_jobs": -1,
                "enable_categorical": True,
                "monotone_constraints": monotone_constraints,
            }
            base_params.update(lgbm_params)

            model = LGBMClassifier(**base_params)
            model.fit(X_tr, y_tr)

            proba_val = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, proba_val))
            briers.append(brier_score_loss(y_val, proba_val))

        mean_auc = float(np.mean(aucs))
        mean_brier = float(np.mean(briers))

        # Multi-objective scalar
        objective_value = mean_auc - LAMBDA_BRIER * mean_brier
        return objective_value

    # ---------- 2) Run Optuna TUNING ----------
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params  # includes interaction flags + LGBM params

    # ============================================================
    # 3) 10×20 REPEATED CV WITH BEST HYPERPARAMETERS (REPORTING)
    # ============================================================

    # Extract interaction flags from best_params (without mutating original)
    bp = dict(best_params)
    use_age_x_nihss = bp.pop("use_age_x_nihss", True)
    use_onset_x_nihss = bp.pop("use_onset_to_puncture_min_x_nihss", True)
    use_age_x_onset = bp.pop("use_age_x_onset_to_puncture_min", True)

    active_features = list(all_features)
    if not use_age_x_nihss and "age_x_nihss" in active_features:
        active_features.remove("age_x_nihss")
    if not use_onset_x_nihss and "onset_to_puncture_min_x_nihss" in active_features:
        active_features.remove("onset_to_puncture_min_x_nihss")
    if not use_age_x_onset and "age_x_onset_to_puncture_min" in active_features:
        active_features.remove("age_x_onset_to_puncture_min")

    monotone_constraints = _get_monotone_constraints(active_features)

    # Repeated 10×20 CV
    N_SPLITS_EVAL = 10
    N_REPEATS_EVAL = 20

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS_EVAL,
        n_repeats=N_REPEATS_EVAL,
        random_state=random_state,
    )

    aucs: List[float] = []
    briers: List[float] = []
    loglosses: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train, y), start=1):
        X_tr = X_train.iloc[train_idx][active_features]
        X_val = X_train.iloc[val_idx][active_features]
        y_tr, y_val = y[train_idx], y[val_idx]

        base_params: Dict[str, Any] = {
            "objective": "binary",
            "random_state": random_state,
            "n_jobs": -1,
            "enable_categorical": True,
            "monotone_constraints": monotone_constraints,
        }
        base_params.update(bp)

        model = LGBMClassifier(**base_params)
        model.fit(X_tr, y_tr)

        proba_val = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, proba_val))
        briers.append(brier_score_loss(y_val, proba_val))
        loglosses.append(log_loss(y_val, proba_val))

    def _summ(v: List[float]) -> Dict[str, float]:
        return {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        }

    # Top trials info (unchanged)
    top_trials_info = []
    sorted_trials = sorted(
        study.trials,
        key=lambda tr: tr.value if tr.value is not None else -np.inf,
        reverse=True,
    )
    for t in sorted_trials[:5]:
        top_trials_info.append(
            {
                "number": int(t.number),
                "objective_value": float(t.value) if t.value is not None else None,
                "params": t.params,
            }
        )

    cv_metrics: Dict[str, Any] = {
        "n_splits": N_SPLITS_EVAL,
        "n_repeats": N_REPEATS_EVAL,
        "summary": {
            "auc": _summ(aucs),
            "brier": _summ(briers),
            "log_loss": _summ(loglosses),
        },
        "per_fold": {
            "auc": aucs,
            "brier": briers,
            "log_loss": loglosses,
        },
        "optuna": {
            "n_trials": n_trials,
            "best_value": float(best_trial.value),
            "best_trial_number": int(best_trial.number),
            "best_params": best_params,  # includes interaction flags
            "top_trials": top_trials_info,
        },
    }

    # ============================================================
    # 4) Build 1-row summary table (for CSV)
    # ============================================================
    cv_summary_df = pd.DataFrame(
        [
            {
                "model": "LGBM_mrs90",
                "strategy": "10x20_repeated_stratified_CV",
                "n_splits": N_SPLITS_EVAL,
                "n_repeats": N_REPEATS_EVAL,
                "n_fits": N_SPLITS_EVAL * N_REPEATS_EVAL,
                "auc_mean": cv_metrics["summary"]["auc"]["mean"],
                "auc_sd": cv_metrics["summary"]["auc"]["std"],
                "brier_mean": cv_metrics["summary"]["brier"]["mean"],
                "brier_sd": cv_metrics["summary"]["brier"]["std"],
                "log_loss_mean": cv_metrics["summary"]["log_loss"]["mean"],
                "log_loss_sd": cv_metrics["summary"]["log_loss"]["std"],
            }
        ]
    )

    # ============================================================
    # 5) AUC distribution figure (for PNG)
    # ============================================================
    fig_auc_dist, ax = plt.subplots(figsize=(6, 4))
    ax.hist(aucs, bins=20, alpha=0.75)
    ax.axvline(np.mean(aucs), linestyle="--")
    ax.set_xlabel("ROC AUC (per CV fold)")
    ax.set_ylabel("Count")
    ax.set_title("10×20 CV ROC AUC distribution (training cohort)")
    ax.grid(alpha=0.3)

    return best_params, cv_metrics, cv_summary_df, fig_auc_dist



# ============================================================
# TEST EVALUATION
# ============================================================
def evaluate_mrs90_lgbm(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,    # 1-col DF
    test_ids: pd.DataFrame,  # 1-col DF
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate the trained LightGBM on the held-out temporal test set.
    Returns metrics + a prediction table.

    IMPORTANT:
      We align X_test columns to the *actual* feature set the model was
      trained on (model.feature_name_). This is necessary because Optuna
      may have dropped some interaction features, so the training feature
      set can be a strict subset of the full X_test columns.
    """
    y = y_test.iloc[:, 0].values

    # Align test data to the feature set used in training
    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
        feature_names_model = list(model.feature_name_)
        # This will fail loudly if some expected features are missing in X_test,
        # which is safer than silently misaligning.
        X_for_pred = X_test[feature_names_model]
    else:
        # Fallback: no feature_name_ available, use X_test as-is
        X_for_pred = X_test

    proba = model.predict_proba(X_for_pred)[:, 1]
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

    NOTE:
      - If feature_list is longer than shap_values_test.shape[1], we truncate
        to the first n_features, so lengths always match.
    """
    if age_col not in X_test.columns:
        raise ValueError(
            f"X_test must contain '{age_col}' column for age-stratified SHAP."
        )

    n_samples, n_features = shap_values_test.shape

    if n_samples != X_test.shape[0]:
        raise ValueError(
            f"shap_values_test has {n_samples} rows, "
            f"but X_test has {X_test.shape[0]} rows."
        )

    # Align feature names to SHAP dimensionality
    if len(feature_list) < n_features:
        raise ValueError(
            f"feature_list has length {len(feature_list)} "
            f"but SHAP has {n_features} features."
        )
    feature_names = feature_list[:n_features]

    age = X_test[age_col]
    records: List[Dict[str, Any]] = []

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

            for feat, val in zip(feature_names, mean_abs):
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

    NOTE:
      - We align X_train / X_test and feature names to the actual features
        used by the LightGBM model (model.feature_name_), if available.
      - If model.feature_name_ is not available, we fall back to feature_list.
    """

    # ------------------------------------------------------------------
    # 0) Determine canonical feature names actually used by the model
    # ------------------------------------------------------------------
    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
        model_feature_names = list(model.feature_name_)
    else:
        model_feature_names = list(feature_list)

    # Align X_train and X_test to the model's feature set
    X_train_aligned = X_train[model_feature_names]
    X_test_aligned = X_test[model_feature_names]

    # -----------------------------
    # Subsample train for speed
    # -----------------------------
    if (nsamples is not None) and (len(X_train_aligned) > nsamples):
        X_train_eval = X_train_aligned.sample(n=nsamples, random_state=42)
    else:
        X_train_eval = X_train_aligned

    X_test_eval = X_test_aligned  # full test set SHAP

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

    # Ensure feature name length matches SHAP dimension
    n_features_train = shap_train.shape[1]
    if len(model_feature_names) != n_features_train:
        model_feature_names = model_feature_names[:n_features_train]

    # Summary for TRAIN
    mean_abs_train = np.mean(np.abs(shap_train), axis=0)
    shap_summary_train = (
        pd.DataFrame(
            {"feature": model_feature_names, "mean_abs_shap": mean_abs_train}
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

    n_features_test = shap_test.shape[1]
    if len(model_feature_names) != n_features_test:
        model_feature_names = model_feature_names[:n_features_test]

    # Summary for TEST (overall)
    mean_abs_test = np.mean(np.abs(shap_test), axis=0)
    shap_summary_test = (
        pd.DataFrame(
            {"feature": model_feature_names, "mean_abs_shap": mean_abs_test}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # -----------------------------
    # Age-stratified TEST SHAP
    # -----------------------------
    shap_summary_test_by_age = mrs90_shap_importance_by_age_groups(
        shap_values_test=shap_test,
        X_test=X_test_aligned,
        feature_list=model_feature_names,
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
        # Age-stratified test set summary
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

    if age_col not in X_test.columns:
        raise ValueError(
            f"X_test must contain '{age_col}' for age-stratified plots."
        )

    n_samples, n_features = shap_values_test.shape

    if n_samples != X_test.shape[0]:
        raise ValueError(
            f"shap_values_test has {n_samples} rows, "
            f"but X_test has {X_test.shape[0]} rows."
        )

    # Align feature names to SHAP dimensionality
    if len(feature_list) < n_features:
        raise ValueError(
            f"feature_list has length {len(feature_list)} "
            f"but SHAP has {n_features} features."
        )
    feature_names = feature_list[:n_features]
    feature_arr = np.array(feature_names)

    age = X_test[age_col]
    figs: List[plt.Figure] = []

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
  
    n_samples, n_features = shap_values_test.shape

    if n_samples != X_test.shape[0]:
        raise ValueError(
            f"shap_values_test has {n_samples} rows, "
            f"but X_test has {X_test.shape[0]} rows."
        )

    if len(feature_list) < n_features:
        raise ValueError(
            f"feature_list has length {len(feature_list)} "
            f"but SHAP has {n_features} features."
        )
    feature_names = feature_list[:n_features]

    # Align X_test to the feature names used in SHAP
    X_for_shap = X_test[feature_names]

    # ------------------------------------------------
    # 1) BARPLOT — LIMITED TO TOP-30 FEATURES
    # ------------------------------------------------
    mean_abs = np.mean(np.abs(shap_values_test), axis=0)
    order = np.argsort(-mean_abs)

    TOP_N = 30
    top_order = order[:TOP_N]
    top_features = np.array(feature_names)[top_order]
    top_values = mean_abs[top_order]

    fig_bar, ax = plt.subplots(figsize=(8, 0.35 * TOP_N + 2))
    ax.barh(top_features, top_values)
    ax.set_title(f"Top {TOP_N} SHAP Feature Importances (Test Set)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # ------------------------------------------------
    # 2) Beeswarm (unchanged)
    # ------------------------------------------------
    fig_bee = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_test,
        X_for_shap,
        feature_names=feature_names,
        show=False,
    )

    # ------------------------------------------------
    # 3) Dependence plot (top feature)
    # ------------------------------------------------
    top_feature = feature_names[order[0]]

    fig_dep = plt.figure(figsize=(6, 5))
    shap.dependence_plot(
        top_feature,
        shap_values_test,
        X_for_shap,
        feature_names=feature_names,
        show=False,
        interaction_index=None,
    )

    return fig_bar, fig_bee, fig_dep



def mrs90_feature_importance(
    model: LGBMClassifier,
    feature_list: List[str],
) -> pd.DataFrame:
    
    importances = model.feature_importances_

    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
        model_feature_names = list(model.feature_name_)
    else:
        # Fallback: use provided feature_list, truncated if necessary
        model_feature_names = list(feature_list)

    n_importances = len(importances)
    if len(model_feature_names) != n_importances:
        model_feature_names = model_feature_names[:n_importances]

    fi = (
        pd.DataFrame(
            {
                "feature": model_feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return fi
