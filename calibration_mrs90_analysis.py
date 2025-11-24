# calibration_mrs90_analysis.py

import os
import json
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from src.mechanical_thrombectomy_outcome_predictor.pipelines.modeling_mrs90.calibration_utils import (
    compute_calibration_slope_intercept,
    apply_logistic_calibration,
)
from src.mechanical_thrombectomy_outcome_predictor.pipelines.modeling_mrs90.config import (
    build_monotone_constraints,
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
MODEL_PATH = "data/06_models/modeling_mrs90/mrs90_lgbm_model.pkl"

X_TRAIN_PATH = "data/06_models/modeling_mrs90/mrs90_X_train.parquet"
Y_TRAIN_PATH = "data/06_models/modeling_mrs90/mrs90_y_train.parquet"
X_TEST_PATH = "data/06_models/modeling_mrs90/mrs90_X_test.parquet"
Y_TEST_PATH = "data/06_models/modeling_mrs90/mrs90_y_test.parquet"

OUT_DIR = "data/08_reporting/modeling_mrs90/calibration"

# K-folds for cross-validated Platt scaling
N_SPLITS_CAL = 5


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    ensure_out_dir(OUT_DIR)

    # -----------------------------
    # Load data
    # -----------------------------
    print(f"Loading X_train: {X_TRAIN_PATH}")
    X_train = pd.read_parquet(X_TRAIN_PATH)
    print(f"Loading y_train: {Y_TRAIN_PATH}")
    y_train = pd.read_parquet(Y_TRAIN_PATH).iloc[:, 0].values

    print(f"Loading X_test: {X_TEST_PATH}")
    X_test = pd.read_parquet(X_TEST_PATH)
    print(f"Loading y_test: {Y_TEST_PATH}")
    y_test = pd.read_parquet(Y_TEST_PATH).iloc[:, 0].values

    feature_names = list(X_train.columns)
    print("X_train shape (before alignment):", X_train.shape)
    print("X_test  shape (before alignment):", X_test.shape)

    # -----------------------------
    # Load tuned LightGBM model
    # -----------------------------
    print(f"\nLoading tuned LightGBM model from: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        final_model: LGBMClassifier = pickle.load(f)

    # ------------------------------------------------------------
    # Align X_train / X_test to the model's feature order
    # ------------------------------------------------------------
    if hasattr(final_model, "feature_name_") and final_model.feature_name_ is not None:
        model_feats = list(final_model.feature_name_)
        if list(feature_names) != model_feats:
            print(
                "WARNING: Feature order/name mismatch between X_train and "
                "loaded model. Aligning to model.feature_name_."
            )

        # Make sure all model features exist in X_train / X_test
        missing_in_train = set(model_feats) - set(X_train.columns)
        missing_in_test = set(model_feats) - set(X_test.columns)
        if missing_in_train or missing_in_test:
            raise ValueError(
                "X_train/X_test are missing features used by the model:\n"
                f"  missing_in_train: {sorted(missing_in_train)}\n"
                f"  missing_in_test:  {sorted(missing_in_test)}"
            )

        # Align to the exact order used during training
        X_train = X_train[model_feats]
        X_test = X_test[model_feats]
        feature_names = model_feats
    else:
        print(
            "WARNING: final_model has no feature_name_. "
            "Using X_train columns as feature_names."
        )

    print("X_train shape (after alignment):", X_train.shape)
    print("X_test  shape (after alignment):", X_test.shape)

    # ------------------------------------------------------------
    # Build monotone_constraints for the CURRENT feature set
    # ------------------------------------------------------------
    monotone_constraints = build_monotone_constraints(feature_names)

    # Base params from the trained model, but with fresh monotone_constraints
    base_params = final_model.get_params()
    base_params["monotone_constraints"] = monotone_constraints

    # Older LightGBM versions don't know enable_categorical -> drop if present
    base_params.pop("enable_categorical", None)

    def make_cv_model() -> LGBMClassifier:
        """Create a new LightGBM model with the same params as the final model,
        but with monotone_constraints matching the current feature set.
        """
        return LGBMClassifier(**base_params)

    # ============================================================
    # 1) K-fold CV on TRAIN to get out-of-fold raw probabilities
    #    and fit Platt (a,b) WITHOUT using TEST at all.
    # ============================================================
    print(f"\nRunning {N_SPLITS_CAL}-fold CV on TRAIN for Platt scaling...")
    skf = StratifiedKFold(
        n_splits=N_SPLITS_CAL, shuffle=True, random_state=42
    )
    oof_proba_train = np.zeros(len(y_train), dtype=float)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        print(f"  Fold {fold_idx}/{N_SPLITS_CAL} ...")
        model_cv = make_cv_model()
        model_cv.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_proba_train[val_idx] = model_cv.predict_proba(
            X_train.iloc[val_idx]
        )[:, 1]

    slope, intercept = compute_calibration_slope_intercept(y_train, oof_proba_train)
    cal_a = intercept
    cal_b = slope

    print("\nCross-validated Platt parameters (TRAIN, OOF preds):")
    print(f"  intercept (a): {cal_a:.6f}")
    print(f"  slope     (b): {cal_b:.6f}")

    # ============================================================
    # 2) RAW probabilities from the *final tuned model*
    #    (using aligned X_train / X_test)
    # ============================================================
    print("\nPredicting RAW probabilities on TRAIN and TEST using tuned model...")
    proba_train_raw = final_model.predict_proba(X_train)[:, 1]
    proba_test_raw = final_model.predict_proba(X_test)[:, 1]

    # ============================================================
    # 3) Compute metrics RAW and CALIBRATED (TRAIN + TEST)
    # ============================================================
    metrics = {}

    # --- RAW metrics ---
    for split_name, y, proba in [
        ("train", y_train, proba_train_raw),
        ("test", y_test, proba_test_raw),
    ]:
        p_clip = np.clip(proba, 1e-15, 1 - 1e-15)

        auc = float(roc_auc_score(y, p_clip))
        brier = float(brier_score_loss(y, p_clip))
        ll = float(log_loss(y, p_clip))
        slope_raw, intercept_raw = compute_calibration_slope_intercept(y, p_clip)

        metrics[split_name] = {
            "dataset": split_name,
            "n": int(len(y)),
            "auc": auc,
            "brier": brier,
            "log_loss": ll,
            "cal_intercept": intercept_raw,
            "cal_slope": slope_raw,
        }

    metrics["roc_auc_test"] = metrics["test"]["auc"]
    metrics["brier_test"] = metrics["test"]["brier"]
    metrics["log_loss_test"] = metrics["test"]["log_loss"]
    metrics["calibration_intercept_test"] = metrics["test"]["cal_intercept"]
    metrics["calibration_slope_test"] = metrics["test"]["cal_slope"]

    # --- CALIBRATED metrics (using Platt a,b from TRAIN OOF CV) ---
    print("\nApplying Platt calibration (from TRAIN-CV) to TRAIN and TEST probs...")
    proba_train_cal = apply_logistic_calibration(proba_train_raw, cal_a, cal_b)
    proba_test_cal = apply_logistic_calibration(proba_test_raw, cal_a, cal_b)

    for split_name, y, proba_cal in [
        ("train_calibrated", y_train, proba_train_cal),
        ("test_calibrated", y_test, proba_test_cal),
    ]:
        p_cal_clip = np.clip(proba_cal, 1e-15, 1 - 1e-15)

        auc_cal = float(roc_auc_score(y, p_cal_clip))
        brier_cal = float(brier_score_loss(y, p_cal_clip))
        ll_cal = float(log_loss(y, p_cal_clip))
        slope_cal, intercept_cal = compute_calibration_slope_intercept(
            y, p_cal_clip
        )

        metrics[split_name] = {
            "dataset": split_name,
            "n": int(len(y)),
            "auc": auc_cal,
            "brier": brier_cal,
            "log_loss": ll_cal,
            "cal_intercept": intercept_cal,
            "cal_slope": slope_cal,
        }

    metrics["roc_auc_test_calibrated"] = metrics["test_calibrated"]["auc"]
    metrics["brier_test_calibrated"] = metrics["test_calibrated"]["brier"]
    metrics["log_loss_test_calibrated"] = metrics["test_calibrated"]["log_loss"]

    # ============================================================
    # 4) Calibration curve plot: RAW vs CALIBRATED (TEST)
    # ============================================================
    frac_pos_raw, mean_pred_raw = calibration_curve(
        y_test, proba_test_raw, n_bins=10, strategy="quantile"
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_test, proba_test_cal, n_bins=10, strategy="quantile"
    )

    plt.figure()
    plt.plot(mean_pred_raw, frac_pos_raw, "s-", label="LGBM (raw)")
    plt.plot(mean_pred_cal, frac_pos_cal, "o-", label="LGBM + Platt (TRAIN-CV)")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed proportion (mRS 0–2)")
    plt.title("Calibration curve – mRS90 LightGBM (TEST)")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)

    calib_plot_path = os.path.join(
        OUT_DIR, "mrs90_calibration_curve_raw_vs_calibrated.png"
    )
    plt.tight_layout()
    plt.savefig(calib_plot_path, dpi=200)
    plt.close()
    print("Saved calibration plot (raw vs calibrated):", calib_plot_path)

    # ============================================================
    # 5) Save metrics JSON
    # ============================================================
    out_json = os.path.join(OUT_DIR, "mrs90_calibration_metrics_raw.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nCalibration / performance metrics (RAW + calibrated):")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {out_json}")

    # ============================================================
    # 6) Save logistic calibration parameters for the app
    # ============================================================
    cal_params = {
        "a": float(cal_a),
        "b": float(cal_b),
        "n_train": int(len(y_train)),
        "n_splits_cv": int(N_SPLITS_CAL),
        "note": (
            "Platt / logistic calibration fitted on TRAIN using "
            f"{N_SPLITS_CAL}-fold cross-validation (out-of-fold predictions). "
            "Use: p_cal = logistic(a + b * logit(p_raw)), "
            "where p_raw is the LightGBM predicted probability for mRS 0 to 2."
        ),
    }
    cal_params_path = os.path.join(
        OUT_DIR, "mrs90_logistic_calibration_params.json"
    )
    with open(cal_params_path, "w") as f:
        json.dump(cal_params, f, indent=2)

    print(f"Saved logistic calibration parameters to: {cal_params_path}")

    # ============================================================
    # 7) (Optional) Trigger DCA rebuild automatically
    # ============================================================
    try:
        from decision_curve_mrs90 import main as dca_main

        print("\nRunning decision_curve_mrs90.main() to rebuild DCA...")
        dca_main()
    except Exception as e:
        print(
            "\nWARNING: Could not automatically run decision_curve_mrs90.\n"
            "Run decision_curve_mrs90.py separately if needed.\n"
            f"Error: {e}"
        )


if __name__ == "__main__":
    main()
