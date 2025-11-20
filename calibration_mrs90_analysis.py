# calibration_mrs90_analysis.py
import os
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
import matplotlib.pyplot as plt

X_TRAIN_PATH = "data/06_models/modeling_mrs90/mrs90_X_train.parquet"
Y_TRAIN_PATH = "data/06_models/modeling_mrs90/mrs90_y_train.parquet"
X_TEST_PATH = "data/06_models/modeling_mrs90/mrs90_X_test.parquet"
Y_TEST_PATH = "data/06_models/modeling_mrs90/mrs90_y_test.parquet"
OUT_DIR = "data/08_reporting/modeling_mrs90/calibration"

RELAX_GLYCEMIA_CONSTRAINT = True  # match modeling_mrs90


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_monotone_model(feature_names):
    """
    Recreate your final LightGBM model with clinically motivated
    monotone constraints. Effect is on probability of GOOD outcome.
    """
    gly_sign = 0 if RELAX_GLYCEMIA_CONSTRAINT else -1

    sign_map = {
        "age": -1,
        "admission_nihss": -1,
        "mrs_before": 0,
        "onset_to_ivt_min": 0,
        "onset_to_puncture_min": -1,
        "aspects": 0,
        "glycemia": gly_sign,
        "age_x_nihss": 0,
        "onset_to_puncture_min_x_nihss": 0,
        "age_x_onset_to_puncture_min": 0,
    }
    monotone_constraints = [sign_map.get(f, 0) for f in feature_names]
    if len(monotone_constraints) != len(feature_names):
        raise ValueError("Monotone constraints length mismatch.")

    return LGBMClassifier(
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
    )


def compute_calibration_slope_intercept(y_true, proba):
    """
    Logistic regression of y on logit(p):
      logit(y) ~ alpha + beta * logit(p)
    slope = beta, intercept = alpha

    Uses sklearn LogisticRegression with L2 penalty to avoid
    separation issues (TRIPOD-friendly).
    """
    y_true = np.asarray(y_true).astype(int)

    eps = 1e-15
    p = np.clip(proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)

    lr = LogisticRegression(
        solver="lbfgs",
        fit_intercept=True,
    )
    lr.fit(logit_p, y_true)

    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    return slope, intercept


def main():
    ensure_out_dir(OUT_DIR)

    print(f"Loading X_train: {X_TRAIN_PATH}")
    X_train = pd.read_parquet(X_TRAIN_PATH)
    print(f"Loading y_train: {Y_TRAIN_PATH}")
    y_train = pd.read_parquet(Y_TRAIN_PATH).iloc[:, 0].values

    print(f"Loading X_test: {X_TEST_PATH}")
    X_test = pd.read_parquet(X_TEST_PATH)
    print(f"Loading y_test: {Y_TEST_PATH}")
    y_test = pd.read_parquet(Y_TEST_PATH).iloc[:, 0].values

    feature_names = list(X_train.columns)
    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)

    # ---- 1) Fit raw LightGBM model on TRAIN (no recalibration) ----
    model = build_monotone_model(feature_names)
    print("Fitting LightGBM model on TRAIN (raw probabilities)...")
    model.fit(X_train, y_train)

    # ---- 2) Predict on TRAIN and TEST (raw probabilities) ----
    print("Predicting on TRAIN and TEST sets with raw model...")
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    # ---- 3) Calibration curve (TEST set, raw probs) ----
    frac_pos, mean_pred = calibration_curve(
        y_test, proba_test, n_bins=10, strategy="quantile"
    )

    plt.figure()
    plt.plot(mean_pred, frac_pos, "s-", label="LGBM (raw)")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed proportion (mRS 0–2)")
    plt.title("Calibration curve – mRS90 LightGBM model (TEST, raw)")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)

    calib_plot_path = os.path.join(OUT_DIR, "mrs90_calibration_curve_raw.png")
    plt.tight_layout()
    plt.savefig(calib_plot_path, dpi=200)
    plt.close()
    print("Saved calibration plot:", calib_plot_path)

    # ---- 4) Calibration + performance metrics (TRAIN & TEST) ----
    metrics = {}

    for split_name, y, proba in [
        ("train", y_train, proba_train),
        ("test", y_test, proba_test),
    ]:
        p_clip = np.clip(proba, 1e-15, 1 - 1e-15)

        auc = float(roc_auc_score(y, p_clip))
        brier = float(brier_score_loss(y, p_clip))
        ll = float(log_loss(y, p_clip))
        slope, intercept = compute_calibration_slope_intercept(y, p_clip)

        metrics[split_name] = {
            "dataset": split_name,
            "n": int(len(y)),
            "auc": auc,
            "brier": brier,
            "log_loss": ll,
            "cal_intercept": intercept,
            "cal_slope": slope,
        }

    # Convenience top-level aliases for TEST (like before)
    metrics["roc_auc_test"] = metrics["test"]["auc"]
    metrics["brier_test"] = metrics["test"]["brier"]
    metrics["log_loss_test"] = metrics["test"]["log_loss"]
    metrics["calibration_intercept_test"] = metrics["test"]["cal_intercept"]
    metrics["calibration_slope_test"] = metrics["test"]["cal_slope"]

    out_json = os.path.join(OUT_DIR, "mrs90_calibration_metrics_raw.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nCalibration / performance metrics (RAW LightGBM):")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {out_json}")


if __name__ == "__main__":
    main()
