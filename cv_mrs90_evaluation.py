# cv_mrs90_evaluation.py
import os
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

X_TRAIN_PATH = "data/06_models/modeling_mrs90/mrs90_X_train.parquet"
Y_TRAIN_PATH = "data/06_models/modeling_mrs90/mrs90_y_train.parquet"
OUT_DIR = "data/08_reporting/modeling_mrs90/cv"

# use the same switches as in modeling_mrs90
RELAX_GLYCEMIA_CONSTRAINT = True  # toggle to match modeling_mrs90

CV_FOLDS = 10
CV_REPEATS = 20
CV_RANDOM_SEED = 123


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_monotone_model(feature_names):
    """
    Recreate your final LightGBM model with clinically motivated
    monotone constraints. Effect is on probability of GOOD outcome.
    """
    gly_sign = 0 if RELAX_GLYCEMIA_CONSTRAINT else -1

    sign_map = {
        "age": -1,                    # older -> worse
        "admission_nihss": -1,        # more severe stroke -> worse
        "mrs_before": 0,             # worse baseline disability -> worse
        "onset_to_ivt_min": 0,       # delay to IVT -> worse
        "onset_to_puncture_min": -1,  # delay to puncture -> worse
        "aspects": 0,                # higher ASPECTS -> better
        "glycemia": gly_sign,
        # interactions, if present in X, will default to 0 (no constraint)
        "age_x_nihss": 0,
        "onset_to_puncture_min_x_nihss": 0,
        "age_x_onset_to_puncture_min": 0,
    }

    monotone_constraints = [sign_map.get(f, 0) for f in feature_names]

    if len(monotone_constraints) != len(feature_names):
        raise ValueError("Monotone constraints length mismatch.")

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
    )
    return model


def compute_calibration_slope_intercept(y_true, proba):
    """
    Logistic regression of y on logit(p):
      logit(y) ~ alpha + beta * logit(p)
    slope = beta, intercept = alpha
    """
    y_true = np.asarray(y_true).astype(int)
    p_clip = np.clip(proba, 1e-15, 1 - 1e-15)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)

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

    print(f"Loading X_train from {X_TRAIN_PATH}")
    X = pd.read_parquet(X_TRAIN_PATH)

    print(f"Loading y_train from {Y_TRAIN_PATH}")
    y = pd.read_parquet(Y_TRAIN_PATH).iloc[:, 0].values  # 1D array

    feature_names = list(X.columns)
    print("X_train shape:", X.shape)

    rows = []

    for r in range(CV_REPEATS):
        print(f"\n=== Repeat {r + 1}/{CV_REPEATS} ===")
        skf = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=CV_RANDOM_SEED + r,
        )
        fold = 0
        for train_idx, val_idx in skf.split(X, y):
            fold += 1
            print(f"  Fold {fold}/{CV_FOLDS}")

            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = build_monotone_model(feature_names)
            model.fit(X_tr, y_tr)

            proba = model.predict_proba(X_val)[:, 1]
            p_clip = np.clip(proba, 1e-15, 1 - 1e-15)

            auc = float(roc_auc_score(y_val, p_clip))
            brier = float(brier_score_loss(y_val, p_clip))
            ll = float(log_loss(y_val, p_clip))
            cal_slope, cal_intercept = compute_calibration_slope_intercept(
                y_val, p_clip
            )

            rows.append(
                {
                    "repeat": r + 1,
                    "fold": fold,
                    "n_val": int(len(y_val)),
                    "auc": auc,
                    "brier": brier,
                    "log_loss": ll,
                    "cal_intercept": cal_intercept,
                    "cal_slope": cal_slope,
                }
            )

    cv_df = pd.DataFrame(rows)

    # Summary (mean / std) across all folds and repeats
    summary_rows = []
    for metric in ["auc", "brier", "log_loss", "cal_intercept", "cal_slope"]:
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(cv_df[metric].mean()),
                "std": float(cv_df[metric].std()),
            }
        )
    cv_summary_df = pd.DataFrame(summary_rows)

    # Save JSON summary (similar to your previous style)
    results = {
        "n_splits": CV_FOLDS,
        "n_repeats": CV_REPEATS,
        "per_fold": rows,  # list of dicts
        "summary": {m["metric"]: {"mean": m["mean"], "std": m["std"]} for m in summary_rows},
    }

    out_json = os.path.join(OUT_DIR, "mrs90_lgbm_cv_metrics_raw_10x20.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== CV summary (RAW LightGBM, 10x20) ===")
    print(json.dumps(results["summary"], indent=2))
    print(f"\nSaved detailed CV metrics to: {out_json}")

    # Optional: AUC distribution histogram (TRIPOD-style)
    plt.figure(figsize=(7, 5))
    plt.hist(cv_df["auc"], bins=15, alpha=0.7)
    plt.xlabel("AUC (CV folds)")
    plt.ylabel("Count")
    plt.title(f"Cross-validated AUC (LightGBM, k={CV_FOLDS} x r={CV_REPEATS})")
    plt.tight_layout()
    auc_hist_path = os.path.join(OUT_DIR, "mrs90_lgbm_cv_auc_distribution_raw.png")
    plt.savefig(auc_hist_path, dpi=200)
    plt.close()
    print("Saved AUC distribution histogram:", auc_hist_path)


if __name__ == "__main__":
    main()
