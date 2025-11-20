# decision_curve_mrs90.py
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use RAW LightGBM test predictions (from Kedro evaluate node)
PRED_PATH = "data/08_reporting/modeling_mrs90/evaluate/mrs90_test_predictions.parquet"
OUT_DIR = "data/08_reporting/modeling_mrs90/decision_curve"


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def decision_curve(y_true, proba, thresholds):
    """
    Compute net benefit for:
      - model
      - treat all
      - treat none

    Here 'positive' = good outcome (mRS 0–2).
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)

    N = len(y_true)
    event_rate = np.mean(y_true)

    rows = []
    for t in thresholds:
        pred_pos = proba >= t
        TP = np.sum((pred_pos == 1) & (y_true == 1))
        FP = np.sum((pred_pos == 1) & (y_true == 0))

        nb_model = (TP / N) - (FP / N) * (t / (1 - t))
        nb_all = event_rate - (1 - event_rate) * (t / (1 - t))
        nb_none = 0.0

        rows.append(
            {
                "threshold": float(t),
                "net_benefit_model": float(nb_model),
                "net_benefit_all": float(nb_all),
                "net_benefit_none": float(nb_none),
            }
        )

    return pd.DataFrame(rows)


def main():
    ensure_out_dir(OUT_DIR)

    print(f"Loading test predictions from: {PRED_PATH}")
    df_pred = pd.read_parquet(PRED_PATH)

    # Expect columns from evaluate_mrs90_lgbm()
    #   'y_true', 'y_pred_proba', 'y_pred_label', 'patient_id'
    y_true = df_pred["y_true"].values
    proba = df_pred["y_pred_proba"].values  # RAW LightGBM probabilities

    # Match regression-style thresholds: 0.05–0.75
    thresholds = np.linspace(0.05, 0.75, 15)
    dca_df = decision_curve(y_true, proba, thresholds)

    # Save raw numbers
    out_csv = os.path.join(OUT_DIR, "mrs90_decision_curve_raw.csv")
    dca_df.to_csv(out_csv, index=False)
    print("Saved DCA table:", out_csv)

    # Plot decision curve
    plt.figure()
    plt.plot(
        dca_df["threshold"],
        dca_df["net_benefit_model"],
        label="LightGBM (raw)",
    )
    plt.plot(
        dca_df["threshold"],
        dca_df["net_benefit_all"],
        linestyle="--",
        label="Treat all",
    )
    plt.plot(
        dca_df["threshold"],
        dca_df["net_benefit_none"],
        linestyle=":",
        label="Treat none",
    )
    plt.xlabel("Threshold probability for mRS 0–2")
    plt.ylabel("Net benefit")
    plt.title("Decision curve – mRS90 LightGBM model (TEST, raw)")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)

    out_png = os.path.join(OUT_DIR, "mrs90_decision_curve_raw.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved decision curve plot:", out_png)


if __name__ == "__main__":
    main()
