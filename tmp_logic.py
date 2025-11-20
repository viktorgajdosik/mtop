import pandas as pd
import numpy as np

df = pd.read_parquet(r"data\02_intermediate\mt_patients_clean.parquet")

violations = {}

if "ivt_given" in df.columns and "onset_to_ivt_min" in df.columns:
    m = (df["ivt_given"].isin([0, 0.0]).fillna(False)) & df["onset_to_ivt_min"].notna()
    violations["ivt_time_present_despite_no_ivt"] = int(m.sum())

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
    violations["duplicate_patient_id"] = int(df["patient_id"].duplicated().sum())

print(violations)
