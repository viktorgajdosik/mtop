# config.py
from __future__ import annotations

from typing import Dict, List

# -------------------------------------------------------------------
# Global switches for experiments
# -------------------------------------------------------------------


# === 1) Monotone constraints ===
# Monotonic constraint on glycemia:
#   False -> monotone decreasing (higher glycemia → worse outcome)
#   True  -> no constraint (0) and let the model learn freely.
RELAX_GLYCEMIA_CONSTRAINT: bool = False


def get_monotone_sign_map() -> Dict[str, int]:
    """
    Clinical prior signs for monotone constraints.
    Effect is on probability of GOOD outcome (mrs90_good = 1).
    """
    gly_sign = 0 if RELAX_GLYCEMIA_CONSTRAINT else -1

    return {
        # core clinical monotone effects
        "age": -1,                     # older -> lower chance of good outcome
        "admission_nihss": -1,         # more severe stroke -> worse
        "mrs_before": 0,               # worse baseline disability -> worse
        "onset_to_ivt_min": 0,         # longer delay -> worse
        "onset_to_puncture_min": -1,   # longer delay -> worse
        "aspects": +1,                 # higher ASPECTS -> better
        "glycemia": gly_sign,          # constrained or relaxed glycemia

        # interactions (if present)
        "age_x_nihss": 0,
        "onset_to_puncture_min_x_nihss": 0,
        "age_x_onset_to_puncture_min": 0,
    }


def build_monotone_constraints(feature_list: List[str]) -> List[int]:
    """
    Build the LightGBM monotone_constraints vector from feature names.
    Unknown features default to 0 (no constraint).
    """
    sign_map = get_monotone_sign_map()
    monotone_constraints = [sign_map.get(f, 0) for f in feature_list]

    if len(monotone_constraints) != len(feature_list):
        raise ValueError(
            f"monotone_constraints length {len(monotone_constraints)} "
            f"does not match n_features {len(feature_list)}"
        )

    return monotone_constraints
