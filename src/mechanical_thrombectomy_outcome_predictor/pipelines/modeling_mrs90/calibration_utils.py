# calibration_utils.py
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def compute_calibration_slope_intercept(y_true, proba):
    """
    Logistic regression of y on logit(p):

        logit(p_obs) ~ alpha + beta * logit(p_model)

    slope = beta, intercept = alpha
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


def apply_logistic_calibration(p_raw, a: float, b: float):
    """
    Apply Platt / logistic calibration:

        logit(p_cal) = a + b * logit(p_raw)
        p_cal = logistic(logit(p_cal))
    """
    eps = 1e-15
    p = np.clip(p_raw, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    z = a + b * logit_p
    return 1.0 / (1.0 + np.exp(-z))
