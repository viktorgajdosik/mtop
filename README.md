Mechanical Thrombectomy Outcome Predictor (MTOP)
mRS90 LightGBM Pipeline & Clinical Calculator
1. Purpose and Scope

This repository implements a mechanical thrombectomy outcome predictor for ischemic stroke, focused on:

Outcome: good functional outcome at 90 days, defined as mRS 0–2 (“mRS90”).

Predictors: pre-procedural variables only (demographic, baseline clinical, imaging, and pre-procedural timing/treatment data). No intra-procedural or post-procedural information is used.

Model: Gradient-boosted decision trees (LightGBM) with clinically motivated monotone constraints.

Cohort: Single-centre, retrospective thrombectomy cohort (patients in mt_patients_master.xlsx).

The project is structured as a Kedro pipeline that:

Cleans the raw Excel registry.

Applies strict data-quality checks and exclusions.

Prepares a LightGBM-ready feature table.

Splits the data temporally into development and internal test sets.

Trains and evaluates a monotone-constrained LightGBM model for mRS90.

Produces comprehensive analytics: cross-validation, calibration, decision-curve analysis, SHAP explainability.

Exposes the final model via a password-protected Streamlit clinical calculator.

Important: This is a research tool only. It is not a clinical decision support system and must not be used to guide individual patient care.

2. High-Level Data Flow

Conceptually, the pipeline looks like this:

Raw registry (Excel)
   │
   ▼
[Data Cleaning & Validation]
   - clean_mt_patients
   - exclude_invalid_cases
   - validate_and_fix
   │
   ├─► EDA & audit (eda_audit) → data_profile, figures
   │
   ▼
Validated cohort
   │
   ├─► make_lightgbm_ready  → mt_lightgbm_ready.parquet
   └─► make_regression_ready → mt_patients_regression_ready.parquet (for logistic models)

   ▼
[Temporal split]
   - create_temporal_split (mt_patients_split.parquet)

   ▼
[LightGBM mRS90 pipeline (modeling_mrs90)]
   - build_mrs90_dataset  → mrs90_X_train / y_train / X_test / y_test / IDs / feature_list
   - train_mrs90_lgbm     → mrs90_lgbm_model, train_metrics
   - evaluate_mrs90_lgbm  → test_metrics, test_predictions
   - mrs90_feature_importance
   - compute_mrs90_shap
   - plot_mrs90_shap, plot_mrs90_shap_age_groups

   ▼
[Standalone analysis scripts]
   - cv_mrs90_evaluation.py           → CV metrics & AUC histogram
   - calibration_mrs90_analysis.py    → calibration metrics & plot
   - decision_curve_mrs90.py          → decision curve CSV & plot

   ▼
[Streamlit app]
   - app_mrs90.py loads:
       mrs90_lgbm_model.pkl
       mrs90_X_train.parquet
       mt_lightgbm_ready.parquet
       calibration & CV & DCA outputs
   - Exposes a password-protected prognostic calculator for a single patient.

3. Project Structure (Relevant to mRS90 LightGBM)

Only key parts are described; the tree is shortened to what is most relevant.

conf/
  base/
    catalog.yml         # Dataset registry for Kedro
    column_map.yml      # Mapping from raw Excel column names → standardized names
    parameters.yml      # (Reserved for pipeline parameters)
    value_maps.yml      # Value translations (e.g. Czech → English; yes/no tokens)
  local/
    credentials.yml     # Local secrets (not committed)

data/
  01_raw/
    mt_patients_master.xlsx        # Original thrombectomy registry
  02_intermediate/
    mt_patients_clean.parquet      # Cleaned but not yet excluded
  03_primary/
    mt_patients_valid.parquet      # After hard exclusions
    mt_patients_validated.parquet  # After range checks / fixes
    mt_patients_split.parquet      # Validated + temporal split labels
  05_model_input/
    mt_lightgbm_ready.parquet              # Main LightGBM feature table
    mt_patients_regression_ready.parquet   # Single imputed dataset for regression
  06_models/
    modeling_mrs90/
      mrs90_X_train.parquet
      mrs90_y_train.parquet
      mrs90_X_test.parquet
      mrs90_y_test.parquet
      mrs90_train_ids.parquet
      mrs90_test_ids.parquet
      mrs90_feature_list.json
      mrs90_lgbm_model.pkl                 # Final LightGBM model
  08_reporting/
    baseline_characteristics/
      baseline_table1.csv
      baseline_smd.csv
    data_cleaning/
      cleaning_log.json
      exclusion_log.json
      data_quality_violations.json
      imputation/regression_imputation_log.json
    eda_audit/
      eda_audit_report.json
      data_profile.json
      figs/* (NIHSS, mRS90, TICI plots)
    modeling_mrs90/
      cv/
        mrs90_lgbm_cv_metrics_raw_10x20.json
        mrs90_lgbm_cv_auc_distribution_raw.png
      calibration/
        mrs90_calibration_metrics_raw.json
        mrs90_calibration_curve_raw.png
      decision_curve/
        mrs90_decision_curve_raw.csv
        mrs90_decision_curve_raw.png
      evaluate/
        mrs90_train_metrics.json
        mrs90_test_metrics.json
        mrs90_test_predictions.parquet
        mrs90_feature_importance.parquet
      shap/
        mrs90_shap_*.png / *.parquet / *.pkl / expected_value.json

src/
  mechanical_thrombectomy_outcome_predictor/
    pipeline_registry.py        # Orchestrates all pipelines
    pipelines/
      data_cleaning/
      data_splitting/
      eda_audit/
      baseline_characteristics/
      modeling_mrs90/
      mt_regression/            # Logistic regression project (separate)
app_mrs90.py                    # Streamlit app for mRS90 LightGBM model

4. From Raw Data to LightGBM-Ready Table
4.1 Raw data (data/01_raw)

mt_patients_master.xlsx
Single-centre thrombectomy registry exported from the hospital system.
Contains all original columns in a mixture of languages and formats.

4.2 Data cleaning pipeline (data_cleaning)

Implemented in src/mechanical_thrombectomy_outcome_predictor/pipelines/data_cleaning.

The key nodes are:

clean_mt_patients

Input:

mt_patients_raw → data/01_raw/mt_patients_master.xlsx

column_map → conf/base/column_map.yml

value_maps → conf/base/value_maps.yml

Output:

mt_patients_clean.parquet

cleaning_log.json

Main responsibilities:

Normalize column names (strip, lower, snake_case).

Rename to standard clinical variables using column_map.yml.

Translate values (Czech → English, harmonize categories) using value_maps.yml.

Normalize missing value markers (e.g. "NA", "n/a", "?", empty strings → NaN).

Normalize TICI scores to an ordered categorical: 0, 1, 2a, 2b, 2c, 3.

Coerce key clinical variables to numeric (age, ASPECTS, delays, NIHSS, mRS, BMI, BP, cholesterol, glycemia), logging how many values were coerced to NaN.

Normalize many yes/no fields to binary (0/1) via _to_binary.

Collapse multi-category heart_condition into a binary variable:

0 – “No finding” / normal heart

1 – any documented cardiac pathology

Enforce consistency rules for IV thrombolysis (e.g., if ivt_given = 0, then onset_to_ivt_min must be missing).

Produce detailed logs of variable renaming, missingness, and coercion.

From a clinical perspective: this step ensures the registry behaves like a clean, analyzable dataset, with obvious data entry artefacts removed and key variables interpretable.

exclude_invalid_cases

Input: mt_patients_clean
Output: mt_patients_valid, exclusion_log.json

Exclusions based on physiologically or logically impossible values, e.g.:

Onset-to-IVT > Onset-to-puncture.

Onset-to-puncture > Onset-to-recanalization.

ASPECTS > 10.

Any NIHSS measure > 42.

This creates a cohort where basic temporal logic and clinical plausibility are enforced, and records that violate these rules are removed completely.

validate_and_fix

Input: mt_patients_valid
Output: mt_patients_validated, data_quality_violations.json

This step keeps the cohort but internally corrects remaining implausible values, such as:

NIHSS 7d > 99 → blanked; 42–99 flagged.

mRS values outside 0–6 → set to NaN and cast to Int64.

ASPECTS outside 0–10 → blanked.

Binary variables cast to compact Int8.

The accompanying JSON (data_quality_violations.json) summarizes remaining issues and corrections.

make_lightgbm_ready

Input: mt_patients_validated
Output: mt_lightgbm_ready.parquet

This creates the main modeling table for LightGBM:

Adds onset_year from onset_date (used by the temporal split and model).

For key numeric variables (age, ASPECTS, onset delays, NIHSS, mRS, BMI, BP, cholesterol, glycemia), it adds explicit *_missing flags (0/1).

For key categorical/binary variables (sex, side, occlusion site, IVT status, risk factors, heart condition, etc.), it also adds *_missing flags.

LightGBM is able to handle missing values internally; here, missing values are not imputed but missingness is made explicit as a potential predictor.

make_regression_ready

Input: mt_lightgbm_ready
Output: mt_patients_regression_ready.parquet, regression_imputation_log.json

This creates a single complete-case dataset for logistic regression models:

Numeric variables: imputed using MICE (IterativeImputer); log records iterations and success.

Categorical variables: imputed using mode (or “Unknown” if no clear mode).

Missingness flags (*_missing) are preserved, not imputed.

Logs before/after missingness per column.

This pipeline is used for the separate mt_regression project; the mRS90 LightGBM pipeline continues from mt_lightgbm_ready.

5. Temporal Split (Internal Validation Design)
create_temporal_split (data_splitting)

File: src/.../pipelines/data_splitting/nodes.py
Output: mt_patients_split.parquet

Logic:

Parses onset_date to an onset year.

Assigns each record to:

train – if onset_year ≤ train_end_year (default 2022).

test – if onset_year > train_end_year.

Records with unparseable or missing onset dates are conservatively assigned to train, to avoid future information leaking into the training set.

This provides a temporal internal validation, which is stronger than random splitting and closer to real-world deployment (past patients for training, newer patients for testing).

6. mRS90 LightGBM Modeling Pipeline
6.1 Kedro pipeline definition

File: src/.../pipelines/modeling_mrs90/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import (
    build_mrs90_dataset,
    train_mrs90_lgbm,
    evaluate_mrs90_lgbm,
    mrs90_feature_importance,
    compute_mrs90_shap,
    plot_mrs90_shap,
    plot_mrs90_shap_age_groups,
)


The pipeline steps:

build_mrs90_dataset
Inputs:

mt_lightgbm_ready

mt_patients_split (with split = train/test)

Outputs:

mrs90_X_train, mrs90_y_train

mrs90_X_test, mrs90_y_test

mrs90_train_ids, mrs90_test_ids

mrs90_feature_list

Conceptually, this node:

Filters the cohort to patients with known mRS90.

Defines the binary outcome (e.g. good_mrs = 1 if mRS90 ∈ {0,1,2}, else 0).

Uses the temporal split label to separate training vs test sets.

Subsets to a curated set of predictors and stores them in mrs90_feature_list.json.

Writes all arrays to data/06_models/modeling_mrs90/.

train_mrs90_lgbm
Inputs:

mrs90_X_train

mrs90_y_train

Outputs:

mrs90_lgbm_model.pkl

mrs90_train_metrics.json

Behaviour:

Fits a LightGBM classifier (LGBMClassifier) with monotone constraints (see below).

Stores basic training metrics (AUC, Brier score, log-loss) in JSON.

Serializes the trained model into mrs90_lgbm_model.pkl, which is later loaded by the Streamlit app.

evaluate_mrs90_lgbm
Inputs:

mrs90_lgbm_model

mrs90_X_test, mrs90_y_test

mrs90_test_ids

Outputs:

mrs90_test_metrics.json

mrs90_test_predictions.parquet

Behaviour:

Computes test-set performance:

ROC AUC

Brier score

Log-loss

Possibly sensitivity/specificity or confusion counts at default thresholds.

Stores per-patient predictions (y_true, y_pred_proba, y_pred_label, patient_id) in mrs90_test_predictions.parquet.

mrs90_feature_importance
Inputs:

mrs90_lgbm_model

mrs90_feature_list

Output:

mrs90_feature_importance.parquet

Behaviour:

Extracts LightGBM feature importances (typically gain-based) and stores them in a tidy table.

compute_mrs90_shap

Inputs:

mrs90_lgbm_model

mrs90_X_train, mrs90_X_test

mrs90_feature_list

Outputs (multiple datasets):

Raw SHAP values for train/test (mrs90_shap_train_values.pkl, mrs90_shap_test_values.pkl).

SHAP expected value (mrs90_shap_expected_value.json).

Aggregated SHAP summaries:

mrs90_shap_summary_train.parquet

mrs90_shap_summary_test.parquet

mrs90_shap_summary_test_by_age.parquet (age-stratified summaries).

These are used to quantify how much each predictor contributes to prognosis and how this varies across age groups.

plot_mrs90_shap and plot_mrs90_shap_age_groups

Inputs:

SHAP values

mrs90_X_test

mrs90_feature_list

Outputs:

Global plots:

mrs90_shap_barplot.png – ranked feature importance (mean |SHAP|).

mrs90_shap_beeswarm.png – global SHAP distribution.

mrs90_shap_dependence_top_feature.png – dependence plot for the top feature.

Age-stratified SHAP plots:

mrs90_shap_age_lt50_ge50.png

mrs90_shap_age_lt75_ge75.png

mrs90_shap_age_lt80_ge80.png

mrs90_shap_age_lt85_ge85.png

For clinicians, these plots show which variables drive risk estimates and how their impact changes for younger vs older patients.

7. Model Specification: Monotone-Constrained LightGBM

The standalone scripts (cv_mrs90_evaluation.py, calibration_mrs90_analysis.py) explicitly define the model used for evaluation:

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


Monotone constraints encode clinical expectations about how predictors should affect the probability of good outcome (mRS 0–2):

age: −1 → higher age should not increase the probability of good outcome.

admission_nihss: −1 → higher stroke severity should not increase the probability of good outcome.

onset_to_puncture_min: −1 → longer delay to puncture should not improve outcomes.

aspects: 0 → allowed to be non-monotone (though clinically higher ASPECTS is generally better).

mrs_before: 0 → effect constrained to be flexible (baseline disability may have complex relationships).

glycemia: 0 or −1 depending on RELAX_GLYCEMIA_CONSTRAINT switch.

Interaction terms (e.g., age_x_nihss) are left unconstrained (0), allowing the model to learn synergistic or antagonistic effects while keeping main effects clinically interpretable.

8. Analytics & Outcomes for the mRS90 Model

All major analytics are stored under data/08_reporting/modeling_mrs90/.

8.1 Cross-validation (internal reproducibility)

Script: cv_mrs90_evaluation.py
Outputs:

mrs90_lgbm_cv_metrics_raw_10x20.json

10-fold StratifiedKFold, repeated 20× (10×20 CV).

For each fold:

AUC, Brier score, log-loss.

Calibration slope and intercept (via logistic regression of outcome on logit(p)).

Summary section with mean and standard deviation for each metric.

mrs90_lgbm_cv_auc_distribution_raw.png

Histogram of AUC values across all folds and repeats, showing the distribution of performance rather than a single number.

From a clinician’s perspective: this provides an estimate of how stable the model’s discrimination is when refitted on different subsets of the same cohort. High variability would caution against over-interpreting a single AUC.

8.2 Calibration (are probabilities trustworthy?)

Script: calibration_mrs90_analysis.py
Outputs:

Per-split metrics in mrs90_calibration_metrics_raw.json:

Train and test AUC.

Brier score.

Log-loss.

Calibration slope and intercept.

Convenience aliases:

roc_auc_test

brier_test

log_loss_test

calibration_intercept_test

calibration_slope_test

Calibration curve (mrs90_calibration_curve_raw.png):

Plots observed vs predicted probability (in deciles of predicted risk).

Includes the reference 45° line (“perfect calibration”).

Clinically:

Calibration intercept gives an idea if the model tends to over- or under-estimate the absolute risk on average.

Calibration slope indicates if predictions are too extreme (<1) or too conservative (>1).

8.3 Decision-curve analysis (clinical utility)

Script: decision_curve_mrs90.py
Inputs:

mrs90_test_predictions.parquet (true outcome, predicted probabilities).

Outputs:

mrs90_decision_curve_raw.csv – net benefit values at thresholds 0.05–0.75.

mrs90_decision_curve_raw.png – plot of net benefit for:

The model.

“Treat all” (assume all have good prognosis).

“Treat none”.

This addresses the question: “If I use this model to make decisions at a given risk threshold, do I gain more than treating everyone or no-one?”

The Streamlit app further summarizes the range of thresholds where the model has positive net benefit and outperforms both treat-all and treat-none.

8.4 SHAP explainability

Files in data/08_reporting/modeling_mrs90/shap/ include:

Raw and aggregated SHAP values (*.pkl, *_summary_*.parquet).

Global importance and beeswarm plots:

mrs90_shap_barplot.png

mrs90_shap_beeswarm.png

mrs90_shap_dependence_top_feature.png

Age-stratified plots (mrs90_shap_age_lt..._ge....png).

Clinically, these show:

Which variables most strongly drive the predicted probability of good outcome overall.

How the impact of key predictors (e.g., age, NIHSS, onset-to-puncture) varies across age strata.

9. Current Stage of the LightGBM Model

Completed:

Deterministic data cleaning with documented logs (TRIPOD-aligned).

Exclusion of implausible cases and internal validation with a temporal split.

Creation of mt_lightgbm_ready and mt_patients_regression_ready.

Training of a monotone-constrained LightGBM mRS90 model.

Internal evaluation with:

Held-out temporal test set.

Extensive cross-validation (10×20).

Calibration metrics and curve.

Decision-curve analysis.

Global and age-stratified SHAP analyses.

Implementation of a password-protected Streamlit application deploying the finalized model.

Not yet done / limitations:

External validation:
The model has only been validated internally on a single centre. Performance and calibration may differ in other hospitals or populations (different case mix, treatment pathways, imaging protocols).

Prospective validation:
No prospective evaluation in a real-time clinical workflow has been performed yet.

Recalibration for other settings:
Although the calibration intercept/slope are estimated, no recalibration to external populations is implemented.

Outcome definition:
Outcome is mRS at 90 days (0–2 vs 3–6) from this registry. Any systematic bias or missingness in follow-up will propagate to the model.

Pre-procedural scope only:
By design, the model does not use intra-procedural data (e.g. TICI, recanalization time, complications), limiting it to pre-procedural prognostication. This is appropriate for early counselling and triage, but not for intra- or post-procedural decision-making.

Single model version:
The repository currently exposes a single final LightGBM model. There is no model versioning or automatic retraining pipeline yet.

Fairness / subgroup analysis beyond age:
Some age-stratified SHAP analysis is present, but broader fairness analysis (sex, comorbidities, centres) is not yet implemented.

10. Streamlit Clinical Calculator (app_mrs90.py)

The Streamlit app provides an interactive front-end to the mRS90 LightGBM model.

10.1 Security: password gate

The app checks st.secrets["app_password"]:

If set, it prompts for a password and stores a session-level “password_correct” flag.

If not set (local development), it warns that the app is running without authentication.

Password is never stored in plaintext in the session.

10.2 Model and data loading

At startup, the app loads:

mrs90_lgbm_model.pkl – final LightGBM model.

mrs90_X_train.parquet – used to:

Determine variable types.

Compute medians/modes for default values.

Provide reference distributions and SHAP fallback values.

mt_lightgbm_ready.parquet (as raw_df) – to map categorical encodings back to human-readable labels when possible.

mrs90_calibration_metrics_raw.json – to display AUC, Brier score, calibration intercept/slope.

mrs90_lgbm_cv_metrics_raw_10x20.json – to display cross-validation summary.

mrs90_decision_curve_raw.csv – for decision-curve summaries and plots.

All these are cached via st.cache_resource / st.cache_data for performance.

10.3 Pre-procedural data entry interface

The app organizes inputs into clinically intuitive sections:

Demographics

age, sex, bmi.

Baseline clinical

Vascular risk factors & history:

hypertension, diabetes, hyperlipidemia, smoking, alcohol_abuse,

arrhythmia, tia_before, cmp_before, heart_condition, statins_before.

Baseline vitals & labs:

systolic_bp, diastolic_bp, glycemia, cholesterol, bmi.

Baseline disability & severity:

mrs_before,

admission_nihss (or nihss_admission depending on naming).

Imaging

aspects,

occlusion_site,

hemisphere.

Treatment / timing

ivt_given (IV thrombolysis yes/no/missing).

thrombolytics (type of thrombolytic agent; gated by IVT status).

ivt_different_hospital (yes/no/missing; gated by IVT status).

transfer_from_other_hospital (yes/no).

Onset-to-IVT delay (if IVT given):

onset_to_ivt_min.

Onset-to-puncture delay:

onset_to_puncture_min.

Etiology:

etiology (e.g. cardioembolic, large artery atherosclerosis, etc., depending on registry).

UI behaviour

Numeric fields:

Use sliders with automatically chosen ranges based on the 1st–99th percentile of the training data.

Each numeric variable has a “missing” checkbox:

If checked, the value is treated as NaN and the underlying *_missing indicator will be set accordingly.

Binary variables:

Presented as tri-state select boxes: “Missing / No (0) / Yes (1)”.

Categorical (multi-state) variables:

Options derived from the raw data via mapping (e.g. occlusion site, hemisphere, thrombolytic type).

Include an explicit “Missing” category.

IVT-related fields:

If IVT information missing is checked:

ivt_given is set to NaN.

IVT-dependent fields are disabled and set to NaN.

If IVT given = False:

IVT-dependent fields (thrombolytic type, different hospital, onset-to-IVT) are disabled.

If IVT given = True:

IVT-dependent fields become active.

Behind the scenes, the app uses helper functions to:

Detect whether a column is truly binary (is_binary_numeric).

Build a complete patient feature vector (build_patient_row) that:

Fills any unspecified features with median (numeric) or mode (categorical) from the training set.

Respects explicit user choices of “Missing”.

Recomputes all *_missing features based on actual NaN values in the assembled row.

Produce a SHAP-safe version of the row (make_shap_safe_row) where any remaining NaNs are filled using training medians/modes (for SHAP only; predictions use the original row).

10.4 Prediction & explanation

When the user clicks:

“Compute probability of good outcome (mRS 0–2)”

The app:

Constructs the feature vector for the patient.

Computes the predicted probability:

proba_good = P(mRS 0–2 | pre-procedural data)

proba_bad = 1 − proba_good

Displays:

Good outcome (mRS 0–2) as a percentage.

Poor outcome (mRS 3–6) as a percentage.

If Debug mode is enabled:

Shows the full encoded feature vector for the patient.

If SHAP is available:

Builds a SHAP explainer (TreeExplainer) for the model.

Computes SHAP values for the current patient.

Shows:

A dataframe of features with values and SHAP contributions.

A bar chart of top contributors for this case.

The SHAP base value (log-odds for a reference patient).

This allows technically inclined clinicians to see which variables are driving the prediction for a specific patient.

10.5 Embedded performance summary in the app

The app reads mrs90_calibration_metrics_raw.json, mrs90_lgbm_cv_metrics_raw_10x20.json, and mrs90_decision_curve_raw.csv and presents:

ROC AUC on the test set, with a narrative explanation of discrimination.

Brier score, calibration slope and intercept, with explanation of calibration.

Decision-curve summary:

The range of thresholds where the model offers positive net benefit vs treat-all / treat-none.

A line chart of net benefit curves.

An additional expandable section explains how to interpret these metrics as a clinician, including:

What AUC means in terms of ranking patients.

How calibration intercept/slope inform trust in the absolute probabilities.

How decision curves link predicted risk to clinical decisions (e.g., thresholds at which a prognostic model might influence counselling or intensity of care).

11. Summary

The repository implements a transparent, reproducible ML pipeline for predicting 90-day functional outcome (mRS 0–2) in mechanical thrombectomy patients using only pre-procedural predictors.

It includes:

Rigorous cleaning and validation of the stroke registry.

A temporally split, monotone-constrained LightGBM model with extensive internal validation.

Rich reporting on discrimination, calibration, clinical utility, and variable importance.

A password-protected Streamlit calculator suitable for clinician-facing demonstrations and internal research use.

Before any real-world clinical use, the model requires external and prospective validation, potential recalibration, and regulatory review.