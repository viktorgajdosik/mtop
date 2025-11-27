Mechanical Thrombectomy Outcome Predictor (MTOP) Technical Overview
The MTOP project is a data-driven prognostic tool developed in Python (Kedro pipeline) to predict 90-day functional outcome (modified Rankin Scale 0–2, “good outcome”) after mechanical thrombectomy. It follows a rigorous pipeline to clean registry data, engineer features, train and evaluate a monotonic LightGBM model, calibrate probabilities, and generate clinical reporting and a Streamlit app. All steps are designed with TRIPOD reproducibility in mind (transparent reporting of a prediction model). The code uses tools such as Pandas, NumPy, Scikit-learn, LightGBM, Optuna, SHAP, Matplotlib, and Streamlit.
Project Structure
Only the most relevant files and folders are highlighted below (the structure is based on Kedro conventions):
•	conf/ – Configuration files
•	base/ – Base configuration (committed)
o	catalog.yml – Catalog of all data inputs/outputs and their file paths (Kedro DataSets)
o	column_map.yml – Mapping of raw Excel column names to standardized names.
o	parameters.yml – (Currently not heavily used; reserved for tunable parameters)
o	value_maps.yml – Definitions for translating raw categorical values (e.g., language translations, yes/no encodings).
•	local/ – Local configuration (not committed)
o	credentials.yml – Secrets or credentials (e.g., for the Streamlit app password)
o	(Also logging.yml for logging config and a README.md specific to config usage)
•	data/ – Data sets tracked by Kedro (not all included in repo due to size/privacy)
•	01_raw/ – Raw data (input)
o	mt_patients_master.xlsx – Original thrombectomy registry export (contains all patient data pre-cleaning).
•	02_intermediate/ – Intermediate data (after initial cleaning)
o	mt_patients_clean.parquet – Cleaned data (normalized and translated, but prior to exclusions).
•	03_primary/ – Primary data sets (after exclusions and validation)
o	mt_patients_valid.parquet – Data after dropping invalid records (hard exclusions).
o	mt_patients_validated.parquet – Data after value range checks/fixes (no new exclusions, but values adjusted or blanked as needed).
o	mt_patients_split.parquet – Same as validated data but with an added split column (train/test assignment per temporal rule).
•	04_feature/ – (Unused in this project; reserved for feature-engineering outputs in other projects.)
•	05_model_input/ – Final input data for modeling
o	mt_lightgbm_ready.parquet – Main modeling table for LightGBM (post-cleaning, with engineered features and missingness indicators, no imputation).
o	(Optionally mt_patients_regression_ready.parquet would live here if logistic regression analysis was performed, but that pipeline is deprecated.)
•	06_models/ – Stored trained model artifacts
o	modeling_mrs90/ – Directory for all mRS90 LightGBM model artifacts
o	mrs90_X_train.parquet, mrs90_y_train.parquet – Training set features and labels (post-split).
o	mrs90_X_test.parquet, mrs90_y_test.parquet – Test set features and labels.
o	mrs90_train_ids.parquet, mrs90_test_ids.parquet – Lists of patient IDs in train and test sets.
o	mrs90_feature_list.json – List of feature names used in modeling (columns of X matrices).
o	mrs90_lgbm_model.pkl – Serialized final LightGBM model object.
o	mrs90_lgbm_optuna_best_params.json – Best hyperparameters found via Optuna tuning (including flags for interactions, etc.).
•	07_model_output/ – (Reserved for model predictions or outputs; not heavily used here since outputs are in 08_reporting)
•	08_reporting/ – All analysis outputs, reports, and figures for review/reporting
o	baseline_characteristics/ – Overall cohort baseline comparison between Train vs Test
o	baseline_table1.csv – “Table 1” style summary of baseline variables in training vs test set (mean±SD or counts, plus SMD).
o	baseline_smd.csv – Simplified table of standardized mean differences for each variable (sorted by imbalance).
o	data_cleaning/ – Logs from data cleaning pipeline
o	cleaning_log.json – Detailed log of cleaning actions (renaming, coercions, translations, etc.).
o	exclusion_log.json – Summary of exclusions made (count of records removed for each rule, etc.).
o	data_quality_violations.json – Log of range or plausibility issues found and fixed (violations of ranges, etc.).
o	imputation/regression_imputation_log.json – (Log for regression imputation step, if it were used).
o	eda_audit/ – Outputs of the exploratory data analysis and audit pipeline
o	eda_audit_report.json – Summary report of the data audit (distribution of variables, outliers, etc.).
o	data_profile.json – Combined data profile (merging cleaning logs and EDA info for an overview of data quality).
o	figs/ – Directory of generated figures (matplotlib) for key distributions:
	e.g., nihss_7d_hist.png, mrs90_bar.png, tici_bar.png – Histograms or bar charts illustrating NIHSS, mRS90 outcome distribution, TICI scores, etc., for audit purposes.
o	modeling_mrs90/ – Outputs related to model performance and analysis
o	cv/ – Cross-validation metrics
	mrs90_lgbm_cv_metrics_raw_10x20.json – Metrics for 10×20 repeated stratified CV on the training set (each fold’s AUC, Brier, etc., plus summary stats).
	mrs90_cv_summary_table.csv – Tabular summary of CV results (mean±SD of metrics) for easy reference.
	mrs90_lgbm_cv_auc_distribution_raw.png – Histogram of AUCs across all 200 CV runs (visualizing performance stability).
o	calibration/ – Calibration analysis
	mrs90_calibration_metrics_raw.json – Calibration metrics on train vs test (AUC, Brier, log-loss, calibration intercept and slope) before any recalibration.
	mrs90_logistic_calibration_params.json – Parameters (slope and intercept) for logistic calibration (Platt scaling) derived from the training set.
	mrs90_calibration_curve_raw.png – Calibration curve plot (observed vs predicted probability in deciles) for the raw model on test.
	mrs90_calibration_curve_raw_vs_calibrated.png – Calibration curves comparing raw vs after logistic calibration (showing improvement in alignment to the 45° line).
o	decision_curve/ – Decision Curve Analysis for clinical utility
	mrs90_decision_curve_raw.csv – Net benefit values of the model across probability thresholds (raw model predictions).
	mrs90_decision_curve_raw.png – Plot of decision curves for the raw model vs “treat-all” vs “treat-none”.
	mrs90_decision_curve_calibrated.csv – Net benefit values if model outputs are calibrated (adjusted probabilities) before thresholding.
	mrs90_decision_curve_calibrated.png – Decision curve plot for the calibrated model.
	Clinically, comparing raw vs calibrated decision curves can show if calibration changes the range of thresholds where using the model adds value.
o	evaluate/ – Test set evaluation outputs
	mrs90_train_metrics.json – Training performance metrics (AUC, Brier score, log-loss on train, etc.).
	mrs90_test_metrics.json – Test set performance metrics (AUC, Brier, log-loss on held-out test).
	mrs90_test_predictions.parquet – Per-patient test set predictions (with true outcome, predicted probability, and predicted class label).
	mrs90_feature_importance.parquet – Feature importance scores from the LightGBM model (e.g., gain or split importance for each feature).
o	performance/ – Threshold-based performance metrics
	mrs90_performance_summary.csv – Summary of model discrimination and calibration on test (e.g. AUC, Brier, calibration slope/intercept for raw model, etc., possibly combining info from above).
	mrs90_threshold_performance.csv – Detailed performance at one or more probability thresholds (e.g., confusion matrix metrics like sensitivity, specificity, PPV at a chosen threshold like 0.5, potentially both raw and calibrated probabilities).
o	shap/ – SHAP (SHapley Additive exPlanations) outputs for model explainability
	mrs90_shap_train_values.pkl, mrs90_shap_test_values.pkl – Raw SHAP value arrays for all training and test set samples (pickle format, large).
	mrs90_shap_expected_value.json – The SHAP “expected value” of the model (base log-odds or probability for an average patient).
	mrs90_shap_summary_train.parquet, mrs90_shap_summary_test.parquet – Precomputed summary of SHAP values by feature (mean, std, etc.) for train and test sets.
	mrs90_shap_summary_test_by_age.parquet – SHAP summary broken down by age groups (to examine how feature impacts differ in younger vs older patients).
	Global SHAP plots:
	mrs90_shap_barplot.png – Bar plot of top features by mean |SHAP| (global feature importance).
	mrs90_shap_beeswarm.png – Beeswarm plot of SHAP values for all features (distribution of impacts).
	mrs90_shap_dependence_top_feature.png – SHAP dependence plot for the highest-impact feature (effect vs feature value).
	Age-stratified SHAP plots (comparing SHAP distributions in younger vs older patients for key age cutoffs):
	mrs90_shap_age_lt50_ge50.png, ...lt75_ge75.png, ...lt80_ge80.png, ...lt85_ge85.png – Beeswarm-like plots split by age group, highlighting interactions between age and other predictors.
	For clinicians, the SHAP outputs identify which variables most strongly drive the model’s predictions and how their influence may change with patient age.
o	spec/ – Model specification details
	mrs90_model_spec.json – Machine-readable model specification including model parameters, selected features, and any constraints (monotonicity signs), to document exactly how the model was built.
o	tripod/ – Compiled TRIPOD report outputs
	mrs90_tripod_excel.xlsx – An Excel workbook containing multiple sheets summarizing the entire modeling process and results (cohort flow, outcome availability, missingness, predictor dictionary, model performance, etc.) for reporting transparency.
o	reporting_mrs90/ – Final tabular outputs specific to mRS90 model reporting (also contained in the TRIPOD Excel above)
o	mt_cohort_flow_table.csv – Table of patient counts at each step of data processing (for CONSORT-style diagram and reporting).
o	mt_cohort_flow_diagram.png – Flow diagram illustrating the cohort selection (a visual representation of the above table).
o	mrs90_outcome_availability.csv – Counts and percentages of patients with 90-day mRS available, overall and by train/test split.
o	mrs90_missingness_table.parquet – Detailed table of missing data by variable (overall, in train, in test; flags for whether a variable has a companion “_missing” indicator, etc.).
o	mrs90_baseline_by_outcome.csv – Baseline characteristics of the training set, stratified by 90-day outcome (mRS 0–2 vs 3–6), including group means/counts and standardized mean differences for each predictor.
o	mrs90_predictor_dictionary.xlsx – Data dictionary for all predictors, including data type, allowed range or categories, whether a monotonic constraint is applied (and direction), whether a missing-value flag exists, and whether the predictor was actually used in the final model.
o	mrs90_cv_summary_table.csv – Summary of cross-validation performance (aggregate of the 10×20 CV results for the final model parameters).
o	mrs90_performance_summary.csv – Summary of final model performance on the internal test set (AUC, Brier, etc., and possibly calibration metrics).
o	mrs90_threshold_performance.csv – Performance metrics at specific probability threshold(s) on the test set (e.g., confusion matrix outcomes like sensitivity, specificity for a chosen operating point, helping to contextualize the model’s output clinically).
o	mrs90_dca_summary.csv – Summary of decision curve analysis results (e.g., the range of risk thresholds at which the model has positive net benefit and surpasses treat-all or treat-none strategies).
•	src/ – Project source code
•	mechanical_thrombectomy_outcome_predictor/ – Python package with the pipeline code
o	pipeline_registry.py – Registers all pipelines and specifies how they combine into the overall workflow (which pipelines run by default, etc.).
o	settings.py – Kedro project settings (e.g., naming the default pipelines package).
o	pipelines/ – Directory containing individual pipeline definitions, each as a subpackage:
o	data_cleaning/ – Data cleaning & validation pipeline
	nodes.py – Functions (“nodes”) that perform cleaning steps (detailed in section 4.2 below).
	pipeline.py – Combines cleaning nodes into a Kedro Pipeline definition.
o	eda_audit/ – Exploratory Data Analysis pipeline
	nodes.py – Functions for profiling data and generating EDA reports/figures (e.g., distribution plots, data quality summary).
	pipeline.py – Pipeline definition for EDA (runs after data_cleaning on the validated data).
o	data_splitting/ – Temporal splitting pipeline
	nodes.py – Function to add the train/test split label based on onset date.
	pipeline.py – Pipeline definition for splitting (runs after validation).
o	baseline_characteristics/ – Baseline comparison pipeline (Train vs Test)
	nodes.py – Function to generate baseline tables comparing train and test cohorts (calculating means and SMDs).
	pipeline.py – Pipeline definition (outputs baseline_table1 and baseline_smd).
o	modeling_mrs90/ – Model development pipeline for the mRS90 LightGBM model
	nodes.py – Core functions for dataset assembly, hyperparameter tuning, model training, evaluation, and SHAP analysis.
	pipeline.py – Pipeline definition listing each step of model development in order.
	config.py – Contains configuration for model features and monotone constraints (e.g., mapping of feature names to their expected monotonic direction).
	calibration_utils.py – Helper functions for calibration (e.g., computing calibration slope and intercept, applying logistic calibration to probabilities).
o	reporting_mrs90/ – Reporting pipeline (post-modeling analyses for TRIPOD)
	nodes.py – Functions to create cohort flow summaries, outcome availability, missingness tables, predictor dictionary, performance summaries, decision curve summaries, and to aggregate all results into the TRIPOD Excel.
	pipeline.py – Pipeline definition chaining the reporting nodes (runs after the modeling_mrs90 pipeline to produce final report artifacts).
o	Other files in src/ root:
o	__main__.py – Entry-point allowing the project to be run as a Python module.
o	(The project is structured so that running kedro run executes the __default__ pipeline which includes data_cleaning, data_splitting, baseline_characteristics, eda_audit, modeling_mrs90 pipelines in sequence. The reporting_mrs90 pipeline can be run separately to generate the documentation outputs after the model is trained.)
•	tests/ – Unit tests
o	test_run.py – A basic test to execute a full pipeline run on a subset of data (or dummy data) to ensure nothing breaks.
•	calibration_mrs90_analysis.py, cv_mrs90_evaluation.py, decision_curve_mrs90.py – Standalone analysis scripts (outside Kedro pipelines) used for additional evaluation. For example, cv_mrs90_evaluation.py may run extensive cross-validation to validate hyperparameters, and calibration_mrs90_analysis.py computes detailed calibration plots (these scripts read from data/06_models and write to data/08_reporting). In the current version, much of their functionality has been integrated into the pipelines (Optuna tuning for CV, and pipeline nodes for decision curves, etc.), though the scripts can still be used or adapted for further analysis.
•	app_mrs90.py – Streamlit application script for deploying the model (detailed in section 10).
Data Flow (Pipelines Overview) 
Conceptually, pipelines flows from raw data to final model and app as follows:
Raw registry (Excel: mt_patients_master.xlsx)
   │
   ▼
[Data Cleaning & Validation] **(data_cleaning pipeline)**
   - clean_mt_patients → mt_patients_clean.parquet  
   - exclude_invalid_cases → mt_patients_valid.parquet  
   - validate_and_fix → mt_patients_validated.parquet  
   │
   ├──► **EDA & Audit** (eda_audit pipeline) → data_profile.json, eda_audit_report.json   │
   ▼
Validated cohort (cleaned & quality-checked data)
   │
   └──► **Feature Engineering**  
        make_lightgbm_ready → **mt_lightgbm_ready.parquet** (LightGBM-ready dataset with missingness indicators)
   ▼
[Temporal Split] **(data_splitting pipeline)**  
   - create_temporal_split → mt_patients_split.parquet (adds `split=train/test` based on onset year)
   ▼
[LightGBM mRS90 Modeling] **(modeling_mrs90 pipeline)**  
   - build_mrs90_dataset → X_train, y_train, X_test, y_test, IDs, feature_list  
   - optuna_tune_mrs90_lgbm → **mrs90_lgbm_optuna_best_params**, CV metrics (10×20) 
   - train_mrs90_lgbm (with best params) → **mrs90_lgbm_model.pkl**, train_metrics  
   - evaluate_mrs90_lgbm → test_metrics, test_predictions  
   - mrs90_feature_importance → feature_importance.parquet  
   - compute_mrs90_shap → raw SHAP values, SHAP summaries (overall & age-stratified)  
   - plot_mrs90_shap, plot_mrs90_shap_age_groups → SHAP global plots, age-stratified SHAP plots  
   ▼
[Reporting & TRIPOD] **(reporting_mrs90 pipeline)**  
   - build_cohort_flow → mt_cohort_flow_table (patient counts at each stage)  
   - plot_cohort_flow_diagram → mt_cohort_flow_diagram.png (CONSORT-style flow chart)  
   - mrs90_outcome_availability → mrs90_outcome_availability.csv (who has 90d mRS, overall/train/test)  
   - mrs90_missingness_table → mrs90_missingness_table.parquet (missing data summary for all features)  
   - make_baseline_by_outcome → mrs90_baseline_by_outcome.csv (train-set baseline vars by outcome 0–2 vs 3–6)  
   - build_predictor_dictionary → mrs90_predictor_dictionary.xlsx (data dictionary of predictors)  
   - build_cv_summary_table → mrs90_cv_summary_table.csv (aggregate CV performance)  
   - build_performance_summary → mrs90_performance_summary.csv (test metrics summary, raw vs calibrated)  
   - build_threshold_performance → mrs90_threshold_performance.csv (test sensitivity/specificity at threshold[s])  
   - build_model_spec → mrs90_model_spec.json (model parameters, features, constraints)  
   - build_shap_top30 → mrs90_shap_top30.parquet (top 30 features by |SHAP|)  
   - build_dca_summary → mrs90_dca_summary.csv (net benefit summary)  
   - build_tripod_excel → mrs90_tripod_excel.xlsx (all key results compiled for reporting)  
   ▼  
[Streamlit App] **(app_mrs90.py)** – uses final model and reporting outputs to provide an interactive clinician interface (with password protection).  
In summary, raw data is cleaned and validated, then turned into a modeling dataset. A temporally separated training set is used to tune and train a LightGBM model. The model is evaluated thoroughly, and results (tables, metrics, figures) are produced for interpretation and reporting. Finally, a Streamlit app serves the model for individual patient predictions with explanation and performance info.
Data Ingestion and Preprocessing
Raw Data and Configuration
•	The raw thrombectomy registry is stored as an Excel file (mt_patients_master.xlsx) and is loaded via Kedro’s Pandas.ExcelDataset (conf/base/catalog.yml).
•	Static configuration maps are provided in YAML: column_map.yml (mapping original column names to normalized names) and value_maps.yml (semantic translations and token lists for binary coding). These ensure that Czech-language or inconsistent values are consistently mapped to English and unified categories.
Data Cleaning Pipeline
The data_cleaning Kedro pipeline (source: src/.../pipelines/data_cleaning/nodes.py) performs deterministic, TRIPOD-compliant preprocessing on the raw data. Key steps include:
•	Column Normalization and Renaming: All column headers are stripped of whitespace, lowercased, and spaces/dashes replaced by underscores for snake_case compliance[. A predefined column_map (loaded from conf/base/column_map.yml) is applied to rename columns with non-standard names, and a log of renamings is recorded.
•	Value Translation: Using the value_maps (from conf/base/value_maps.yml), many categorical fields (e.g. thrombolysis codes, hemisphere labels) are translated from source language/notation to standardized English categories.
•	Whitespace and Missing Marker Cleanup: Cells are trimmed of invisible characters (including non-breaking spaces) and empty-string or token markers ("none", "N/A", "?", etc.) are uniformly replaced with pd.NA. Clinical “No finding” categories are preserved as real categories.
•	TICI Score Normalization: The perfusion grade (tici) is normalized to an ordered categorical with levels {0, 1, 2a, 2b, 2c, 3}, accounting for various input formats (e.g. converting Roman numerals, "2b/3" → "2b").
•	Numeric Conversion: Critical numeric fields (age, NIHSS scores, time intervals, lab values, etc.) are coerced to numeric types (integer or float). Non-numeric entries become NA, and the number of coerced values per field is logged.
•	Binary Field Encoding: Many yes/no or presence/absence variables (e.g. ivt_given, sich, hypertension, etc.) are normalized to 0/1. A helper _to_binary function maps known true/false tokens to 1/0. After mapping, these columns are cast to integer (nullable Int8) for compact storage.
•	Composite Variables: Some fields are simplified (e.g. multiple heart condition categories are collapsed into a single binary “heart_condition” flag) to increase statistical power.
All transformations generate a detailed cleaning_log.json (data/08_reporting/data_cleaning/cleaning_log.json) containing counts of renamed columns, missing values introduced, out-of-range fixes, etc., to document the cleaning steps.
Validation and Range Checks
After initial cleaning and exclusion of invalid records, a validate_and_fix node performs range and plausibility checks:
•	Monotonic Time Checks: Ensures logical ordering of time intervals. For example, if onset-to-IVT time is larger than onset-to-puncture time for the same patient (an impossible scenario), the pipeline blanks the later (shorter) interval, and logs the violation.
•	Physiological Ranges: Fields like ASPECTS score (should be 0–10) and NIHSS at 7 days (capped at 0–42) are checked. Out-of-range values are set to NA and flagged. Similarly, mRS scores are forced to 0–6 and cast to nullable Int64.
•	Binary Casting: Binary columns are cast to Int8 (0/1) once validated.
The result is a validated cohort (mt_patients_validated.parquet) with patients meeting all data-quality rules. A log of range violations and fixes (data_quality_violations.json) is saved for auditing.
LightGBM-Ready Feature Table
A make_lightgbm_ready node constructs the modeling dataset (mt_lightgbm_ready.parquet). This simply adds derived features and missingness indicators, without imputing values:
•	Derived Features: E.g. an onset_year column is extracted from the onset date.
•	Missing Indicators: For clinically important variables (age, ASPECTS, BP, glucose, NIHSS, etc., both numeric and categorical), the pipeline creates companion *_missing columns (0/1 flags) indicating whether the value was missing. This allows LightGBM to use missingness itself as an informative feature (since LightGBM can handle NAs natively, but explicit missing flags can further aid interpretability).
At this stage, the data is “LightGBM-ready”: cleaned, validated, with missingness flags, but with no imputation of missing values (as per LightGBM’s internal handling and missing indicators).
Cohort Splitting
Before modeling, the cohort is split temporally into development (train) and internal validation (test) sets. The create_temporal_split node (pipeline data_splitting) labels each patient as “train” or “test” based on the date of stroke onset. Typically, earlier cases form the training set and the most recent cases form the test set (simulating prospective validation). The split labels are appended to the validated dataset (mt_patients_split.parquet).
Modeling Pipeline: Predicting mRS-90
The modeling_mrs90 Kedro pipeline handles the core model development. Its steps include dataset construction, hyperparameter tuning, model training, evaluation, and interpretability.
Dataset Construction (build_mrs90_dataset)
A node build_mrs90_dataset merges the LightGBM-ready data with split labels, filters to cases with observed 90-day mRS, and creates the binary outcome. Patients with missing 90-day outcome are dropped. The outcome is defined as good (1) if mRS 0–2, and poor (0) if mRS 3–6.
Interaction features are added deterministically to the dataset (even if not ultimately used) to maintain TRIPOD traceability. Specifically, if present, the code computes three interactions: - age_x_nihss = age * admission_NIHSS, - onset_to_puncture_min_x_nihss = onset_to_puncture_min * admission_NIHSS, - age_x_onset_to_puncture_min = age * onset_to_puncture_min.
These columns are always created if their parent columns exist. Optuna will later decide whether to use or discard each interaction via boolean flags; creating them early makes the data pipeline deterministic.
Non-predictor columns are dropped (identifiers, split labels, raw timestamps, and all post-treatment/outcome variables) so that the model is strictly pre-procedural. Categorical features remaining in the feature set are then converted to Pandas category dtype for LightGBM compatibility.
Finally, the data is split into training and test sets based on the earlier label:

X_train = X[split=="train"],  X_test = X[split=="test"],  
y_train = (mrs90_good for train),  y_test = (mrs90_good for test).
The function returns X_train, y_train, X_test, y_test, train_ids, test_ids, feature_list.
Monotonicity Constraints
Domain knowledge is encoded via monotonic constraints in LightGBM. A configuration function build_monotone_constraints(feature_list) returns a list of -1, 0, +1 for each feature. The signs are determined by clinical priors (in get_monotone_sign_map): e.g. higher age or NIHSS should decrease the probability of good outcome (monotone sign -1), while higher ASPECTS (pre-stroke CT score) should increase it (+1). By default the constraint on blood glucose is also negative (hyperglycemia lowers good outcome), unless relaxed in config. Interaction terms by default have no sign enforced (0). During training, these constraints are passed to LightGBM so the model’s learned function respects them (LightGBM’s monotone_constraints parameter).
Hyperparameter Tuning (Optuna)
The optuna_tune_mrs90_lgbm node performs automated hyperparameter search on the training set. Key points:
•	Objective Function: Rather than optimizing AUC alone, the code balances discrimination and calibration. It computes stratified K-fold CV on the training data for each trial, calculating mean ROC AUC and mean Brier score (mean squared error of predicted probabilities). The Optuna objective is max AUC – λ·Brier, with λ=0.75 (so a small penalty on poor calibration).
•	Tunable Parameters: The search includes standard LightGBM parameters (number of trees 200–600, learning rate 0.005–0.05, num_leaves 15–45, etc.) and regularization (reg_alpha, reg_lambda). It also toggles class_weight between None and "balanced".
•	Feature Selection (Interaction Flags): Crucially, Optuna also treats whether to use each interaction term as hyperparameters (boolean flags use_age_x_nihss, etc.). This effectively makes feature selection part of tuning, automatically dropping interactions that do not improve the objective. The code constructs the “active feature” set based on these flags in each trial.
•	Results: After n_trials (default 50), Optuna returns the best hyperparameters (including chosen flags) and cross-validation metrics summary (auc, brier, log_loss means and std across folds). These outputs are saved for reporting: mrs90_lgbm_optuna_best_params.json and a nested dict cv_metrics_raw (later used by the app).
Final Model Training
Using the best parameters from Optuna, train_mrs90_lgbm_with_optuna fits a final LightGBM model on the entire training set. It extracts the interaction flags from best_params, removes those interaction columns if flagged off, and rebuilds the monotonic constraints list for the actual active features. All training rows (X_train_active) are then used to fit the LGBMClassifier.
The function returns the trained model and a dict of in-sample performance metrics (ROC AUC, accuracy, F1, log loss, average precision) on the training data. These metrics appear in mrs90_train_metrics.json.
Model Evaluation
The evaluate_mrs90_lgbm node assesses performance on the held-out test set. It first aligns the test features to exactly the feature set the model was trained on (using model.feature_name_). Then it obtains predicted probabilities and class labels on the test samples, computes standard metrics (ROC AUC, accuracy, F1, log loss), and packages them. It outputs both a JSON of metrics (mrs90_test_metrics.json) and a table of individual predictions (mrs90_test_predictions.parquet, including patient ID, true outcome, probability, predicted label).
These results allow evaluation of discrimination (ROC AUC) and accuracy of predictions.
Calibration and Decision Analysis
Logistic (Platt) Calibration
To ensure calibrated probabilities, the project applies Platt scaling via logistic regression on the model’s outputs. The function apply_logistic_calibration(p_raw, a, b) (in calibration_utils) implements: $$\text{logit}(p_\text{cal}) = a + b\,\text{logit}(p_\text{raw}),$$ where a and b are calibration intercept/slope estimated on the training set. A standalone script calibration_mrs90_analysis.py performs:
1.	Cross-validated fitting: On the training data, it does K-fold cross-validation (e.g. 5-fold) using the final LightGBM model, obtaining out-of-fold probabilities for each patient. A logistic regression is fit to true y vs logit of these probabilities to estimate calibration intercept a and slope b.
2.	Application to Predictions: The script then loads the tuned model and computes raw probabilities on both train and test sets. It applies the calibration formula (with clipped probabilities to avoid 0/1).
3.	Metrics: It computes and records pre- and post-calibration metrics (ROC AUC, Brier score, log loss) for train and test. These metrics are saved as JSON (mrs90_calibration_metrics_raw.json) for the Streamlit app and reporting.
By default, the calibrated probabilities are used in the final application (the app and decision curve). If a≈0 and b≈1, the app notes that no recalibration was needed.
Decision Curve Analysis (DCA)
A standalone script decision_curve_mrs90.py computes decision curves on the test set. For a range of risk thresholds (e.g. 5% to 75%), it calculates net benefit: $$\text{NB}_{model} = \frac{TP}{N} - \frac{FP}{N}\frac{t}{1-t},$$ where “positive” is defined as a good outcome (mRS 0–2). It also computes net benefit for “treat all” and “treat none” strategies. Using the calibrated probabilities (applying the learned a,b), it saves a CSV of net benefit values at each threshold (mrs90_decision_curve_calibrated.csv). A Matplotlib plot (mrs90_decision_curve_calibrated.png) is also generated.
In the Streamlit app, a helper (dca_useful_range) finds the contiguous threshold range where the model’s net benefit exceeds both treat-all and treat-none (and is positive). This “useful range” is displayed to clinicians as guidance. The app also plots the net benefit curves interactively.
Interpretability and Feature Analysis
Feature Importance
Global feature importance is extracted from the trained model via its feature_importances_ array. A utility node mrs90_feature_importance aligns these importances with feature names and sorts them descending. This table (mrs90_feature_importance.parquet) is used in reports and for the SHAP analysis.
SHAP Explainability
Model-agnostic explanations are computed using SHAP (TreeExplainer for LightGBM). The pipeline has a compute_mrs90_shap node that produces:
•	Train and Test SHAP values: It subsamples the training set (up to 300 rows for speed) and uses the full test set. It calls shap.TreeExplainer(model).shap_values(...) on both, retrieving SHAP values for class 1 (good outcome). It also obtains the model’s “expected value” (baseline log-odds).
•	SHAP Summaries: It computes mean(|SHAP|) per feature on train and test, producing summary tables (mrs90_shap_summary_train.parquet, ..._test.parquet) sorted by importance.
•	Age-Stratified SHAP: Using function mrs90_shap_importance_by_age_groups, it further splits the test set by age thresholds (<50 vs ≥50, <75/≥75, etc.) and computes mean(|SHAP|) per feature within each age subgroup. The combined long-format table (mrs90_shap_summary_test_by_age.parquet) shows how feature importance changes with age.
Plotting nodes create global SHAP figures (bar plot of mean|SHAP|, beeswarm summary plot, and a dependence plot for the top feature). Additionally, four age-group bar charts are generated by plot_mrs90_shap_age_groups, comparing the top-N features for younger vs older patients at each age threshold.
These SHAP results are saved (raw values and plots) to aid interpretation: researchers can see which variables most influence predictions.
Reporting and Outputs (TRIPOD Compliance)
Beyond model training, MTOP generates extensive documentation outputs (in data/08_reporting) in line with TRIPOD guidelines:
•	Cohort Flow Diagram: A node build_cohort_flow tallies counts at each cleaning stage (raw, cleaned, valid, split, mRS-observed) and creates a simple flowchart figure. This is saved as mt_cohort_flow_table.csv and mt_cohort_flow_diagram.png.
•	Outcome Availability: Tables summarizing how many patients have 90-day mRS recorded, overall and by split.
•	Baseline Characteristics by Outcome: The make_baseline_by_outcome node computes group summaries in the training set. For each predictor, it outputs mean±SD (with sample size) among patients with good vs poor outcome, along with the standardized mean difference (SMD). Categorical variables have counts by category. This yields baseline_table1.csv (means/counts) and baseline_smd.csv.
•	Missingness Table: The pipeline compiles a table of missingness for each feature in the modeling set (overall, and by outcome group).
•	Predictor Dictionary: The build_predictor_dictionary node creates a TRIPOD-style dictionary (mrs90_predictor_dictionary.xlsx) listing each feature, its type, monotonicity sign, whether it has a _missing flag, and whether it was used in the final model.
•	Cross-Validation Summary: From Optuna’s CV metrics, build_cv_summary_table produces a CSV of mean±SD for AUC, Brier, log loss across folds.
•	Performance Summary: mrs90_performance_summary.csv collates final train/test AUCs, accuracies, etc. A threshold analysis yields mrs90_threshold_performance.csv, enumerating sensitivity, specificity, etc. at various probability cutoffs.
•	Model Specification: Key model info (hyperparameters, number of features, interaction inclusion, monotone settings) is saved as JSON mrs90_model_spec.json.
•	Calibration Metrics: The calibration script outputs raw and calibrated metrics (mrs90_calibration_metrics_raw.json). The intercept and slope are saved separately (mrs90_logistic_calibration_params.json).
•	SHAP Outputs: Shap summary tables and plots as above (mrs90_shap_expected_value.json, ..._summary*.parquet, and PNG figures).
•	Decision Curve Results: The DCA CSV and derived summary (mrs90_dca_summary.csv) are saved for report.
•	TRIPOD Excel: Finally, many of these tables and figures are assembled into a single Excel workbook mrs90_tripod_excel.xlsx for convenient review.
Each of these steps is implemented in the reporting_mrs90 pipeline (with dedicated nodes in nodes.py and pipelines assembled in pipeline_registry.py). For example, cohort flow and outcome availability are covered in “A”, baseline in “B”, CV in “C” sections of the script. This ensures end-to-end reproducibility.
Web Application (Streamlit)
The final predictive tool is deployed as a Streamlit web app (app_mrs90.py). This serves the model to end-users (e.g. clinicians) with an interactive interface for single-patient prediction.
App Architecture
•	The app loads the final LightGBM model (pickle), training features (X_train), raw LightGBM-ready data (mt_lightgbm_ready.parquet) and pre-computed metrics from the reporting step (calibration metrics JSON, CV metrics JSON, decision curve CSV). These are cached to speed up usage.
•	An optional password gate (via st.secrets) can restrict access.
•	In the sidebar, the app displays calibration status (if trivial, or showing a,b used), and a brief “About” description noting that this is a research-only prognostic model (not clinical advice). A checkbox toggles “debug” mode to reveal internal vectors.
User Interface
Input fields are organized into sections:
•	Demographics: Age (slider), Sex (select box).
•	Baseline Clinical: Vascular risk factors (hypertension, diabetes, etc. as checkboxes), admission NIHSS (slider), pre-stroke mRS, vitals (BP, glucose), laboratory values, etc.
•	Imaging: ASPECTS score, occlusion site (dropdown), affected hemisphere.
•	Treatment/Timing: IVT given (yes/no), whether IVT was at a different hospital or transferred, time to IVT, time to groin puncture (sliders), stroke etiology, etc.
For categorical inputs, the app uses mappings derived from the raw training data: it infers which raw labels correspond to each encoded numeric value by aligning X_train with the original mt_lightgbm_ready values (via build_category_mapping). For example, “Left hemisphere” and “Right hemisphere” options map to 0/1 internally. Users can also explicitly select “Missing” for fields like Sex or others, which is then passed through as np.nan.
Any field not set by the user is filled with the median (for numeric) or mode (for categorical) from the training data, as implemented in build_patient_row. After constructing the feature vector, the app coerces each column to the training dtype (crucial for categorical columns to match LightGBM’s categories). It then enforces any *_missing indicator columns according to whether the base value is NA. This yields a 1-row DataFrame matching the model’s input specification.
Prediction and Display
When the user clicks “Compute probability of good outcome (mRS 0–2)”, the app:
1.	Align features: Reorders columns to match model.feature_name_ (dropping any unused interactions).
2.	Predict: Calls model.predict_proba(patient_row) to get the raw probability of good outcome.
3.	Calibrate: Applies the logistic recalibration with stored parameters (a,b). Outputs proba_good = calibrated probability of mRS 0–2, and proba_bad = 1 - proba_good.
4.	Display: Shows two metric boxes: % chance of good outcome and % chance of poor outcome (with one decimal).
If debug mode is on, the app additionally reveals: - The encoded feature vector (as a 1×N table).
- The raw vs calibrated probability and calibration parameters.
- A SHAP summary chart: it attempts to compute SHAP values for this patient using the LightGBM TreeExplainer. If successful, it lists the top feature contributions (positive SHAP means pushing toward good outcome) and a horizontal bar chart of the top ~20 SHAP values. If SHAP is not installed or fails, it shows a warning.
Clinical Performance Dashboard
Below the prediction, the app provides contextual performance information for clinicians (pulling from saved metrics). It shows:
•	ROC AUC (test set): Displayed with two decimal places and explanatory text (e.g. “AUC ≈0.85 means model rank-orders well”).
•	Brier Score (calibrated probabilities on test): Key calibration metric. The app compares the uncalibrated vs calibrated Brier and log loss, slope, intercept, emphasizing that the calibrated values are used in the app.
•	Decision Curve Summary: If available, it shows the “useful threshold range” (e.g. 20–40%) where net benefit exceeds treat-all/none. It also plots the net benefit curves (model vs all vs none) as an interactive line chart.
•	Cross-Validation Summary: In an expander, it reports the mean±SD of AUC, Brier, log loss from the Optuna tuning CV on development data.
•	Interpretation Guide: Another expander provides plain-language notes on what AUC and calibration mean for patient care.
This front-end integrates clinical concepts seamlessly: for example, IVT timing, NIHSS, ASPECTS, age, etc. are explicitly input fields. The predicted output is the probability of good outcome (mRS 0–2), with the complementary chance of poor outcome shown side by side. The interface also notes that all other features are set to median/mode, and cautions that this is “research use only” (not for real-time clinical decisions).
Technologies and Libraries
Throughout the project, a range of scientific computing tools are used:
•	Python Libraries: Data processing with Pandas, NumPy, and Matplotlib (for plots) are ubiquitous. Scikit-learn provides metrics (ROC AUC, Brier, log loss, accuracy, F1), stratified K-fold CV, and the IterativeImputer (though in this project missing data is not imputed for final model).
•	LightGBM: The predictive model is implemented via lightgbm.LGBMClassifier, taking advantage of its categorical feature support and monotonic constraint options.
•	Optuna: Automated tuning of hyperparameters and feature flags is done with Optuna. The study is seeded for reproducibility.
•	SHAP: The SHAP library is used for explainability. Its TreeExplainer computes contributions of each feature to the prediction.
•	Streamlit: The web app uses Streamlit for the GUI. It dynamically builds widgets (sliders, select boxes) and displays results.
•	Kedro: The project is organized as a Kedro pipeline, with configurations (conf/), a catalog of datasets (catalog.yml), and modular nodes/pipelines. This enforces structure (each node’s inputs/outputs are defined) and makes data lineage clear. The pipeline_registry.py ties together pipelines (cleaning, EDA, splitting, modeling, reporting).
•	Storage Formats: Parquet files are used for intermediate tabular data for efficiency. JSON and CSV are used for metrics and reports. An ExcelDataset is also used to produce the final TRIPOD report (mrs90_tripod_excel.xlsx).
All code and configuration files (including logging setup conf/logging.yml) are version-controlled. Unit tests (e.g. tests/test_run.py) ensure the pipeline executes end-to-end without errors on sample data.
Conclusion
In summary, the MTOP project implements a full machine-learning workflow for predicting stroke outcomes after thrombectomy. It emphasizes clinical validity by incorporating domain knowledge (monotone constraints on stroke severity and age), thorough data cleaning (with logs for audit), and comprehensive evaluation (discrimination, calibration, decision analysis). The modular code (spanning data cleaning to a user-friendly app) is fully described by the repository. Clinicians can input patient features into the web interface and obtain a calibrated probability of good outcome, along with model explanations and performance context, all backed by rigorous pipeline processes documented in code.

