from __future__ import annotations

from kedro.pipeline import Pipeline, node

from mechanical_thrombectomy_outcome_predictor.pipelines.reporting_mrs90.nodes import (
    build_cohort_flow,
    plot_cohort_flow_diagram,
    mrs90_outcome_availability,
    mrs90_missingness_table,
    make_baseline_by_outcome,
    build_predictor_dictionary,
    build_cv_summary_table,
    build_performance_summary,
    build_threshold_performance,
    build_model_spec,
    build_shap_top30,
    build_dca_summary,
    build_tripod_excel,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # 1) Cohort flow table
            node(
                func=build_cohort_flow,
                inputs=[
                    "mt_patients_master",
                    "mt_patients_clean",
                    "mt_patients_valid",
                    "mt_patients_validated",
                    "mt_patients_split",
                ],
                outputs="mt_cohort_flow_table",
                name="build_cohort_flow_node",
            ),

            # 2) Cohort flow diagram (figure)
            node(
                func=plot_cohort_flow_diagram,
                inputs="mt_cohort_flow_table",
                outputs="mt_cohort_flow_diagram",
                name="plot_cohort_flow_diagram_node",
            ),

            # 3) Outcome availability (mRS90)
            node(
                func=mrs90_outcome_availability,
                inputs=["mt_patients_validated", "mt_patients_split"],
                outputs="mrs90_outcome_availability",
                name="mrs90_outcome_availability_node",
            ),

            # 4) Missingness table for predictors
            node(
                func=mrs90_missingness_table,
                inputs=["mt_lightgbm_ready", "mt_patients_split"],
                outputs="mrs90_missingness_table",
                name="mrs90_missingness_table_node",
            ),

            # 5) Baseline by outcome (TRAIN only)
            node(
                func=make_baseline_by_outcome,
                inputs=["mrs90_X_train", "mrs90_y_train"],
                outputs="mrs90_baseline_by_outcome",
                name="baseline_by_outcome_node",
            ),

            # 6) Predictor dictionary (TRIPOD)
            node(
                func=build_predictor_dictionary,
                inputs=[
                    "mt_lightgbm_ready",
                    "mrs90_feature_list",
                    "mrs90_lgbm_model",
                ],
                outputs="mrs90_predictor_dictionary",
                name="build_predictor_dictionary_node",
            ),

            # 7) CV metrics summary (Optuna)
            node(
                func=build_cv_summary_table,
                inputs="mrs90_lgbm_cv_metrics_raw_10x20",
                outputs="mrs90_cv_summary_table",
                name="build_cv_summary_table_node",
            ),

            # 8) Performance summary (raw + calibrated)
            node(
                func=build_performance_summary,
                inputs="mrs90_calibration_metrics_raw",
                outputs="mrs90_performance_summary",
                name="build_performance_summary_node",
            ),

            # 9) Threshold-based performance on test
            node(
                func=build_threshold_performance,
                inputs=["mrs90_test_predictions", "mrs90_logistic_calibration_params"],
                outputs="mrs90_threshold_performance",
                name="build_threshold_performance_node",
            ),

            # 10) Full model spec for implementation
            node(
                func=build_model_spec,
                inputs=[
                    "mrs90_lgbm_model",
                    "mrs90_lgbm_optuna_best_params",
                    "mrs90_logistic_calibration_params",
                    "mrs90_feature_list",
                ],
                outputs="mrs90_model_spec",
                name="build_model_spec_node",
            ),

            # 11) SHAP top 30 table
            node(
                func=build_shap_top30,
                inputs="mrs90_shap_summary_test",
                outputs="mrs90_shap_top30",
                name="build_shap_top30_node",
            ),

            # 12) DCA summary at key thresholds
            node(
                func=build_dca_summary,
                inputs="mrs90_decision_curve_calibrated",
                outputs="mrs90_dca_summary",
                name="build_dca_summary_node",
            ),

            # 13) TRIPOD Excel – combine everything into one workbook-like dict
            node(
                func=build_tripod_excel,
                inputs=[
                    "mt_cohort_flow_table",
                    "mrs90_outcome_availability",
                    "mrs90_missingness_table",
                    "mrs90_predictor_dictionary",
                    "mrs90_model_spec",
                    "mrs90_cv_summary_table",
                    "mrs90_performance_summary",
                    "mrs90_threshold_performance",
                    "mrs90_shap_top30",
                    "mrs90_dca_summary",
                ],
                outputs="mrs90_tripod_excel",
                name="build_tripod_excel_node",
            ),
        ]
    )
