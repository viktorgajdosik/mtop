from kedro.pipeline import Pipeline, node

from .nodes import run_mt_stroke_regression_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_mt_stroke_regression_pipeline,
                inputs=[
                    "mt_patients_regression_ready",
                    "mt_patients_split",
                ],
                outputs=[
                    "mt_coef_pre_full",
                    "mt_coef_post_full",
                    "mt_model_fit_stats",
                    "mt_roc_values",
                    "mt_calibration_values",
                    "mt_dca_values",
                    "mt_shap_like_pre_full",
                    "mt_shap_like_post_full",
                    "mt_cv_results",
                    "mt_cv_summary",
                    "mt_vif_pre_full",
                    "mt_vif_post_full",
                    "mt_nri_idi_pre_vs_post",
                ],
                name="run_mt_stroke_regression_pipeline_node",
            ),
        ]
    )
