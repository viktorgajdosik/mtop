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


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_mrs90_dataset,
                inputs=["mt_lightgbm_ready", "mt_patients_split"],
                outputs=[
                    "mrs90_X_train",
                    "mrs90_y_train",
                    "mrs90_X_test",
                    "mrs90_y_test",
                    "mrs90_train_ids",
                    "mrs90_test_ids",
                    "mrs90_feature_list",
                ],
                name="build_mrs90_dataset_node",
            ),
            node(
                func=train_mrs90_lgbm,
                inputs=["mrs90_X_train", "mrs90_y_train"],
                outputs=["mrs90_lgbm_model", "mrs90_train_metrics"],
                name="train_mrs90_lgbm_node",
            ),
            node(
                func=evaluate_mrs90_lgbm,
                inputs=[
                    "mrs90_lgbm_model",
                    "mrs90_X_test",
                    "mrs90_y_test",
                    "mrs90_test_ids",
                ],
                outputs=["mrs90_test_metrics", "mrs90_test_predictions"],
                name="evaluate_mrs90_lgbm_node",
            ),
            node(
                func=mrs90_feature_importance,
                inputs=["mrs90_lgbm_model", "mrs90_feature_list"],
                outputs="mrs90_feature_importance",
                name="mrs90_feature_importance_node",
            ),
            node(
                func=compute_mrs90_shap,
                inputs=[
                    "mrs90_lgbm_model",
                    "mrs90_X_train",
                    "mrs90_X_test",
                    "mrs90_feature_list",
                ],
                outputs={
                    "mrs90_shap_train_values": "mrs90_shap_train_values",
                    "mrs90_shap_test_values": "mrs90_shap_test_values",
                    "mrs90_shap_expected_value": "mrs90_shap_expected_value",
                    "mrs90_shap_summary_train": "mrs90_shap_summary_train",
                    "mrs90_shap_summary_test": "mrs90_shap_summary_test",
                    "mrs90_shap_summary_test_by_age": "mrs90_shap_summary_test_by_age",
                },
                name="mrs90_shap_values_node",
            ),
            node(
                func=plot_mrs90_shap,
                inputs=[
                    "mrs90_shap_test_values",
                    "mrs90_X_test",
                    "mrs90_feature_list",
               ],
                outputs=[
                    "mrs90_shap_barplot",
                    "mrs90_shap_beeswarm",
                    "mrs90_shap_dependence",
               ],
                name="mrs90_shap_plots_node",
            ),
            # Age-stratified SHAP plots (<T vs ≥T) – 4 PNGs
            node(
                func=plot_mrs90_shap_age_groups,
                inputs=[
                    "mrs90_shap_test_values",
                    "mrs90_X_test",
                    "mrs90_feature_list",
                ],
                outputs=[
                    "mrs90_shap_age_lt50_ge50",
                    "mrs90_shap_age_lt75_ge75",
                    "mrs90_shap_age_lt80_ge80",
                    "mrs90_shap_age_lt85_ge85",
                ],
                name="plot_mrs90_shap_age_groups",
            ),
        ]
    )
