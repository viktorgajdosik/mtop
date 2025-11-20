from kedro.pipeline import Pipeline, node
from .nodes import audit_dataset
# Reuse helpers for plots / merge
from mechanical_thrombectomy_outcome_predictor.pipelines.data_cleaning.nodes import (
    plot_distributions,
    merge_reports,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # EDA is run on VALIDATED data (already excluded)
        node(
            func=audit_dataset,
            inputs="mt_patients_validated",
            outputs="eda_audit_report",
            name="audit_dataset_node",
        ),
        node(
            func=plot_distributions,
            inputs="mt_patients_validated",
            outputs="eda_topn_plots",
            name="plot_distributions_node",
        ),
        node(
            func=merge_reports,
            inputs=["eda_audit_report", "data_quality_violations"],
            outputs="data_profile",
            name="merge_reports_node",
        ),
    ])
