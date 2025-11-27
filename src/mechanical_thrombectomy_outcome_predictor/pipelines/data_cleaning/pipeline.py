from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_mt_patients,
    exclude_invalid_cases,
    validate_and_fix,
    make_lightgbm_ready,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_mt_patients,
                inputs=dict(
                    mt_patients_raw="mt_patients_master",
                    column_map="column_map",
                    value_maps="value_maps",
                ),
                outputs=["mt_patients_clean", "cleaning_log"],
                name="clean_mt_patients_node",
            ),
            node(
                func=exclude_invalid_cases,
                inputs="mt_patients_clean",
                outputs=["mt_patients_valid", "exclusion_log"],
                name="exclude_invalid_cases_node",
            ),
            node(
                func=validate_and_fix,
                inputs="mt_patients_valid",
                outputs=["mt_patients_validated", "data_quality_violations"],
                name="validate_and_fix_node",
            ),
            node(
                func=make_lightgbm_ready,
                inputs="mt_patients_validated",
                outputs="mt_lightgbm_ready",
                name="make_lightgbm_ready_node",
            ),
            

        ]
    )
