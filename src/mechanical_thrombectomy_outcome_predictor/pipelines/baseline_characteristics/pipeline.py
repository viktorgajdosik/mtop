from kedro.pipeline import Pipeline, node
from .nodes import make_baseline_tables


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=make_baseline_tables,
                inputs="mt_patients_split",
                outputs=["baseline_table1", "baseline_smd"],
                name="make_baseline_tables_node",
            ),
        ]
    )
