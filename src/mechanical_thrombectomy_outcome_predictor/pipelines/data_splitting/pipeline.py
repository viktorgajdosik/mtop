from kedro.pipeline import Pipeline, node
from .nodes import create_temporal_split


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_temporal_split,
                inputs="mt_patients_validated",
                outputs="mt_patients_split",
                name="create_temporal_split_node",
            ),
        ]
    )
