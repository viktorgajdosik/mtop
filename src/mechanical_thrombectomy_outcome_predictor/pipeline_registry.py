from kedro.pipeline import Pipeline
from mechanical_thrombectomy_outcome_predictor.pipelines import (
    data_cleaning,
    eda_audit,
    data_splitting,
    baseline_characteristics,
    modeling_mrs90,
    mt_regression,
)


def register_pipelines() -> dict[str, Pipeline]:
    dc = data_cleaning.create_pipeline()
    eda = eda_audit.create_pipeline()
    split = data_splitting.create_pipeline()
    base = baseline_characteristics.create_pipeline()
    mrs = modeling_mrs90.create_pipeline()
    regression = mt_regression.create_pipeline()

    return {
        "data_cleaning": dc,
        "eda_audit": eda,
        "data_splitting": split,
        "baseline_characteristics": base,
        "modeling_mrs90": mrs,
        "mt_regression": regression,
        "__default__": dc + split + base + eda + mrs,
    }
