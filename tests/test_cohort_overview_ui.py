import pandas as pd
from streamlit.testing.v1 import AppTest

from src.column_annotations import ColumnAnnotation, USE_BASELINE, USE_GROUP, USE_IGNORE
from src.profiling import profile_dataframe
from src.survival_mapping import SurvivalConfig, create_survival_ready_dataframe


def cohort_overview_harness():
    from app import _render_cohort_overview_tab

    _render_cohort_overview_tab()


def test_cohort_overview_renders_patient_aware_cards_characteristics_and_table_one():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P3"],
            "time": [10, 20, 30, 40],
            "status": [1, 0, 1, 0],
            "age": [50, 50, 60, 70],
            "sex": ["F", "F", "M", "F"],
            "diagnosis": ["A", "A", "B", "B"],
            "treatment": ["Drug", "Drug", "Control", "Drug"],
            "outcome": ["Response", "Response", "Stable", "Progression"],
        }
    )
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        id_col="patient_id",
        group_col="treatment",
        time_unit="months",
    )
    annotations = {
        "patient_id": ColumnAnnotation("patient_id", "Patient ID", frozenset({USE_IGNORE})),
        "time": ColumnAnnotation("time", "Follow-up time / survival time"),
        "status": ColumnAnnotation("status", "Event status"),
        "age": ColumnAnnotation("age", "Age", frozenset({USE_BASELINE})),
        "sex": ColumnAnnotation("sex", "Sex / gender", frozenset({USE_BASELINE})),
        "diagnosis": ColumnAnnotation("diagnosis", "Diagnosis", frozenset({USE_BASELINE})),
        "treatment": ColumnAnnotation(
            "treatment",
            "Treatment / exposure group",
            frozenset({USE_BASELINE, USE_GROUP}),
        ),
        "outcome": ColumnAnnotation(
            "outcome",
            "Outcome other than survival",
            frozenset({USE_BASELINE}),
        ),
    }
    app_test = AppTest.from_function(cohort_overview_harness)
    app_test.session_state["uploaded_df"] = df
    app_test.session_state["profile_df"] = profile_dataframe(df)
    app_test.session_state["survival_config"] = config
    app_test.session_state["survival_ready_df"] = create_survival_ready_dataframe(df, config)
    app_test.session_state["column_annotations"] = annotations

    app_test.run(timeout=20)

    assert not app_test.exception
    metrics = {metric.label: metric.value for metric in app_test.metric}
    assert metrics["Total patients"] == "3"
    assert metrics["Rows"] == "4"
    assert metrics["Median age"] == "60"
    assert "Key cohort characteristics" in [item.value for item in app_test.subheader]
    assert {item.label for item in app_test.expander}.issuperset(
        {
            "Age: age",
            "Sex / gender: sex",
            "Diagnosis: diagnosis",
            "Treatment: treatment",
            "Outcome: outcome",
        }
    )
    baseline_table = next(
        dataframe.value
        for dataframe in app_test.dataframe
        if "Variable" in dataframe.value.columns
    )
    assert baseline_table["Variable"].head(3).tolist() == [
        "n",
        "Events, n (%)",
        "Observed duration, median [IQR]",
    ]
    assert baseline_table.columns.tolist() == ["Variable", "Overall", "Control", "Drug"]
    assert baseline_table.loc[baseline_table["Variable"] == "n", "Drug"].iloc[0] == "2"
