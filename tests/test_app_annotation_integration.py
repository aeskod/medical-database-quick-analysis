import pandas as pd
from streamlit.testing.v1 import AppTest

from src.column_annotations import (
    ColumnAnnotation,
    USE_BASELINE,
    USE_CHARTS,
    USE_GROUP,
    USE_IGNORE,
    apply_survival_roles,
    build_default_annotations,
)
from src.profiling import profile_dataframe
from src.survival_mapping import SurvivalConfig, create_survival_ready_dataframe


def annotation_component_harness(df, profile):
    from app import _render_column_annotations

    _render_column_annotations(df, profile)


def _integration_dataset():
    return pd.DataFrame(
        {
            "time": list(range(10, 130, 10)),
            "status": [1, 0] * 6,
            "age": list(range(50, 62)),
            "sex": ["F", "M"] * 6,
            "treatment": ["A", "B", "C"] * 4,
            "notes": [f"Clinical note {index}" for index in range(12)],
        }
    )


def _survival_config():
    return SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        group_col="sex",
        time_unit="days",
    )


def test_annotation_component_renders_editor_actions_and_use_summary():
    df = _integration_dataset()
    profile = profile_dataframe(df)
    config = _survival_config()
    annotations = build_default_annotations(df, profile, config)
    app_test = AppTest.from_function(
        annotation_component_harness,
        args=(df, profile),
    )
    app_test.session_state["survival_config"] = config
    app_test.session_state["column_annotations"] = annotations

    app_test.run(timeout=10)

    assert not app_test.exception
    assert [item.value for item in app_test.subheader] == ["Column annotations"]
    assert {button.label for button in app_test.button} == {
        "Save annotations",
        "Reset to suggested annotations",
    }
    metrics = {metric.label: metric.value for metric in app_test.metric}
    assert list(metrics) == ["Filters", "Groups", "Baseline", "Cox", "Charts", "Ignored"]
    assert metrics["Ignored"] == "1"


def test_all_analysis_tabs_consume_the_same_saved_annotations():
    df = _integration_dataset()
    profile = profile_dataframe(df)
    config = _survival_config()
    annotations = {
        column: ColumnAnnotation(column, "Ignore column", frozenset({USE_IGNORE}))
        for column in df.columns
    }
    annotations.update(
        {
            "age": ColumnAnnotation("age", "Age", frozenset({USE_BASELINE, USE_CHARTS})),
            "sex": ColumnAnnotation(
                "sex",
                "Sex / gender",
                frozenset({USE_BASELINE, USE_GROUP, USE_CHARTS}),
            ),
            "treatment": ColumnAnnotation(
                "treatment",
                "Treatment / exposure group",
                frozenset({USE_GROUP}),
            ),
        }
    )
    annotations = apply_survival_roles(annotations, config)

    app_test = AppTest.from_file("app.py")
    app_test.session_state["uploaded_df"] = df
    app_test.session_state["profile_df"] = profile
    app_test.session_state["survival_config"] = config
    app_test.session_state["survival_ready_df"] = create_survival_ready_dataframe(df, config)
    app_test.session_state["column_annotations"] = annotations

    app_test.run(timeout=30)

    assert not app_test.exception
    group_by = next(item for item in app_test.selectbox if item.label == "Group by")
    survival_group = next(
        item
        for item in app_test.selectbox
        if item.label == "Group / stratification variable"
    )
    chart_x = next(item for item in app_test.selectbox if item.label == "X variable")
    continuous = next(
        item
        for item in app_test.multiselect
        if item.label == "Continuous variables"
    )
    categorical = next(
        item
        for item in app_test.multiselect
        if item.label == "Categorical variables"
    )

    assert group_by.options == ["No grouping", "sex", "treatment"]
    assert survival_group.options == ["No grouping", "sex", "treatment"]
    assert chart_x.options == ["None", "time", "status", "age", "sex"]
    assert continuous.value == ["age"]
    assert categorical.value == ["sex"]
