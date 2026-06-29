from datetime import date

import pandas as pd
from streamlit.testing.v1 import AppTest

from app import _apply_survival_filters, _survival_dataframe_for_group
from src.column_annotations import ColumnAnnotation, USE_FILTER, USE_GROUP, USE_IGNORE
from src.profiling import profile_dataframe
from src.survival_mapping import SurvivalConfig, create_survival_ready_dataframe


def _dataset():
    return pd.DataFrame(
        {
            "time": [100, 200, 400, 500, 700, 800, 1100, 1200, 1500, 1600, 1900, 2000],
            "status": [1, 0] * 6,
            "age": list(range(50, 62)),
            "sex": ["F", "M"] * 6,
            "treatment": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "visit": pd.date_range("2020-01-01", periods=12, freq="YS").astype(str),
        }
    )


def _config():
    return SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        group_col="sex",
        time_unit="days",
    )


def _annotations(df):
    annotations = {
        column: ColumnAnnotation(column, "Ignore column", frozenset({USE_IGNORE}))
        for column in df
    }
    annotations.update(
        {
            "age": ColumnAnnotation("age", "Age", frozenset({USE_FILTER})),
            "sex": ColumnAnnotation(
                "sex",
                "Sex / gender",
                frozenset({USE_FILTER, USE_GROUP}),
            ),
            "treatment": ColumnAnnotation(
                "treatment",
                "Treatment / exposure group",
                frozenset({USE_FILTER, USE_GROUP}),
            ),
        }
    )
    return annotations


def test_apply_survival_filters_combines_types_without_mutating_input():
    df = _dataset()
    original = df.copy(deep=True)

    result = _apply_survival_filters(
        df,
        {
            "age": ("numeric", (52, 59)),
            "treatment": ("categorical", ["B"]),
            "visit": ("date", (date(2024, 1, 1), date(2027, 1, 1))),
        },
    )

    assert result.index.tolist() == [4, 5, 6, 7]
    pd.testing.assert_frame_equal(df, original)


def test_group_selector_rebuilds_the_filtered_survival_cohort():
    df = _dataset()
    config = _config()
    stored = create_survival_ready_dataframe(df, config)
    filtered = df[df["treatment"] == "A"]

    result = _survival_dataframe_for_group(filtered, config, stored, "treatment")

    assert len(result) == 4
    assert result["_group"].tolist() == ["A"] * 4


def test_survival_controls_update_counts_and_expose_requested_cards():
    df = _dataset()
    config = _config()
    app_test = AppTest.from_file("app.py")
    app_test.session_state["uploaded_df"] = df
    app_test.session_state["profile_df"] = profile_dataframe(df)
    app_test.session_state["survival_config"] = config
    app_test.session_state["survival_ready_df"] = create_survival_ready_dataframe(df, config)
    app_test.session_state["column_annotations"] = _annotations(df)

    app_test.run(timeout=30)

    assert not app_test.exception
    assert next(item for item in app_test.checkbox if item.label == "Show censor marks").value is True
    group = next(
        item
        for item in app_test.selectbox
        if item.label == "Group / stratification variable"
    )
    assert group.options == ["No grouping", "sex", "treatment"]
    metric_labels = {metric.label for metric in app_test.metric}
    assert {
        "Current cohort",
        "Events",
        "Censored",
        "1-year survival",
        "3-year survival",
        "5-year survival",
        "Log-rank p-value",
    }.issubset(metric_labels)
    assert any("camera button" in caption.value for caption in app_test.caption)
    image_format = next(
        item
        for item in app_test.radio
        if item.label == "Plot image download format"
    )
    assert image_format.options == ["PNG", "SVG"]

    treatment_filter = next(
        item for item in app_test.multiselect if item.label == "Filter by treatment"
    )
    app_test = treatment_filter.set_value(["A"]).run(timeout=30)

    metrics = {metric.label: metric.value for metric in app_test.metric}
    assert metrics["Current cohort"] == "4"
    assert metrics["Events"] == "2"
    assert metrics["Censored"] == "2"
