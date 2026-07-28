import pandas as pd
from streamlit.testing.v1 import AppTest

from app import _default_censor_values, _default_event_values, _suggest_time_unit


def event_mapping_harness(df):
    import streamlit as st

    from app import _render_event_value_mapping

    result = _render_event_value_mapping(df, "status")
    st.write(result)


def test_conservative_censor_defaults_leave_unknown_values_unmapped():
    unique_values = ["Alive", "Dead", "Unknown", "Pending"]

    defaults = _default_censor_values(unique_values, ["Dead"])

    assert defaults == ["Alive"]


def test_two_value_numeric_status_keeps_confident_binary_default():
    assert _default_censor_values([1, 2], [2]) == [1]
    assert _default_censor_values([0, 1], [1]) == [0]


def test_ambiguous_values_are_not_guessed_as_censored():
    assert _default_censor_values(
        ["Complete", "Pending", "Unknown"],
        [],
    ) == []


def test_nonstandard_numeric_binary_values_are_not_guessed_as_events():
    assert _default_event_values([1, 2]) == []
    assert _default_event_values([0, 1]) == [1]


def test_week_time_unit_is_detected():
    assert _suggest_time_unit("week") == "weeks"
    assert _suggest_time_unit("followup_wk") == "weeks"


def test_event_mapping_ui_defaults_to_explicit_alive_and_excludes_unknown():
    df = pd.DataFrame(
        {
            "status": [
                "Alive",
                "Dead",
                "Unknown",
                "Alive",
                "Dead",
                "Pending",
            ]
        }
    )
    app_test = AppTest.from_function(event_mapping_harness, args=(df,))

    app_test.run(timeout=10)

    assert not app_test.exception
    event_values = next(
        item
        for item in app_test.multiselect
        if item.label == "Which value(s) mean the event occurred?"
    )
    censor_values = next(
        item
        for item in app_test.multiselect
        if item.label == "Which value(s) explicitly mean censored?"
    )
    unmapped_handling = next(
        item
        for item in app_test.radio
        if item.label == "Unmapped non-missing values"
    )
    missing_handling = next(
        item
        for item in app_test.radio
        if item.label == "Missing event values"
    )

    assert event_values.value == ["Dead"]
    assert censor_values.value == ["Alive"]
    assert unmapped_handling.value == "Exclude from survival analysis"
    assert missing_handling.value == "Exclude from survival analysis"
    assert any(
        "Currently unmapped: Unknown, Pending" in warning.value
        for warning in app_test.warning
    )
