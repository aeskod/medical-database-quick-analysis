import pandas as pd
from streamlit.testing.v1 import AppTest

from src.column_annotations import build_default_annotations
from src.profiling import profile_dataframe
from src.survival_mapping import SurvivalConfig, create_survival_ready_dataframe


def data_quality_harness(df, profile, config, ready, annotations):
    import streamlit as st

    from app import _render_data_quality_tab

    st.session_state["uploaded_df"] = df
    st.session_state["profile_df"] = profile
    st.session_state["survival_config"] = config
    st.session_state["survival_ready_df"] = ready
    st.session_state["column_annotations"] = annotations
    _render_data_quality_tab()


def test_data_quality_tab_integrates_cards_heatmap_and_clinical_checks():
    df = pd.DataFrame(
        {
            "patient_id": ["A", "B", "B", "D"],
            "age": [40, -1, 121, 60],
            "diagnosis_date": ["2024-01-10", "2024-01-10", "2024-01-10", "bad-date"],
            "event_date": ["2024-01-09", None, "2024-01-20", "2024-02-01"],
            "last_followup_date": ["2024-01-08", "2024-01-30", "2024-01-30", "2024-02-10"],
        }
    )
    profile = profile_dataframe(df)
    config = SurvivalConfig(
        time_col=None,
        event_col=None,
        event_values=[],
        censor_values=[],
        id_col="patient_id",
        time_source="dates",
        start_date_col="diagnosis_date",
        event_date_col="event_date",
        last_followup_date_col="last_followup_date",
        missing_event_handling="treat_as_censored",
        time_unit="days",
    )
    ready = create_survival_ready_dataframe(df, config)
    annotations = build_default_annotations(df, profile, config)

    app_test = AppTest.from_function(
        data_quality_harness,
        args=(df, profile, config, ready, annotations),
    ).run(timeout=15)

    assert not app_test.exception
    metrics = {metric.label: metric.value for metric in app_test.metric}
    assert metrics == {
        "Missing cells": "5.0%",
        "Duplicate IDs": "1",
        "Invalid ages": "2",
        "Invalid event values": "0",
        "Zero follow-up rows": "0",
        "Analysis-ready rows": "2 / 4",
    }
    assert "Age and date checks" in [item.value for item in app_test.subheader]
    assert app_test.get("plotly_chart")
    issue_messages = [
        issue.message
        for issue in app_test.session_state["data_quality_report"]["issues"]
    ]
    assert "Age values must be numeric and between 0 and 120." in issue_messages
    assert "Event date before diagnosis/start date found." in issue_messages
    assert "Last follow-up before start date found." in issue_messages
