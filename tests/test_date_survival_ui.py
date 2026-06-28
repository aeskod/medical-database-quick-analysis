import pandas as pd
from streamlit.testing.v1 import AppTest

from src.profiling import profile_dataframe


def date_setup_harness(df, profile):
    from app import _render_survival_setup

    _render_survival_setup(df, profile)


def _date_dataset():
    return pd.DataFrame(
        {
            "patient_id": ["A", "B", "C", "D", "E", "F"],
            "diagnosis_date": ["2024-01-01"] * 6,
            "death_date": [
                "2024-01-11",
                None,
                "2024-01-21",
                None,
                "2024-01-31",
                None,
            ],
            "last_followup_date": ["2024-02-01"] * 6,
        }
    )


def _date_mode_app():
    df = _date_dataset()
    app_test = AppTest.from_function(
        date_setup_harness,
        args=(df, profile_dataframe(df)),
    ).run(timeout=10)
    setup_mode = next(
        item for item in app_test.radio if item.label == "Survival time setup"
    )
    return app_test, setup_mode.set_value("Derive from date columns").run(timeout=10)


def test_date_setup_is_discoverable_and_uses_date_column_defaults():
    _, app_test = _date_mode_app()

    assert not app_test.exception
    selectors = {item.label: item.value for item in app_test.selectbox}
    assert selectors["Start date column"] == "diagnosis_date"
    assert selectors["Event date column"] == "death_date"
    assert selectors["Last follow-up date column"] == "last_followup_date"
    missing_dates = next(
        item
        for item in app_test.radio
        if item.label == "How should missing event dates be interpreted?"
    )
    assert missing_dates.value == "Unknown / exclude from survival analysis"
    assert any(
        "Duration is calculated in days" in caption.value
        for caption in app_test.caption
    )


def test_date_setup_confirmation_updates_config_and_derived_results():
    _, app_test = _date_mode_app()
    missing_dates = next(
        item
        for item in app_test.radio
        if item.label == "How should missing event dates be interpreted?"
    )
    app_test = missing_dates.set_value(
        "Censored / event did not occur before last follow-up"
    ).run(timeout=10)
    confirm = next(
        item for item in app_test.button if item.label == "Confirm survival mapping"
    )

    app_test = confirm.click().run(timeout=10)

    assert not app_test.exception
    config = app_test.session_state["survival_config"]
    ready = app_test.session_state["survival_ready_df"]
    assert config.time_source == "dates"
    assert config.start_date_col == "diagnosis_date"
    assert config.event_date_col == "death_date"
    assert config.last_followup_date_col == "last_followup_date"
    assert config.missing_event_handling == "treat_as_censored"
    assert config.time_unit == "days"
    assert ready["_event"].tolist() == [1, 0, 1, 0, 1, 0]
    assert ready["_time"].tolist() == [10.0, 31.0, 20.0, 31.0, 30.0, 31.0]
    assert any(
        "Usable rows: 6" in message.value
        and "Events: 3" in message.value
        and "Censored: 3" in message.value
        for message in app_test.success
    )


def test_uploaded_date_dataset_flows_through_all_dashboard_tabs():
    csv_bytes = _date_dataset().to_csv(index=False).encode()
    app_test = AppTest.from_file("app.py").run(timeout=30)
    app_test.file_uploader[0].upload(
        "date_survival.csv",
        csv_bytes,
        "text/csv",
    ).run(timeout=30)
    setup_mode = next(
        item for item in app_test.radio if item.label == "Survival time setup"
    )
    app_test = setup_mode.set_value("Derive from date columns").run(timeout=30)
    missing_dates = next(
        item
        for item in app_test.radio
        if item.label == "How should missing event dates be interpreted?"
    )
    app_test = missing_dates.set_value(
        "Censored / event did not occur before last follow-up"
    ).run(timeout=30)
    confirm = next(
        item for item in app_test.button if item.label == "Confirm survival mapping"
    )

    app_test = confirm.click().run(timeout=30)

    assert not app_test.exception
    assert len(app_test.session_state["survival_ready_df"]) == 6
    annotations = app_test.session_state["column_annotations"]
    assert annotations["diagnosis_date"].meaning == "Start time"
    assert annotations["death_date"].meaning == "Date"
    assert annotations["last_followup_date"].meaning == "End time"
    assert any(
        message.value == "Survival-ready data validated."
        for message in app_test.success
    )
    assert not any(
        "Mapped survival time column" in message.value
        or "No event values are selected" in message.value
        for message in app_test.error
    )
