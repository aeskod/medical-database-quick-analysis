import pandas as pd
from streamlit.testing.v1 import AppTest

from app import _assess_patient_level_structure
from src.profiling import profile_dataframe


def survival_setup_harness(df, profile):
    from app import _render_survival_setup

    _render_survival_setup(df, profile)


def test_patient_level_suitability_detects_repeated_and_missing_ids():
    repeated = pd.DataFrame({"patient_id": ["P1", "P1", "P2", None]})
    missing = pd.DataFrame({"patient_id": ["P1", None, "P2"]})

    repeated_status, repeated_message = _assess_patient_level_structure(
        repeated,
        "patient_id",
    )
    missing_status, missing_message = _assess_patient_level_structure(
        missing,
        "patient_id",
    )

    assert repeated_status == "warning"
    assert "1 patient_id value repeat across 2 rows" in repeated_message
    assert "repeated visits or records" in repeated_message
    assert missing_status == "warning"
    assert "missing in 1 row" in missing_message


def test_validation_summary_is_persistent_and_links_to_invalid_controls():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2"],
            "time": [10, -2, 30],
            "status": ["Unknown", "Pending", "Unknown"],
        }
    )
    app_test = AppTest.from_function(
        survival_setup_harness,
        args=(df, profile_dataframe(df)),
    )
    app_test.session_state["id_col_recommended"] = "__search_all__"
    app_test.session_state["id_col_all_columns"] = "patient_id"
    app_test.run(timeout=20)

    assert not app_test.exception
    confirm = next(
        button
        for button in app_test.button
        if button.label == "Confirm survival mapping"
    )
    issue_markdown = [block.value for block in app_test.markdown]

    assert confirm.disabled
    assert "#### Validation summary" in issue_markdown
    assert any(
        "negative values" in value and "href='#follow-up-time'" in value
        for value in issue_markdown
    )
    assert any(
        "event value" in value.lower() and "href='#event-values'" in value
        for value in issue_markdown
    )
    assert any(
        "unmapped values" in value.lower() and "href='#event-values'" in value
        for value in issue_markdown
    )
    assert any(
        "Poor fit for basic patient-level analysis" in warning.value
        for warning in app_test.warning
    )
    assert any(
        "not currently suitable for survival analysis" in warning.value
        for warning in app_test.warning
    )


def test_valid_mapping_can_continue_to_data_quality():
    csv_bytes = (
        b"patient_id,time,status\n"
        b"A,10,1\nB,20,0\nC,30,1\nD,40,0\nE,50,1\nF,60,0\n"
    )
    app_test = AppTest.from_file("app.py").run(timeout=30)
    app_test.file_uploader[0].upload(
        "clinical.csv",
        csv_bytes,
        "text/csv",
    ).run(timeout=30)
    confirm = next(
        button
        for button in app_test.button
        if button.label == "Confirm survival mapping"
    )

    app_test = confirm.click().run(timeout=30)

    assert not app_test.exception
    assert any(
        message.value == "Mapping is valid and confirmed."
        for message in app_test.success
    )
    continue_button = next(
        button
        for button in app_test.button
        if button.label == "Continue to Data Quality"
    )

    app_test = continue_button.click().run(timeout=30)

    assert not app_test.exception
    assert app_test.session_state["main_tab"] == "Data Quality"
