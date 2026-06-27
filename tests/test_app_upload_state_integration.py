import hashlib

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.survival_mapping import SurvivalConfig, create_survival_ready_dataframe


FIRST_CSV = b"time,status\n10,1\n20,0\n"
CHANGED_CSV = b"time,status\n90,0\n80,1\n"


def _config() -> SurvivalConfig:
    return SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        time_unit="days",
    )


def test_reupload_changed_content_with_same_name_resets_full_app_state():
    app_test = AppTest.from_file("app.py").run(timeout=30)
    app_test.file_uploader[0].upload(
        "clinical.csv",
        FIRST_CSV,
        "text/csv",
    ).run(timeout=30)
    assert not app_test.exception

    first_df = app_test.session_state["uploaded_df"]
    config = _config()
    app_test.session_state["survival_config"] = config
    app_test.session_state["survival_ready_df"] = create_survival_ready_dataframe(
        first_df,
        config,
    )
    app_test.session_state["chart_x_col"] = "status"
    app_test.session_state["event_values_status"] = [0]

    app_test.file_uploader[0].clear().upload(
        "clinical.csv",
        CHANGED_CSV,
        "text/csv",
    ).run(timeout=30)

    assert not app_test.exception
    assert "survival_config" not in app_test.session_state
    assert "survival_ready_df" not in app_test.session_state
    assert app_test.session_state["chart_x_col"] == "time"
    assert "event_values_status" in app_test.session_state
    assert app_test.session_state["event_values_status"] == [1]
    assert app_test.session_state["uploaded_dataset_signature"] == (
        f"sha256:{hashlib.sha256(CHANGED_CSV).hexdigest()}"
    )
    pd.testing.assert_frame_equal(
        app_test.session_state["uploaded_df"],
        pd.DataFrame({"time": [90, 80], "status": [0, 1]}),
    )
    changed_profile = app_test.session_state["profile_df"].set_index("column_name")
    assert changed_profile.loc["time", "min_value"] == 80
    assert changed_profile.loc["time", "max_value"] == 90
    assert any(
        "A different dataset was detected" in message.value
        for message in app_test.info
    )
    assert any(
        "No survival mapping has been confirmed yet" in message.value
        for message in app_test.info
    )


def test_same_content_under_new_filename_preserves_confirmed_analysis_state():
    app_test = AppTest.from_file("app.py").run(timeout=30)
    app_test.file_uploader[0].upload(
        "clinical.csv",
        FIRST_CSV,
        "text/csv",
    ).run(timeout=30)
    assert not app_test.exception

    df = app_test.session_state["uploaded_df"]
    config = _config()
    ready_df = create_survival_ready_dataframe(df, config)
    app_test.session_state["survival_config"] = config
    app_test.session_state["survival_ready_df"] = ready_df

    app_test.file_uploader[0].clear().upload(
        "renamed.csv",
        FIRST_CSV,
        "text/csv",
    ).run(timeout=30)

    assert not app_test.exception
    assert app_test.session_state["survival_config"] == config
    pd.testing.assert_frame_equal(
        app_test.session_state["survival_ready_df"],
        ready_df,
    )
    assert not any(
        "A different dataset was detected" in message.value
        for message in app_test.info
    )
