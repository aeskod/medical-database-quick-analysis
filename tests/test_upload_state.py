from io import BytesIO

import pandas as pd

import app
from src.upload_state import (
    dataframe_content_digest,
    dataset_content_signature,
    uploaded_file_content_digest,
)


class NamedBytesIO(BytesIO):
    name = "clinical.csv"


def test_uploaded_file_digest_tracks_bytes_and_preserves_stream_position():
    first = NamedBytesIO(b"time,status\n10,1\n20,0\n")
    second = NamedBytesIO(b"time,status\n90,0\n80,1\n")
    first.seek(7)

    first_digest = uploaded_file_content_digest(first)

    assert first.tell() == 7
    assert first_digest == uploaded_file_content_digest(
        NamedBytesIO(b"time,status\n10,1\n20,0\n")
    )
    assert first_digest != uploaded_file_content_digest(second)


def test_dataframe_digest_changes_when_only_cell_values_change():
    first = pd.DataFrame({"time": [10, 20], "status": [1, 0]})
    second = pd.DataFrame({"time": [90, 80], "status": [0, 1]})

    assert first.shape == second.shape
    assert list(first.columns) == list(second.columns)
    assert dataframe_content_digest(first) != dataframe_content_digest(second)


def test_dataset_signature_accounts_for_parser_and_ignores_basename():
    df = pd.DataFrame({"time": [10, 20], "status": [1, 0]})
    digest = "a" * 64

    csv_signature = dataset_content_signature(
        "clinical.csv",
        df,
        content_digest=digest,
    )

    assert csv_signature == dataset_content_signature(
        "renamed.CSV",
        df,
        content_digest=digest,
    )
    assert csv_signature != dataset_content_signature(
        "clinical.tsv",
        df,
        content_digest=digest,
    )


def test_dataset_replacement_clears_all_derived_state_but_keeps_unrelated_state(
    monkeypatch,
):
    first = pd.DataFrame({"time": [10, 20], "status": [1, 0]})
    second = pd.DataFrame({"time": [90, 80], "status": [0, 1]})
    first_digest = dataframe_content_digest(first)
    second_digest = dataframe_content_digest(second)
    state = {"unrelated_preference": "keep"}
    monkeypatch.setattr(app.st, "session_state", state)

    assert not app._sync_uploaded_dataset_state(
        "clinical.csv",
        first,
        content_digest=first_digest,
    )

    stale_keys = [
        "survival_config",
        "survival_ready_df",
        "data_quality_report",
        "cohort_group_col",
        "cohort_continuous_vars",
        "cohort_categorical_vars",
        "chart_type_label",
        "chart_x_col",
        "chart_y_col",
        "chart_color_col",
        "survival_analysis_group_col",
        "survival_timepoints",
        "group_value_labels",
        "time_col_recommended",
        "event_col_all_columns",
        "event_values_status",
        "censor_values_status",
        "missing_event_handling_status",
        "unmapped_event_handling_status",
        "time_unit_time",
        "survival_group_label_sex_A",
        "column_annotation_editor_3",
    ]
    state.update({key: object() for key in stale_keys})

    replaced = app._sync_uploaded_dataset_state(
        "clinical.csv",
        second,
        content_digest=second_digest,
    )

    assert replaced
    assert all(key not in state for key in stale_keys)
    assert set(state["column_annotations"]) == {"time", "status"}
    assert state["unrelated_preference"] == "keep"
    pd.testing.assert_frame_equal(state["uploaded_df"], second)
    assert state["uploaded_dataset_signature"] == (
        f"sha256:{second_digest};parser-extension:.csv"
    )


def test_same_content_preserves_confirmed_mapping_and_returns_not_replaced(
    monkeypatch,
):
    df = pd.DataFrame({"time": [10, 20], "status": [1, 0]})
    digest = dataframe_content_digest(df)
    state = {}
    monkeypatch.setattr(app.st, "session_state", state)
    app._sync_uploaded_dataset_state("clinical.csv", df, content_digest=digest)
    preserved_config = object()
    preserved_ready_df = object()
    state["survival_config"] = preserved_config
    state["survival_ready_df"] = preserved_ready_df

    replaced = app._sync_uploaded_dataset_state(
        "renamed.csv",
        df.copy(deep=True),
        content_digest=digest,
    )

    assert not replaced
    assert state["survival_config"] is preserved_config
    assert state["survival_ready_df"] is preserved_ready_df
