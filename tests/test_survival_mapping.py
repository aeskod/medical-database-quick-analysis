import pandas as pd

from src.survival_mapping import (
    SurvivalConfig,
    create_binary_event_series,
    create_cleaned_mapped_dataframe,
    create_survival_ready_dataframe,
    derive_survival_from_dates,
    validate_survival_config,
)


def test_create_binary_event_series_excludes_missing_and_unmapped_values():
    df = pd.DataFrame({"status": [1, 0, None, "unknown", "1", "0"]})
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
    )

    event_series = create_binary_event_series(df, config)

    assert str(event_series.dtype) == "Int64"
    assert event_series.tolist() == [1, 0, pd.NA, pd.NA, 1, 0]


def test_create_binary_event_series_can_treat_missing_as_censored():
    df = pd.DataFrame({"status": [1, None, 0]})
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        missing_event_handling="treat_as_censored",
    )

    event_series = create_binary_event_series(df, config)

    assert event_series.tolist() == [1, 0, 0]


def test_create_binary_event_series_can_handle_unmapped_values_explicitly():
    df = pd.DataFrame({"status": ["Dead", "Alive", "Unknown", "Pending"]})
    censored_config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=["Dead"],
        censor_values=["Alive"],
        unmapped_event_handling="treat_as_censored",
    )
    event_config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=["Dead"],
        censor_values=["Alive"],
        unmapped_event_handling="treat_as_event",
    )

    assert create_binary_event_series(df, censored_config).tolist() == [1, 0, 0, 0]
    assert create_binary_event_series(df, event_config).tolist() == [1, 0, 1, 1]


def test_validate_survival_config_valid_mapping_returns_warnings_not_errors():
    df = pd.DataFrame(
        {
            "time": [10, 20, 30, 40, 50, 60],
            "status": [1, 0, 1, 0, 1, 0],
            "patient_id": ["A", "B", "C", "D", "E", "F"],
            "sex": ["M", "F", "M", "F", "M", "F"],
        }
    )
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        id_col="patient_id",
        group_col="sex",
        time_unit="days",
    )

    errors, warnings = validate_survival_config(df, config)

    assert errors == []
    assert any("Event count is very low" in warning for warning in warnings)
    assert any("Censored count is very low" in warning for warning in warnings)


def test_validate_survival_config_reports_blocking_errors():
    df = pd.DataFrame({"time": [1, -2, 3], "status": [1, 0, 1]})
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[1],
    )

    errors, warnings = validate_survival_config(df, config)

    assert any("negative" in error for error in errors)
    assert any("overlap" in error for error in errors)
    assert any("No patient ID column selected" in warning for warning in warnings)


def test_validate_survival_config_rejects_invalid_unmapped_handling():
    df = pd.DataFrame({"time": [1, 2], "status": [1, 0]})
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        unmapped_event_handling="guess",
    )

    errors, _ = validate_survival_config(df, config)

    assert any("Unmapped event handling" in error for error in errors)


def test_validate_survival_config_describes_missing_and_unmapped_actions():
    df = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "status": ["Dead", "Alive", "Unknown", None, "Dead", "Alive"],
        }
    )
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=["Dead"],
        censor_values=["Alive"],
        missing_event_handling="treat_as_censored",
        unmapped_event_handling="treat_as_event",
    )

    _, warnings = validate_survival_config(df, config)

    assert any("missing values" in warning and "treated as censored" in warning for warning in warnings)
    assert any("unmapped values" in warning and "treated as events" in warning for warning in warnings)


def test_validate_survival_config_reports_no_usable_rows():
    df = pd.DataFrame({"time": [1, 2, 3], "status": ["unknown", "other", "pending"]})
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=["dead"],
        censor_values=["alive"],
    )

    errors, warnings = validate_survival_config(df, config)

    assert any("No usable rows remain" in error for error in errors)
    assert any("unmapped values" in warning for warning in warnings)


def test_create_survival_ready_dataframe_drops_unusable_rows_and_keeps_optional_columns():
    df = pd.DataFrame(
        {
            "time": [10, "20", None, 30],
            "status": ["Dead", "Alive", "Dead", "unknown"],
            "patient_id": ["A", "B", "C", "D"],
            "arm": ["A", "B", "A", "B"],
        }
    )
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=["Dead"],
        censor_values=["Alive"],
        id_col="patient_id",
        group_col="arm",
    )

    survival_ready_df = create_survival_ready_dataframe(df, config)

    assert list(survival_ready_df.columns) == ["_time", "_event", "_id", "_group"]
    assert survival_ready_df["_time"].tolist() == [10, 20]
    assert survival_ready_df["_event"].tolist() == [1, 0]
    assert survival_ready_df["_id"].tolist() == ["A", "B"]


def test_derive_survival_from_dates_uses_event_or_last_followup_date():
    df = pd.DataFrame(
        {
            "start": ["2024-01-01", "2024-01-10", "2024-02-01"],
            "event_date": ["2024-01-11", None, "2024-02-01 12:00"],
            "last_followup": ["2024-01-20", "2024-02-09", "2024-03-01"],
            "patient_id": ["A", "B", "C"],
        }
    )
    config = SurvivalConfig(
        time_col=None,
        event_col=None,
        event_values=[],
        censor_values=[],
        time_source="dates",
        start_date_col="start",
        event_date_col="event_date",
        last_followup_date_col="last_followup",
        missing_event_handling="treat_as_censored",
        id_col="patient_id",
        time_unit="days",
    )

    derived = derive_survival_from_dates(df, config)
    ready = create_survival_ready_dataframe(df, config)

    assert derived["_time"].tolist() == [10.0, 30.0, 0.5]
    assert derived["_event"].tolist() == [1, 0, 1]
    assert ready["_id"].tolist() == ["A", "B", "C"]


def test_date_derivation_excludes_missing_event_dates_when_requested():
    df = pd.DataFrame(
        {
            "start": ["2024-01-01", "2024-01-01"],
            "event_date": ["2024-01-03", None],
            "last_followup": ["2024-01-05", "2024-01-10"],
        }
    )
    config = SurvivalConfig(
        time_col=None,
        event_col=None,
        event_values=[],
        censor_values=[],
        time_source="dates",
        start_date_col="start",
        event_date_col="event_date",
        last_followup_date_col="last_followup",
        missing_event_handling="exclude",
        time_unit="days",
    )

    ready = create_survival_ready_dataframe(df, config)

    assert ready[["_time", "_event"]].to_dict("records") == [
        {"_time": 2.0, "_event": 1},
    ]


def test_validate_date_derivation_rejects_unparseable_and_negative_dates():
    df = pd.DataFrame(
        {
            "start": ["2024-01-10", "not-a-date"],
            "event_date": ["2024-01-05", None],
            "last_followup": ["2024-01-20", "2024-02-01"],
        }
    )
    config = SurvivalConfig(
        time_col=None,
        event_col=None,
        event_values=[],
        censor_values=[],
        time_source="dates",
        start_date_col="start",
        event_date_col="event_date",
        last_followup_date_col="last_followup",
        missing_event_handling="treat_as_censored",
        time_unit="days",
    )

    errors, _ = validate_survival_config(df, config)

    assert any("start" in error and "unparseable" in error for error in errors)
    assert any("negative" in error for error in errors)


def test_validation_rejects_infinite_time_and_unknown_time_unit():
    df = pd.DataFrame({"time": [1, float("inf")], "status": [1, 0]})
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        time_unit="fortnights",
    )

    errors, _ = validate_survival_config(df, config)
    ready = create_survival_ready_dataframe(df, config)

    assert any("infinite" in error for error in errors)
    assert any("Time unit" in error for error in errors)
    assert ready["_time"].tolist() == [1.0]


def test_validation_rejects_same_column_for_multiple_roles():
    df = pd.DataFrame({"outcome": [1, 0], "group": ["A", "B"]})
    config = SurvivalConfig(
        time_col="outcome",
        event_col="outcome",
        event_values=[1],
        censor_values=[0],
        group_col="group",
    )

    errors, _ = validate_survival_config(df, config)

    assert any("different column" in error and "outcome" in error for error in errors)


def test_cleaned_export_does_not_overwrite_reserved_source_columns():
    df = pd.DataFrame(
        {
            "time": [1, 2],
            "status": [1, 0],
            "_time": ["source-a", "source-b"],
            "_event": ["source-c", "source-d"],
        }
    )
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
    )

    result = create_cleaned_mapped_dataframe(df, config)

    assert result["_time"].tolist() == ["source-a", "source-b"]
    assert result["_event"].tolist() == ["source-c", "source-d"]
    assert result["_time_mapped"].tolist() == [1, 2]
    assert result["_event_mapped"].tolist() == [1, 0]


def test_validation_rejects_ambiguous_dates_and_event_after_followup():
    df = pd.DataFrame(
        {
            "start": ["03/04/2024"],
            "event_date": ["2024-05-10"],
            "last_followup": ["2024-05-01"],
        }
    )
    config = SurvivalConfig(
        time_col=None,
        event_col=None,
        event_values=[],
        censor_values=[],
        time_source="dates",
        start_date_col="start",
        event_date_col="event_date",
        last_followup_date_col="last_followup",
        time_unit="days",
    )

    errors, _ = validate_survival_config(df, config)

    assert any("ambiguous" in error and "ISO" in error for error in errors)
    assert any("after last follow-up" in error for error in errors)
