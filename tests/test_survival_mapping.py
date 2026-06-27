import pandas as pd

from src.survival_mapping import (
    SurvivalConfig,
    create_binary_event_series,
    create_survival_ready_dataframe,
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
