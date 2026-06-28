from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.profiling import normalize_missing_values


ALLOWED_MISSING_EVENT_HANDLING = {"exclude", "treat_as_censored"}
ALLOWED_UNMAPPED_EVENT_HANDLING = {
    "exclude",
    "treat_as_censored",
    "treat_as_event",
}
ALLOWED_TIME_SOURCES = {"duration", "dates"}


@dataclass
class SurvivalConfig:
    time_col: Optional[str]
    event_col: Optional[str]
    event_values: list[Any]
    censor_values: list[Any]
    id_col: Optional[str] = None
    group_col: Optional[str] = None
    time_unit: str = "unknown"
    missing_event_handling: str = "exclude"
    unmapped_event_handling: str = "exclude"
    time_source: str = "duration"
    start_date_col: Optional[str] = None
    event_date_col: Optional[str] = None
    last_followup_date_col: Optional[str] = None


def validate_survival_config(
    df: pd.DataFrame,
    config: SurvivalConfig,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    normalized = normalize_missing_values(df)

    if config.time_source not in ALLOWED_TIME_SOURCES:
        errors.append("Time source must be 'duration' or 'dates'.")

    if config.missing_event_handling not in ALLOWED_MISSING_EVENT_HANDLING:
        errors.append("Missing event handling must be 'exclude' or 'treat_as_censored'.")

    if config.unmapped_event_handling not in ALLOWED_UNMAPPED_EVENT_HANDLING:
        errors.append(
            "Unmapped event handling must be 'exclude', "
            "'treat_as_censored', or 'treat_as_event'."
        )

    if config.time_source == "dates":
        _validate_date_derivation(normalized, config, errors, warnings)
    elif config.time_source == "duration":
        time_exists = config.time_col is not None and config.time_col in normalized.columns
        event_exists = config.event_col is not None and config.event_col in normalized.columns

        if not time_exists:
            errors.append(f"Time column '{config.time_col}' is missing from the dataset.")
        if not event_exists:
            errors.append(f"Event column '{config.event_col}' is missing from the dataset.")
        if not config.event_values:
            errors.append("At least one event value must be selected.")

        event_value_keys = {_canonical_value(value) for value in config.event_values}
        censor_value_keys = {_canonical_value(value) for value in config.censor_values}
        if event_value_keys & censor_value_keys:
            errors.append("Event values and censor values cannot overlap.")

        if time_exists:
            time_series = normalized[config.time_col]
            time_non_missing = time_series.dropna()
            parsed_time = pd.to_numeric(time_series, errors="coerce")

            if time_non_missing.empty:
                errors.append("Time column has no non-missing values.")
            elif parsed_time.loc[time_non_missing.index].isna().any():
                errors.append("Time column must contain numeric values.")

            parsed_non_missing_time = parsed_time.dropna()
            if not parsed_non_missing_time.empty and (parsed_non_missing_time < 0).any():
                errors.append("Time column contains negative values.")
            if time_series.isna().any():
                warnings.append("Time column has missing values; those rows will be excluded.")
            if not parsed_non_missing_time.empty and (parsed_non_missing_time == 0).any():
                warnings.append("Time column contains zero values.")

        if event_exists:
            event_series = normalized[config.event_col]
            event_non_missing = event_series.dropna()
            if event_non_missing.empty:
                errors.append("Event column has no non-missing values.")

            if event_series.isna().any():
                missing_action = (
                    "will be treated as censored"
                    if config.missing_event_handling == "treat_as_censored"
                    else "will be excluded"
                )
                warnings.append(f"Event column has missing values; those rows {missing_action}.")

            mapped_value_keys = event_value_keys | censor_value_keys
            unmapped_values = [
                value
                for value in pd.unique(event_non_missing)
                if _canonical_value(value) not in mapped_value_keys
            ]
            if unmapped_values:
                unmapped_action = {
                    "exclude": "will be excluded",
                    "treat_as_censored": "will be treated as censored",
                    "treat_as_event": "will be treated as events",
                }.get(config.unmapped_event_handling, "have invalid handling")
                warnings.append(
                    "Event column contains unmapped values: "
                    + ", ".join(_format_value(value) for value in unmapped_values[:10])
                    + f"; those rows {unmapped_action}."
                )

    if config.id_col is None:
        warnings.append("No patient ID column selected; row number will be used.")
    elif config.id_col not in normalized.columns:
        warnings.append(f"Patient ID column '{config.id_col}' is missing from the dataset.")
    else:
        id_values = normalized[config.id_col].dropna()
        if id_values.duplicated().any():
            warnings.append("Patient ID column has duplicate values.")

    if config.group_col is not None:
        if config.group_col not in normalized.columns:
            warnings.append(f"Group column '{config.group_col}' is missing from the dataset.")
        else:
            group_values = normalized[config.group_col].dropna()
            group_count = int(group_values.nunique(dropna=True))
            if group_count < 2:
                warnings.append("Group column has fewer than 2 groups.")

            small_groups = group_values.value_counts(dropna=True)
            small_groups = small_groups[small_groups < 5]
            if not small_groups.empty:
                warnings.append(
                    "Group column has groups with fewer than 5 rows: "
                    + ", ".join(_format_value(value) for value in small_groups.index[:10])
                )

    if not errors:
        survival_ready_df = create_survival_ready_dataframe(normalized, config)
        if survival_ready_df.empty:
            errors.append("No usable rows remain after applying time and event requirements.")
        else:
            event_count = int((survival_ready_df["_event"] == 1).sum())
            censored_count = int((survival_ready_df["_event"] == 0).sum())

            if event_count < 5:
                warnings.append(f"Event count is very low ({event_count}).")

            if censored_count < 5:
                warnings.append(f"Censored count is very low ({censored_count}).")

    return errors, warnings


def create_binary_event_series(df: pd.DataFrame, config: SurvivalConfig) -> pd.Series:
    normalized = normalize_missing_values(df)
    if config.event_col is None or config.event_col not in normalized.columns:
        return pd.Series(pd.NA, index=df.index, name="_event", dtype="Int64")
    event_series = normalized[config.event_col]
    event_value_keys = {_canonical_value(value) for value in config.event_values}
    censor_value_keys = {_canonical_value(value) for value in config.censor_values}

    mapped_values = []
    for value in event_series:
        if pd.isna(value):
            mapped_values.append(0 if config.missing_event_handling == "treat_as_censored" else pd.NA)
            continue

        value_key = _canonical_value(value)
        if value_key in event_value_keys:
            mapped_values.append(1)
        elif value_key in censor_value_keys:
            mapped_values.append(0)
        else:
            mapped_values.append(
                {
                    "treat_as_censored": 0,
                    "treat_as_event": 1,
                }.get(config.unmapped_event_handling, pd.NA)
            )

    return pd.Series(mapped_values, index=df.index, name="_event", dtype="Int64")


def create_survival_ready_dataframe(df: pd.DataFrame, config: SurvivalConfig) -> pd.DataFrame:
    normalized = normalize_missing_values(df)
    if config.time_source == "dates":
        survival_df = derive_survival_from_dates(normalized, config)
    else:
        survival_df = pd.DataFrame(index=normalized.index)
        if config.time_col is None or config.time_col not in normalized.columns:
            survival_df["_time"] = pd.Series(float("nan"), index=normalized.index)
        else:
            survival_df["_time"] = pd.to_numeric(normalized[config.time_col], errors="coerce")
        survival_df["_event"] = create_binary_event_series(normalized, config)

    if config.id_col is not None and config.id_col in normalized.columns:
        survival_df["_id"] = normalized[config.id_col]

    if config.group_col is not None and config.group_col in normalized.columns:
        survival_df["_group"] = normalized[config.group_col]

    return survival_df.dropna(subset=["_time", "_event"]).reset_index(drop=True)


def derive_survival_from_dates(df: pd.DataFrame, config: SurvivalConfig) -> pd.DataFrame:
    normalized = normalize_missing_values(df)
    result = pd.DataFrame(index=normalized.index)
    required_columns = (
        config.start_date_col,
        config.event_date_col,
        config.last_followup_date_col,
    )
    if any(column is None or column not in normalized.columns for column in required_columns):
        result["_time"] = pd.Series(float("nan"), index=normalized.index)
        result["_event"] = pd.Series(pd.NA, index=normalized.index, dtype="Int64")
        return result

    start_dates = _parse_dates(normalized[config.start_date_col])
    event_dates = _parse_dates(normalized[config.event_date_col])
    last_followup_dates = _parse_dates(normalized[config.last_followup_date_col])
    event_date_missing = normalized[config.event_date_col].isna()

    end_dates = event_dates.where(~event_date_missing, last_followup_dates)
    result["_time"] = (end_dates - start_dates).dt.total_seconds() / 86_400

    events = pd.Series(pd.NA, index=normalized.index, dtype="Int64")
    events.loc[event_dates.notna()] = 1
    if config.missing_event_handling == "treat_as_censored":
        events.loc[event_date_missing & last_followup_dates.notna()] = 0
    result["_event"] = events
    return result


def _validate_date_derivation(
    normalized: pd.DataFrame,
    config: SurvivalConfig,
    errors: list[str],
    warnings: list[str],
) -> None:
    columns = {
        "Start date": config.start_date_col,
        "Event date": config.event_date_col,
        "Last follow-up date": config.last_followup_date_col,
    }
    selected = [column for column in columns.values() if column is not None]
    if len(selected) != len(set(selected)):
        errors.append("Start, event, and last follow-up date columns must be different.")

    for label, column in columns.items():
        if column is None:
            errors.append(f"{label} column must be selected.")
        elif column not in normalized.columns:
            errors.append(f"{label} column '{column}' is missing from the dataset.")
        else:
            raw_dates = normalized[column]
            parsed_dates = _parse_dates(raw_dates)
            unparseable_count = int((raw_dates.notna() & parsed_dates.isna()).sum())
            if unparseable_count:
                errors.append(
                    f"{label} column '{column}' contains {unparseable_count} unparseable value(s)."
                )

    if any(column is None or column not in normalized.columns for column in columns.values()):
        return

    derived = derive_survival_from_dates(normalized, config)
    parsed_time = derived["_time"]
    if parsed_time.dropna().empty:
        errors.append("Date derivation produced no survival times.")
    if (parsed_time < 0).any():
        errors.append("Date-derived survival time contains negative values.")
    if parsed_time.isna().any():
        warnings.append("Some rows cannot produce a survival time and will be excluded.")
    if (parsed_time == 0).any():
        warnings.append("Date-derived survival time contains zero values.")

    event_date_missing = normalized[config.event_date_col].isna()
    if event_date_missing.any():
        action = (
            "will be censored at last follow-up"
            if config.missing_event_handling == "treat_as_censored"
            else "will be excluded"
        )
        warnings.append(f"Event date is missing for some rows; those rows {action}.")


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed", utc=True)


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"

    if isinstance(value, str):
        return value.strip().lower()

    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()

    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))
        return str(float_value)

    return str(value).strip().lower()


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    return str(value)
