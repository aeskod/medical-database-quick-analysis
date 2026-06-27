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


@dataclass
class SurvivalConfig:
    time_col: str
    event_col: str
    event_values: list[Any]
    censor_values: list[Any]
    id_col: Optional[str] = None
    group_col: Optional[str] = None
    time_unit: str = "unknown"
    missing_event_handling: str = "exclude"
    unmapped_event_handling: str = "exclude"


def validate_survival_config(
    df: pd.DataFrame,
    config: SurvivalConfig,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    normalized = normalize_missing_values(df)

    if config.missing_event_handling not in ALLOWED_MISSING_EVENT_HANDLING:
        errors.append("Missing event handling must be 'exclude' or 'treat_as_censored'.")

    if config.unmapped_event_handling not in ALLOWED_UNMAPPED_EVENT_HANDLING:
        errors.append(
            "Unmapped event handling must be 'exclude', "
            "'treat_as_censored', or 'treat_as_event'."
        )

    time_exists = config.time_col in normalized.columns
    event_exists = config.event_col in normalized.columns

    if not time_exists:
        errors.append(f"Time column '{config.time_col}' is missing from the dataset.")

    if not event_exists:
        errors.append(f"Event column '{config.event_col}' is missing from the dataset.")

    if not config.event_values:
        errors.append("At least one event value must be selected.")

    event_value_keys = {_canonical_value(value) for value in config.event_values}
    censor_value_keys = {_canonical_value(value) for value in config.censor_values}
    overlap = event_value_keys & censor_value_keys
    if overlap:
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
            warnings.append(
                f"Event column has missing values; those rows {missing_action}."
            )

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

    if not _has_blocking_column_errors(errors):
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
    survival_df = pd.DataFrame(index=normalized.index)

    survival_df["_time"] = pd.to_numeric(normalized[config.time_col], errors="coerce")
    survival_df["_event"] = create_binary_event_series(normalized, config)

    if config.id_col is not None and config.id_col in normalized.columns:
        survival_df["_id"] = normalized[config.id_col]

    if config.group_col is not None and config.group_col in normalized.columns:
        survival_df["_group"] = normalized[config.group_col]

    return survival_df.dropna(subset=["_time", "_event"]).reset_index(drop=True)


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


def _has_blocking_column_errors(errors: list[str]) -> bool:
    blocking_fragments = (
        "is missing from the dataset",
        "must contain numeric values",
        "has no non-missing values",
        "contains negative values",
        "At least one event value",
        "cannot overlap",
        "Missing event handling",
        "Unmapped event handling",
    )
    return any(any(fragment in error for fragment in blocking_fragments) for error in errors)
