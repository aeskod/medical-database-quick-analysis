from dataclasses import dataclass, field, replace
import re
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from src.profiling import normalize_missing_values
from src.survival_mapping import (
    ALLOWED_MISSING_EVENT_HANDLING,
    ALLOWED_UNMAPPED_EVENT_HANDLING,
    create_binary_event_series,
    derive_survival_from_dates,
)


HIGH_RISK_NAME_HINTS = [
    "name",
    "first_name",
    "last_name",
    "full_name",
    "email",
    "phone",
    "mobile",
    "address",
    "street",
    "passport",
    "ssn",
    "national_id",
    "id_card",
    "mrn",
    "medical_record",
]

MEDIUM_RISK_NAME_HINTS = [
    "dob",
    "date_of_birth",
    "birth_date",
    "zip",
    "postal",
    "city",
    "state",
    "country",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "notes",
    "comment",
    "free_text",
]

COMMON_DEIDENTIFIED_ID_HINTS = ["patient_id", "subject_id", "record_id"]


@dataclass
class QualityIssue:
    severity: str
    category: str
    message: str
    affected_columns: list[str] = field(default_factory=list)
    affected_rows_count: Optional[int] = None
    details: dict[str, Any] = field(default_factory=dict)


def build_data_quality_report(
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None = None,
    survival_config=None,
    survival_ready_df: pd.DataFrame | None = None,
    annotations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_df = normalize_missing_values(df)
    issues: list[QualityIssue] = []

    overview = compute_dataset_overview(normalized_df)
    missingness_by_column = compute_missingness_by_column(normalized_df)
    missingness_by_row = compute_missingness_by_row(normalized_df)
    issues.extend(detect_high_missingness(missingness_by_column))

    duplicate_rows = check_duplicate_rows(normalized_df)
    if duplicate_rows["duplicate_row_count"] > 0:
        issues.append(
            QualityIssue(
                severity="warning",
                category="duplicates",
                message="Dataset contains exact duplicate rows.",
                affected_rows_count=duplicate_rows["duplicate_row_count"],
                details=duplicate_rows,
            )
        )

    id_col = getattr(survival_config, "id_col", None) if survival_config is not None else None
    duplicate_ids = check_duplicate_patient_ids(normalized_df, id_col)
    if not duplicate_ids["checked"]:
        issues.append(
            QualityIssue(
                severity="warning",
                category="patient_id",
                message="No patient ID column selected; row number is being used.",
            )
        )
    elif duplicate_ids["duplicate_id_row_count"] > 0:
        issues.append(
            QualityIssue(
                severity="warning",
                category="patient_id",
                message=(
                    "Selected patient ID column contains duplicate values. "
                    "Basic survival analysis expects one row per patient."
                ),
                affected_columns=[duplicate_ids["id_col"]],
                affected_rows_count=duplicate_ids["duplicate_id_row_count"],
                details={
                    "duplicate_id_examples": duplicate_ids["duplicate_id_examples"],
                    "duplicate_id_value_count": duplicate_ids["duplicate_id_value_count"],
                },
            )
        )

    age_quality, age_issues = compute_age_quality(normalized_df, annotations)
    issues.extend(age_issues)

    date_quality, date_issues = compute_date_quality(
        normalized_df,
        profile_df,
        survival_config,
        annotations,
    )
    issues.extend(date_issues)

    survival_quality = compute_survival_quality(
        normalized_df,
        survival_config,
        survival_ready_df,
    )
    issues.extend(survival_quality.get("issues", []))

    group_quality = None
    if survival_ready_df is not None and "_group" in survival_ready_df.columns:
        group_quality, group_issues = compute_group_quality(survival_ready_df)
        issues.extend(group_issues)

    sensitive_column_candidates = detect_sensitive_column_candidates(normalized_df, profile_df)
    if not sensitive_column_candidates.empty:
        issues.append(
            QualityIssue(
                severity="warning",
                category="sensitive_columns",
                message="Possible identifier or sensitive columns detected.",
                affected_columns=sensitive_column_candidates["column_name"].tolist(),
                affected_rows_count=None,
                details={"candidate_count": len(sensitive_column_candidates)},
            )
        )

    return {
        "overview": overview,
        "issues": issues,
        "missingness_by_column": missingness_by_column,
        "missingness_by_row": missingness_by_row,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "age_quality": age_quality,
        "date_quality": date_quality,
        "survival_quality": survival_quality,
        "survival_exclusion_breakdown": survival_quality.get(
            "survival_exclusion_breakdown",
            _empty_survival_exclusion_breakdown(),
        ),
        "group_quality": group_quality,
        "sensitive_column_candidates": sensitive_column_candidates,
    }


def compute_dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    n_rows = len(df)
    n_columns = len(df.columns)
    total_cells = n_rows * n_columns
    missing_cells = int(df.isna().sum().sum())
    complete_rows = int(df.notna().all(axis=1).sum()) if n_columns else n_rows

    return {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percent": round((missing_cells / total_cells * 100) if total_cells else 0.0, 2),
        "complete_rows": complete_rows,
        "complete_rows_percent": round((complete_rows / n_rows * 100) if n_rows else 0.0, 2),
    }


def compute_missingness_by_column(df: pd.DataFrame) -> pd.DataFrame:
    row_count = len(df)
    rows = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        rows.append(
            {
                "column_name": column,
                "missing_count": missing_count,
                "missing_percent": round((missing_count / row_count * 100) if row_count else 0.0, 2),
                "non_missing_count": int(row_count - missing_count),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["column_name", "missing_count", "missing_percent", "non_missing_count"],
    ).sort_values(["missing_percent", "missing_count"], ascending=[False, False], ignore_index=True)


def compute_missingness_by_row(df: pd.DataFrame) -> pd.DataFrame:
    column_count = len(df.columns)
    missing_count = df.isna().sum(axis=1)
    missing_rows = pd.DataFrame(
        {
            "row_index": df.index,
            "missing_count": missing_count.astype(int),
            "missing_percent": [
                round((count / column_count * 100) if column_count else 0.0, 2)
                for count in missing_count
            ],
        }
    )
    missing_rows = missing_rows[missing_rows["missing_count"] > 0]
    return missing_rows.sort_values("missing_count", ascending=False, ignore_index=True)


def detect_high_missingness(
    missingness_by_column: pd.DataFrame,
    warning_threshold: float = 20.0,
    severe_threshold: float = 50.0,
) -> list[QualityIssue]:
    issues = []

    for _, row in missingness_by_column.iterrows():
        missing_percent = float(row["missing_percent"])
        if missing_percent >= severe_threshold:
            message = f"Column {row['column_name']} has very high missingness."
        elif missing_percent >= warning_threshold:
            message = f"Column {row['column_name']} has moderate/high missingness."
        else:
            continue

        issues.append(
            QualityIssue(
                severity="warning",
                category="missingness",
                message=message,
                affected_columns=[str(row["column_name"])],
                affected_rows_count=int(row["missing_count"]),
                details={"missing_percent": missing_percent},
            )
        )

    return issues


def check_duplicate_rows(df: pd.DataFrame, max_examples: int = 10) -> dict[str, Any]:
    duplicated_mask = df.duplicated(keep=False)
    duplicate_row_count = int(duplicated_mask.sum())
    duplicate_group_count = int(df.loc[duplicated_mask].drop_duplicates().shape[0]) if duplicate_row_count else 0

    return {
        "duplicate_row_count": duplicate_row_count,
        "duplicate_row_percent": round((duplicate_row_count / len(df) * 100) if len(df) else 0.0, 2),
        "duplicate_group_count": duplicate_group_count,
        "example_duplicate_indices": df.index[duplicated_mask].tolist()[:max_examples],
    }


def check_duplicate_patient_ids(
    df: pd.DataFrame,
    id_col: str | None,
    max_examples: int = 10,
) -> dict[str, Any]:
    if not id_col or id_col not in df.columns:
        return {
            "checked": False,
            "id_col": id_col if id_col in df.columns else None,
            "duplicate_id_row_count": None,
            "duplicate_id_value_count": None,
            "duplicate_id_percent": None,
            "duplicate_id_examples": [],
        }

    id_values = df[id_col].dropna()
    duplicate_mask = id_values.duplicated(keep=False)
    duplicate_values = id_values[duplicate_mask]

    return {
        "checked": True,
        "id_col": id_col,
        "duplicate_id_row_count": int(duplicate_mask.sum()),
        "duplicate_id_value_count": int(duplicate_values.nunique(dropna=True)),
        "duplicate_id_percent": round((int(duplicate_mask.sum()) / len(df) * 100) if len(df) else 0.0, 2),
        "duplicate_id_examples": [str(value) for value in pd.unique(duplicate_values)[:max_examples]],
    }


def compute_age_quality(
    df: pd.DataFrame,
    annotations: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[QualityIssue]]:
    age_columns = [
        str(column)
        for column in df.columns
        if _annotation_meaning(annotations, str(column)) == "Age"
        or "age" in _column_tokens(str(column))
    ]
    invalid_rows = pd.Series(False, index=df.index)
    details = []

    for column in age_columns:
        values = df[column]
        numeric = pd.to_numeric(values, errors="coerce")
        non_numeric = values.notna() & numeric.isna()
        below_zero = numeric < 0
        above_120 = numeric > 120
        invalid = non_numeric | below_zero | above_120
        invalid_rows |= invalid.fillna(False)
        details.append(
            {
                "column_name": column,
                "invalid_count": int(invalid.sum()),
                "below_zero_count": int(below_zero.sum()),
                "above_120_count": int(above_120.sum()),
                "non_numeric_count": int(non_numeric.sum()),
            }
        )

    invalid_count = int(invalid_rows.sum())
    issues = []
    if invalid_count:
        issues.append(
            QualityIssue(
                severity="warning",
                category="age",
                message="Age values must be numeric and between 0 and 120.",
                affected_columns=age_columns,
                affected_rows_count=invalid_count,
            )
        )

    return {
        "checked": bool(age_columns),
        "age_columns": age_columns,
        "invalid_age_count": invalid_count if age_columns else None,
        "details": pd.DataFrame(
            details,
            columns=[
                "column_name",
                "invalid_count",
                "below_zero_count",
                "above_120_count",
                "non_numeric_count",
            ],
        ),
    }, issues


def compute_date_quality(
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None = None,
    survival_config=None,
    annotations: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[QualityIssue]]:
    date_columns = _date_columns(df, profile_df, survival_config, annotations)
    parsed_dates = {
        column: pd.to_datetime(df[column], errors="coerce", format="mixed", utc=True)
        for column in date_columns
    }
    parsing_rows = []
    issues: list[QualityIssue] = []

    for column in date_columns:
        invalid = df[column].notna() & parsed_dates[column].isna()
        invalid_count = int(invalid.sum())
        parsing_rows.append(
            {
                "column_name": column,
                "non_missing_count": int(df[column].notna().sum()),
                "invalid_count": invalid_count,
                "invalid_percent": round(
                    (invalid_count / int(df[column].notna().sum()) * 100)
                    if df[column].notna().any()
                    else 0.0,
                    2,
                ),
            }
        )
        if invalid_count:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="date_parsing",
                    message=f"Date column '{column}' contains unparseable values.",
                    affected_columns=[column],
                    affected_rows_count=invalid_count,
                )
            )

    start_col, event_col, followup_col = _date_role_columns(
        date_columns,
        survival_config,
        annotations,
    )
    consistency_rows = []
    for label, earlier_col, later_col in [
        ("Event date before diagnosis/start date", start_col, event_col),
        ("Last follow-up before start date", start_col, followup_col),
        ("Event date after last follow-up date", event_col, followup_col),
    ]:
        if not earlier_col or not later_col:
            continue
        inconsistent = parsed_dates[later_col] < parsed_dates[earlier_col]
        affected_count = int(inconsistent.sum())
        consistency_rows.append(
            {
                "check": label,
                "earlier_column": earlier_col,
                "later_column": later_col,
                "affected_rows_count": affected_count,
            }
        )
        if affected_count:
            issues.append(
                QualityIssue(
                    severity="error",
                    category="date_consistency",
                    message=f"{label} found.",
                    affected_columns=[earlier_col, later_col],
                    affected_rows_count=affected_count,
                )
            )

    parsing = pd.DataFrame(
        parsing_rows,
        columns=[
            "column_name",
            "non_missing_count",
            "invalid_count",
            "invalid_percent",
        ],
    )
    return {
        "checked": bool(date_columns),
        "date_columns": date_columns,
        "invalid_date_count": int(parsing["invalid_count"].sum()) if not parsing.empty else None,
        "parsing": parsing,
        "consistency": pd.DataFrame(
            consistency_rows,
            columns=[
                "check",
                "earlier_column",
                "later_column",
                "affected_rows_count",
            ],
        ),
    }, issues


def compute_survival_quality(
    df: pd.DataFrame,
    survival_config,
    survival_ready_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if survival_config is None:
        return {
            "has_mapping": False,
            "issues": [],
            "survival_exclusion_breakdown": _empty_survival_exclusion_breakdown(),
        }

    normalized_df = normalize_missing_values(df)
    if getattr(survival_config, "time_source", "duration") == "dates":
        derived = derive_survival_from_dates(normalized_df, survival_config)
        quality_df = normalized_df.assign(
            _derived_time=derived["_time"],
            _derived_event=derived["_event"],
        )
        quality = compute_survival_quality(
            quality_df,
            replace(
                survival_config,
                time_source="duration",
                time_col="_derived_time",
                event_col="_derived_event",
                event_values=[1],
                censor_values=[0],
                missing_event_handling="exclude",
                unmapped_event_handling="exclude",
            ),
            survival_ready_df,
        )
        quality["time_col"] = "Derived from dates"
        quality["event_col"] = survival_config.event_date_col
        quality["event_missing_count"] = int(
            normalized_df[survival_config.event_date_col].isna().sum()
        )
        quality["missing_event_handling"] = survival_config.missing_event_handling
        for issue in quality["issues"]:
            if "_derived_time" in issue.affected_columns:
                issue.affected_columns = [
                    survival_config.start_date_col,
                    survival_config.event_date_col,
                    survival_config.last_followup_date_col,
                ]
                issue.message = issue.message.replace(
                    "Survival time column",
                    "Date-derived survival time",
                )
            if "_derived_event" in issue.affected_columns:
                issue.affected_columns = [survival_config.event_date_col]
                issue.message = issue.message.replace("Event column", "Event date")
        return quality

    issues: list[QualityIssue] = []
    event_values = list(getattr(survival_config, "event_values", []))
    censor_values = list(getattr(survival_config, "censor_values", []))
    event_keys = {_canonical_value(value) for value in event_values}
    censor_keys = {_canonical_value(value) for value in censor_values}
    mapped_event_keys = event_keys | censor_keys
    missing_event_handling = getattr(
        survival_config,
        "missing_event_handling",
        "exclude",
    )
    unmapped_event_handling = getattr(
        survival_config,
        "unmapped_event_handling",
        "exclude",
    )

    time_col = getattr(survival_config, "time_col", None)
    event_col = getattr(survival_config, "event_col", None)
    raw_rows = len(normalized_df)

    result: dict[str, Any] = {
        "has_mapping": True,
        "time_col": time_col,
        "event_col": event_col,
        "raw_rows": raw_rows,
        "usable_survival_rows": 0,
        "excluded_rows": raw_rows,
        "events": 0,
        "censored": 0,
        "event_rate": 0.0,
        "time_missing_count": None,
        "event_missing_count": None,
        "time_non_numeric_count": None,
        "negative_time_count": None,
        "infinite_time_count": None,
        "zero_time_count": None,
        "unmapped_event_value_count": None,
        "unmapped_event_values": [],
        "missing_event_handling": missing_event_handling,
        "unmapped_event_handling": unmapped_event_handling,
    }

    if time_col not in normalized_df.columns:
        issues.append(
            QualityIssue(
                severity="error",
                category="survival_time",
                message=f"Mapped survival time column '{time_col}' is missing from the dataset.",
                affected_columns=[str(time_col)],
            )
        )

    if event_col not in normalized_df.columns:
        issues.append(
            QualityIssue(
                severity="error",
                category="survival_event",
                message=f"Mapped event column '{event_col}' is missing from the dataset.",
                affected_columns=[str(event_col)],
            )
        )

    if not event_values:
        issues.append(
            QualityIssue(
                severity="error",
                category="survival_event",
                message="No event values are selected.",
                affected_columns=[str(event_col)],
            )
        )

    if event_keys & censor_keys:
        issues.append(
            QualityIssue(
                severity="error",
                category="survival_event",
                message="Event and censor value mappings overlap.",
                affected_columns=[str(event_col)],
            )
        )

    if missing_event_handling not in ALLOWED_MISSING_EVENT_HANDLING:
        issues.append(
            QualityIssue(
                severity="error",
                category="survival_event",
                message="Missing event handling is invalid.",
                affected_columns=[str(event_col)],
            )
        )

    if unmapped_event_handling not in ALLOWED_UNMAPPED_EVENT_HANDLING:
        issues.append(
            QualityIssue(
                severity="error",
                category="survival_event",
                message="Unmapped event handling is invalid.",
                affected_columns=[str(event_col)],
            )
        )

    exclusion_masks: dict[str, pd.Series] = {}

    if time_col in normalized_df.columns:
        time_series = normalized_df[time_col]
        parsed_time = pd.to_numeric(time_series, errors="coerce")
        time_missing_mask = time_series.isna()
        time_non_numeric_mask = time_series.notna() & parsed_time.isna()
        negative_time_mask = parsed_time < 0
        infinite_time_mask = parsed_time.notna() & ~np.isfinite(parsed_time)
        zero_time_mask = parsed_time == 0

        result["time_missing_count"] = int(time_missing_mask.sum())
        result["time_non_numeric_count"] = int(time_non_numeric_mask.sum())
        result["negative_time_count"] = int(negative_time_mask.sum())
        result["infinite_time_count"] = int(infinite_time_mask.sum())
        result["zero_time_count"] = int(zero_time_mask.sum())

        if result["time_missing_count"] > 0:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="survival_time",
                    message="Survival time column has missing values.",
                    affected_columns=[time_col],
                    affected_rows_count=result["time_missing_count"],
                )
            )

        if result["time_non_numeric_count"] > 0:
            issues.append(
                QualityIssue(
                    severity="error",
                    category="survival_time",
                    message="Survival time column contains non-numeric values.",
                    affected_columns=[time_col],
                    affected_rows_count=result["time_non_numeric_count"],
                )
            )

        if result["negative_time_count"] > 0:
            issues.append(
                QualityIssue(
                    severity="error",
                    category="survival_time",
                    message="Survival time column contains negative values.",
                    affected_columns=[time_col],
                    affected_rows_count=result["negative_time_count"],
                )
            )

        if result["infinite_time_count"] > 0:
            issues.append(
                QualityIssue(
                    severity="error",
                    category="survival_time",
                    message="Survival time column contains infinite values.",
                    affected_columns=[time_col],
                    affected_rows_count=result["infinite_time_count"],
                )
            )

        if result["zero_time_count"] > 0:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="survival_time",
                    message="Survival time column contains zero values.",
                    affected_columns=[time_col],
                    affected_rows_count=result["zero_time_count"],
                )
            )

        exclusion_masks["missing time"] = time_missing_mask
        exclusion_masks["non-numeric time"] = time_non_numeric_mask
        exclusion_masks["negative time"] = negative_time_mask.fillna(False)
        exclusion_masks["infinite time"] = infinite_time_mask.fillna(False)

    if event_col in normalized_df.columns:
        event_series = normalized_df[event_col]
        event_missing_mask = event_series.isna()
        event_key_series = event_series.map(_canonical_value)
        unmapped_event_mask = event_series.notna() & ~event_key_series.isin(mapped_event_keys)
        unmapped_event_values = [
            str(value)
            for value in pd.unique(event_series[unmapped_event_mask])
        ]
        binary_event_series = create_binary_event_series(
            normalized_df,
            survival_config,
        )

        result["event_missing_count"] = int(event_missing_mask.sum())
        result["unmapped_event_value_count"] = int(unmapped_event_mask.sum())
        result["unmapped_event_values"] = unmapped_event_values

        if result["event_missing_count"] > 0:
            missing_action = (
                "treated as censored"
                if missing_event_handling == "treat_as_censored"
                else "excluded"
            )
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="survival_event",
                    message=f"Event column has missing values; those rows are {missing_action}.",
                    affected_columns=[event_col],
                    affected_rows_count=result["event_missing_count"],
                    details={"missing_event_handling": missing_event_handling},
                )
            )

        if result["unmapped_event_value_count"] > 0:
            unmapped_action = {
                "exclude": "excluded",
                "treat_as_censored": "treated as censored",
                "treat_as_event": "treated as events",
            }.get(unmapped_event_handling, "handled by an invalid policy")
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="survival_event",
                    message=(
                        "Event column contains unmapped non-missing values; "
                        f"those rows are {unmapped_action}."
                    ),
                    affected_columns=[event_col],
                    affected_rows_count=result["unmapped_event_value_count"],
                    details={
                        "unmapped_event_values": unmapped_event_values,
                        "unmapped_event_handling": unmapped_event_handling,
                    },
                )
            )
        if missing_event_handling != "treat_as_censored":
            exclusion_masks["missing event"] = event_missing_mask
        if unmapped_event_handling == "exclude":
            exclusion_masks["unmapped event value"] = unmapped_event_mask

    exclusion_union_mask = _union_exclusion_masks(exclusion_masks, normalized_df.index)
    result["excluded_rows"] = int(exclusion_union_mask.sum())
    result["usable_survival_rows"] = max(raw_rows - result["excluded_rows"], 0)

    if event_col in normalized_df.columns:
        usable_binary_events = binary_event_series.loc[~exclusion_union_mask].dropna()
        result["events"] = int((usable_binary_events == 1).sum())
        result["censored"] = int((usable_binary_events == 0).sum())
        result["event_rate"] = round(
            (result["events"] / (result["events"] + result["censored"]) * 100)
            if (result["events"] + result["censored"])
            else 0.0,
            2,
        )

        if result["events"] == 0:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="survival_event",
                    message="No event rows are available after event mapping.",
                    affected_columns=[event_col],
                )
            )

        if result["censored"] == 0:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="survival_event",
                    message="No censored rows are available after event mapping.",
                    affected_columns=[event_col],
                )
            )

    result["survival_exclusion_breakdown"] = _build_survival_exclusion_breakdown(
        exclusion_masks,
        raw_rows,
    )
    result["issues"] = issues
    return result


def compute_group_quality(
    survival_ready_df: pd.DataFrame,
    min_group_size: int = 5,
    max_groups: int = 8,
) -> tuple[pd.DataFrame | None, list[QualityIssue]]:
    if "_group" not in survival_ready_df.columns:
        return None, []

    group_df = survival_ready_df.dropna(subset=["_group"]).copy()
    if group_df.empty:
        return _empty_group_quality(), [
            QualityIssue(
                severity="warning",
                category="grouping",
                message="Grouping column has no non-missing values.",
                affected_columns=["_group"],
            )
        ]

    rows = []
    typed_keys = group_df["_group"].map(
        lambda value: (
            f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"
        )
    )
    group_entries = []
    for key in sorted(typed_keys.unique()):
        subset = group_df.loc[typed_keys.eq(key)]
        group_entries.append((subset["_group"].iloc[0], subset))
    duplicate_displays = {
        str(group_value)
        for group_value, _ in group_entries
        if sum(
            str(other_value) == str(group_value)
            for other_value, _ in group_entries
        )
        > 1
    }
    for group_value, subset in group_entries:
        n = len(subset)
        events = int((subset["_event"] == 1).sum())
        censored = int((subset["_event"] == 0).sum())
        rows.append(
            {
                "group": (
                    f"{group_value} ({type(group_value).__name__})"
                    if str(group_value) in duplicate_displays
                    else str(group_value)
                ),
                "n": n,
                "events": events,
                "censored": censored,
                "event_rate": round((events / n * 100) if n else 0.0, 2),
                "median_time": round(float(pd.to_numeric(subset["_time"], errors="coerce").median()), 2),
            }
        )

    quality_df = pd.DataFrame(
        rows,
        columns=["group", "n", "events", "censored", "event_rate", "median_time"],
    )
    issues = []
    group_count = len(quality_df)

    if group_count > max_groups:
        issues.append(
            QualityIssue(
                severity="warning",
                category="grouping",
                message="Selected grouping column has too many groups for a readable survival plot.",
                affected_columns=["_group"],
                details={"group_count": group_count},
            )
        )

    if (quality_df["n"] < min_group_size).any():
        issues.append(
            QualityIssue(
                severity="warning",
                category="grouping",
                message=f"One or more groups have fewer than {min_group_size} rows.",
                affected_columns=["_group"],
                details={"small_groups": quality_df.loc[quality_df["n"] < min_group_size, "group"].tolist()},
            )
        )

    if (quality_df["events"] == 0).any():
        issues.append(
            QualityIssue(
                severity="warning",
                category="grouping",
                message="One or more groups have no observed events.",
                affected_columns=["_group"],
                details={"groups": quality_df.loc[quality_df["events"] == 0, "group"].tolist()},
            )
        )

    if (quality_df["censored"] == 0).any():
        issues.append(
            QualityIssue(
                severity="warning",
                category="grouping",
                message="One or more groups have no censored observations.",
                affected_columns=["_group"],
                details={"groups": quality_df.loc[quality_df["censored"] == 0, "group"].tolist()},
            )
        )

    return quality_df, issues


def detect_sensitive_column_candidates(
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    profile_lookup = {}
    if profile_df is not None and "column_name" in profile_df.columns:
        profile_lookup = {
            str(row["column_name"]): row.to_dict()
            for _, row in profile_df.iterrows()
        }

    for column in df.columns:
        column_name = str(column)
        normalized_name = _normalize_name(column_name)
        profile_row = profile_lookup.get(column_name, {})
        reasons: list[tuple[str, str]] = []

        high_hint = _matching_hint(normalized_name, HIGH_RISK_NAME_HINTS)
        medium_hint = _matching_hint(normalized_name, MEDIUM_RISK_NAME_HINTS)
        common_id_hint = _matching_hint(normalized_name, COMMON_DEIDENTIFIED_ID_HINTS)

        if high_hint:
            reasons.append(("high", f"Column name contains '{high_hint}'"))
        elif medium_hint:
            reasons.append(("medium", f"Column name contains '{medium_hint}'"))
        elif common_id_hint or bool(profile_row.get("is_id_like")):
            reasons.append(("low", "Identifier-like column; verify that values are de-identified."))

        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            sampled = df[column].dropna().astype(str).head(200)
            if not sampled.empty:
                if sampled.str.contains(r"[\w.+-]+@[\w-]+\.[\w.-]+", regex=True).any():
                    reasons.append(("high", "Column contains email-like values."))

                phone_like_rate = sampled.str.contains(r"(?:\+?\d[\d\s().-]{7,}\d)", regex=True).mean()
                if phone_like_rate >= 0.3:
                    reasons.append(("high", "Column contains phone-like values."))

                if sampled.str.len().mean() > 80:
                    reasons.append(("medium", "Column contains long free-text values."))

        if reasons:
            risk_level = _max_risk_level([risk for risk, _ in reasons])
            rows.append(
                {
                    "column_name": column_name,
                    "reason": "; ".join(reason for _, reason in reasons),
                    "risk_level": risk_level,
                    "example_values": _example_values(df[column]),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["column_name", "reason", "risk_level", "example_values"],
    )


def determine_quality_status(issues: list[QualityIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "error"

    if any(issue.severity == "warning" for issue in issues):
        return "warning"

    return "success"


def issues_to_dataframe(issues: list[QualityIssue]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "severity": issue.severity,
                "category": issue.category,
                "message": issue.message,
                "affected_columns": ", ".join(issue.affected_columns),
                "affected_rows_count": issue.affected_rows_count,
            }
            for issue in issues
        ],
        columns=["severity", "category", "message", "affected_columns", "affected_rows_count"],
    )


def _build_survival_exclusion_breakdown(
    exclusion_masks: dict[str, pd.Series],
    row_count: int,
) -> pd.DataFrame:
    rows = []

    for reason, mask in exclusion_masks.items():
        row_total = int(mask.fillna(False).sum())
        if row_total == 0:
            continue

        rows.append(
            {
                "reason": reason,
                "row_count": row_total,
                "percent_of_dataset": round((row_total / row_count * 100) if row_count else 0.0, 2),
            }
        )

    return pd.DataFrame(rows, columns=["reason", "row_count", "percent_of_dataset"])


def _union_exclusion_masks(exclusion_masks: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    if not exclusion_masks:
        return pd.Series(False, index=index)

    union_mask = pd.Series(False, index=index)
    for mask in exclusion_masks.values():
        union_mask = union_mask | mask.reindex(index, fill_value=False).fillna(False)

    return union_mask


def _empty_survival_exclusion_breakdown() -> pd.DataFrame:
    return pd.DataFrame(columns=["reason", "row_count", "percent_of_dataset"])


def _empty_group_quality() -> pd.DataFrame:
    return pd.DataFrame(columns=["group", "n", "events", "censored", "event_rate", "median_time"])


def _date_columns(
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None,
    survival_config,
    annotations: Mapping[str, Any] | None,
) -> list[str]:
    candidates = set()
    if profile_df is not None and {"column_name", "detected_type"}.issubset(profile_df.columns):
        candidates.update(
            profile_df.loc[profile_df["detected_type"] == "date", "column_name"].astype(str)
        )

    for column in df.columns:
        column_name = str(column)
        if (
            _annotation_meaning(annotations, column_name) in {"Date", "Start time", "End time"}
            or _column_tokens(column_name) & {"date", "dob"}
        ):
            candidates.add(column_name)

    if survival_config is not None:
        candidates.update(
            column
            for column in [
                getattr(survival_config, "start_date_col", None),
                getattr(survival_config, "event_date_col", None),
                getattr(survival_config, "last_followup_date_col", None),
            ]
            if column
        )

    return [str(column) for column in df.columns if str(column) in candidates]


def _date_role_columns(
    date_columns: list[str],
    survival_config,
    annotations: Mapping[str, Any] | None,
) -> tuple[str | None, str | None, str | None]:
    configured = (
        getattr(survival_config, "start_date_col", None),
        getattr(survival_config, "event_date_col", None),
        getattr(survival_config, "last_followup_date_col", None),
    )
    if all(column in date_columns for column in configured if column) and any(configured):
        return configured

    start_col = _first_role_column(
        date_columns,
        annotations,
        "Start time",
        {"start", "diagnosis", "diagnostic", "enrollment", "index"},
    )
    event_col = _first_role_column(
        date_columns,
        annotations,
        None,
        {"event", "death", "relapse", "progression"},
    )
    followup_col = _first_role_column(
        date_columns,
        annotations,
        "End time",
        {"followup", "follow", "contact", "seen", "end"},
    )
    return start_col, event_col, followup_col


def _first_role_column(
    columns: list[str],
    annotations: Mapping[str, Any] | None,
    meaning: str | None,
    name_markers: set[str],
) -> str | None:
    if meaning:
        for column in columns:
            if _annotation_meaning(annotations, column) == meaning:
                return column
    return next(
        (column for column in columns if _column_tokens(column) & name_markers),
        None,
    )


def _annotation_meaning(
    annotations: Mapping[str, Any] | None,
    column: str,
) -> str | None:
    if not isinstance(annotations, Mapping) or column not in annotations:
        return None
    annotation = annotations[column]
    if hasattr(annotation, "resolved_meaning"):
        return str(annotation.resolved_meaning)
    if isinstance(annotation, Mapping):
        return str(annotation.get("meaning") or "")
    return None


def _column_tokens(column: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", column.lower()) if token}


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"

    if isinstance(value, str):
        return value.strip().lower()

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, (int, float)) and not pd.isna(value):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return str(numeric_value)

    return str(value).strip().lower()


def _normalize_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _matching_hint(column_name: str, hints: list[str]) -> str | None:
    for hint in hints:
        normalized_hint = _normalize_name(hint)
        if normalized_hint in column_name:
            return hint
    return None


def _max_risk_level(risk_levels: list[str]) -> str:
    ranking = {"low": 1, "medium": 2, "high": 3}
    return max(risk_levels, key=lambda risk_level: ranking[risk_level])


def _example_values(series: pd.Series, max_examples: int = 5) -> str:
    examples = []
    seen = set()
    for value in series.dropna():
        formatted = str(value)
        if formatted in seen:
            continue

        seen.add(formatted)
        examples.append(formatted)
        if len(examples) == max_examples:
            break

    return ", ".join(examples)
