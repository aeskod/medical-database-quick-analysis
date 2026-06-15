from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from src.profiling import normalize_missing_values


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
    issues: list[QualityIssue] = []
    event_values = list(getattr(survival_config, "event_values", []))
    censor_values = list(getattr(survival_config, "censor_values", []))
    event_keys = {_canonical_value(value) for value in event_values}
    censor_keys = {_canonical_value(value) for value in censor_values}
    mapped_event_keys = event_keys | censor_keys

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
        "zero_time_count": None,
        "unmapped_event_value_count": None,
        "unmapped_event_values": [],
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

    exclusion_masks: dict[str, pd.Series] = {}

    if time_col in normalized_df.columns:
        time_series = normalized_df[time_col]
        parsed_time = pd.to_numeric(time_series, errors="coerce")
        time_missing_mask = time_series.isna()
        time_non_numeric_mask = time_series.notna() & parsed_time.isna()
        negative_time_mask = parsed_time < 0
        zero_time_mask = parsed_time == 0

        result["time_missing_count"] = int(time_missing_mask.sum())
        result["time_non_numeric_count"] = int(time_non_numeric_mask.sum())
        result["negative_time_count"] = int(negative_time_mask.sum())
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

    if event_col in normalized_df.columns:
        event_series = normalized_df[event_col]
        event_missing_mask = event_series.isna()
        event_key_series = event_series.map(_canonical_value)
        unmapped_event_mask = event_series.notna() & ~event_key_series.isin(mapped_event_keys)
        event_mask = event_key_series.isin(event_keys)
        censor_mask = event_key_series.isin(censor_keys)
        unmapped_event_values = [
            str(value)
            for value in pd.unique(event_series[unmapped_event_mask])
        ]

        result["event_missing_count"] = int(event_missing_mask.sum())
        result["unmapped_event_value_count"] = int(unmapped_event_mask.sum())
        result["unmapped_event_values"] = unmapped_event_values
        result["events"] = int(event_mask.sum())
        result["censored"] = int(censor_mask.sum())
        result["event_rate"] = round(
            (result["events"] / (result["events"] + result["censored"]) * 100)
            if (result["events"] + result["censored"])
            else 0.0,
            2,
        )

        if result["event_missing_count"] > 0:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="survival_event",
                    message="Event column has missing values.",
                    affected_columns=[event_col],
                    affected_rows_count=result["event_missing_count"],
                )
            )

        if result["unmapped_event_value_count"] > 0:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="survival_event",
                    message="Event column contains unmapped non-missing values.",
                    affected_columns=[event_col],
                    affected_rows_count=result["unmapped_event_value_count"],
                    details={"unmapped_event_values": unmapped_event_values},
                )
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

        exclusion_masks["missing event"] = event_missing_mask
        exclusion_masks["unmapped event value"] = unmapped_event_mask

    exclusion_union_mask = _union_exclusion_masks(exclusion_masks, normalized_df.index)
    result["excluded_rows"] = int(exclusion_union_mask.sum())
    result["usable_survival_rows"] = max(raw_rows - result["excluded_rows"], 0)

    if event_col in normalized_df.columns:
        usable_event_series = normalized_df.loc[~exclusion_union_mask, event_col]
        usable_event_keys = usable_event_series.map(_canonical_value)
        result["events"] = int(usable_event_keys.isin(event_keys).sum())
        result["censored"] = int(usable_event_keys.isin(censor_keys).sum())
        result["event_rate"] = round(
            (result["events"] / (result["events"] + result["censored"]) * 100)
            if (result["events"] + result["censored"])
            else 0.0,
            2,
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
    for group_value, subset in group_df.groupby("_group", sort=True, dropna=True):
        n = len(subset)
        events = int((subset["_event"] == 1).sum())
        censored = int((subset["_event"] == 0).sum())
        rows.append(
            {
                "group": str(group_value),
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
