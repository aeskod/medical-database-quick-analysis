import re
from typing import Any

import pandas as pd

from src.profiling import normalize_missing_values, profile_dataframe


ID_NAME_HINTS = ["id", "patient_id", "subject_id", "record_id", "case_id", "mrn"]
CATEGORICAL_NAME_HINTS = [
    "sex",
    "gender",
    "stage",
    "ecog",
    "karno",
    "status",
    "treatment",
    "arm",
    "group",
]
EXCLUDED_NAME_HINTS = ["inst", "institution", "site_id", "center_id", "hospital_id"]
CONTINUOUS_NAME_HINTS = [
    "age",
    "weight",
    "height",
    "bmi",
    "cal",
    "loss",
    "lab",
    "level",
    "score",
    "dose",
    "count",
]
TEXT_LENGTH_THRESHOLD = 80


def compute_cohort_overview_metrics(
    df: pd.DataFrame,
    survival_ready_df: pd.DataFrame | None = None,
    time_unit: str = "unknown",
) -> dict[str, Any]:
    source_df = normalize_missing_values(df)
    n_rows = len(source_df)
    n_columns = len(source_df.columns)
    complete_rows = int(source_df.notna().all(axis=1).sum()) if n_columns else n_rows
    total_cells = n_rows * n_columns
    missing_cells = int(source_df.isna().sum().sum()) if total_cells else 0

    metrics: dict[str, Any] = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "complete_rows": complete_rows,
        "complete_rows_percent": round((complete_rows / n_rows * 100) if n_rows else 0.0, 2),
        "missing_cells_percent": round((missing_cells / total_cells * 100) if total_cells else 0.0, 2),
        "usable_survival_rows": None,
        "events": None,
        "censored": None,
        "event_rate": None,
        "median_followup": None,
        "max_followup": None,
        "time_unit": time_unit,
    }

    if survival_ready_df is None:
        return metrics

    survival_df = survival_ready_df.copy(deep=True)
    survival_df["_time"] = pd.to_numeric(survival_df["_time"], errors="coerce")
    survival_df["_event"] = pd.to_numeric(survival_df["_event"], errors="coerce")
    survival_df = survival_df.dropna(subset=["_time", "_event"])
    usable_rows = len(survival_df)
    events = int((survival_df["_event"] == 1).sum())
    censored = int((survival_df["_event"] == 0).sum())

    metrics.update(
        {
            "usable_survival_rows": usable_rows,
            "events": events,
            "censored": censored,
            "event_rate": round((events / usable_rows * 100) if usable_rows else 0.0, 2),
            "median_followup": round(float(survival_df["_time"].median()), 2) if usable_rows else None,
            "max_followup": round(float(survival_df["_time"].max()), 2) if usable_rows else None,
        }
    )
    return metrics


def classify_summary_variable(
    column_name: str,
    series: pd.Series,
    profile_row: dict | None = None,
    survival_config=None,
) -> str:
    if survival_config is not None:
        if column_name in {getattr(survival_config, "time_col", None), getattr(survival_config, "event_col", None)}:
            return "excluded"

        if column_name == getattr(survival_config, "id_col", None):
            return "id"

        if column_name == getattr(survival_config, "group_col", None):
            return "categorical"

    if _has_excluded_name_hint(column_name):
        return "excluded"

    normalized = normalize_missing_values(series.to_frame(name=column_name))[column_name]
    profile_row = profile_row or _profile_row_for_series(column_name, normalized)
    unique_count = int(profile_row.get("unique_count") or 0)
    non_missing_count = int(profile_row.get("non_missing_count") or normalized.dropna().shape[0])
    unique_ratio = float(profile_row.get("unique_ratio") or 0)
    numeric_parse_rate = float(profile_row.get("numeric_parse_rate") or 0)
    date_parse_rate = float(profile_row.get("date_parse_rate") or 0)
    detected_type = str(profile_row.get("detected_type") or "")
    is_binary_like = bool(profile_row.get("is_binary_like"))
    is_id_like = bool(profile_row.get("is_id_like"))
    has_continuous_name_hint = _has_continuous_name_hint(column_name)

    if _looks_like_id_column(
        column_name,
        unique_ratio,
        non_missing_count,
        is_id_like,
        numeric_parse_rate,
        is_binary_like,
    ):
        return "id"

    if date_parse_rate >= 0.8 and numeric_parse_rate < 0.95:
        return "datetime"

    if _has_categorical_name_hint(column_name) and unique_count <= 30:
        return "categorical"

    if numeric_parse_rate >= 0.95 and not is_binary_like:
        if unique_count > 10 or has_continuous_name_hint:
            return "continuous"

    if is_binary_like or detected_type in {"boolean", "categorical"} or 2 <= unique_count <= 20:
        return "categorical"

    if detected_type == "text" or _looks_like_free_text(normalized):
        return "text"

    if numeric_parse_rate >= 0.95:
        return "continuous"

    return "text"


def get_default_baseline_variables(
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None = None,
    survival_config=None,
) -> dict[str, list[str]]:
    normalized_df = normalize_missing_values(df)
    profile_lookup = _profile_lookup(normalized_df, profile_df)
    result = {
        "continuous": [],
        "categorical": [],
        "excluded": [],
    }

    for column in normalized_df.columns:
        classification = classify_summary_variable(
            str(column),
            normalized_df[column],
            profile_lookup.get(str(column)),
            survival_config,
        )

        if classification == "continuous":
            result["continuous"].append(str(column))
        elif classification == "categorical":
            result["categorical"].append(str(column))
        else:
            result["excluded"].append(str(column))

    return result


def summarize_continuous_variable(df: pd.DataFrame, column: str) -> dict[str, Any]:
    series = pd.to_numeric(normalize_missing_values(df[[column]])[column], errors="coerce")
    non_missing = series.dropna()
    missing = int(len(df) - len(non_missing))

    if non_missing.empty:
        return {
            "variable": column,
            "n": 0,
            "missing": missing,
            "mean": None,
            "sd": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
            "summary": "Not available",
        }

    mean = round(float(non_missing.mean()), 2)
    sd_value = non_missing.std()
    sd = round(float(sd_value), 2) if not pd.isna(sd_value) else 0.0
    median = round(float(non_missing.median()), 2)
    q1 = round(float(non_missing.quantile(0.25)), 2)
    q3 = round(float(non_missing.quantile(0.75)), 2)
    min_value = round(float(non_missing.min()), 2)
    max_value = round(float(non_missing.max()), 2)

    return {
        "variable": column,
        "n": int(len(non_missing)),
        "missing": missing,
        "mean": mean,
        "sd": sd,
        "median": median,
        "q1": q1,
        "q3": q3,
        "min": min_value,
        "max": max_value,
        "summary": f"{_format_number(mean)} +/- {_format_number(sd)}; median {_format_number(median)} [{_format_number(q1)}, {_format_number(q3)}]",
    }


def summarize_categorical_variable(
    df: pd.DataFrame,
    column: str,
    max_levels: int = 10,
    include_missing: bool = True,
) -> pd.DataFrame:
    normalized = normalize_missing_values(df[[column]])[column]
    total_count = len(normalized)
    non_missing = normalized.dropna()
    non_missing_count = len(non_missing)
    level_counts = _collapsed_value_counts(non_missing, max_levels)
    rows = [
        {
            "variable": column,
            "level": str(level),
            "count": int(count),
            "percent": round((int(count) / non_missing_count * 100) if non_missing_count else 0.0, 2),
            "summary": _format_count_percent(int(count), (int(count) / non_missing_count * 100) if non_missing_count else 0.0),
        }
        for level, count in level_counts.items()
    ]

    missing_count = int(normalized.isna().sum())
    if include_missing and missing_count > 0:
        missing_percent = (missing_count / total_count * 100) if total_count else 0.0
        rows.append(
            {
                "variable": column,
                "level": "Missing",
                "count": missing_count,
                "percent": round(missing_percent, 2),
                "summary": _format_count_percent(missing_count, missing_percent),
            }
        )

    return pd.DataFrame(rows, columns=["variable", "level", "count", "percent", "summary"])


def summarize_continuous_by_group(
    df: pd.DataFrame,
    column: str,
    group_col: str,
) -> pd.DataFrame:
    normalized = normalize_missing_values(df[[column, group_col]])
    rows = []

    for group_value, group_df in normalized.dropna(subset=[group_col]).groupby(group_col, sort=True):
        summary = summarize_continuous_variable(group_df, column)
        summary["group"] = str(group_value)
        rows.append(summary)

    return pd.DataFrame(
        rows,
        columns=["group", "variable", "n", "missing", "mean", "sd", "median", "q1", "q3", "min", "max", "summary"],
    )


def summarize_categorical_by_group(
    df: pd.DataFrame,
    column: str,
    group_col: str,
    max_levels: int = 10,
    include_missing: bool = True,
) -> pd.DataFrame:
    normalized = normalize_missing_values(df[[column, group_col]])
    frames = []

    for group_value, group_df in normalized.dropna(subset=[group_col]).groupby(group_col, sort=True):
        summary = summarize_categorical_variable(
            group_df,
            column,
            max_levels=max_levels,
            include_missing=include_missing,
        )
        summary.insert(0, "group", str(group_value))
        frames.append(summary)

    if not frames:
        return pd.DataFrame(columns=["group", "variable", "level", "count", "percent", "summary"])

    return pd.concat(frames, ignore_index=True)


def build_baseline_table(
    df: pd.DataFrame,
    continuous_vars: list[str],
    categorical_vars: list[str],
    group_col: str | None = None,
    max_levels: int = 10,
    include_missing: bool = True,
    group_value_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    normalized = normalize_missing_values(df)
    group_values = _group_values(normalized, group_col)
    group_value_labels = group_value_labels or {}
    rows = []

    for variable in continuous_vars:
        if variable not in normalized.columns:
            continue

        row = {
            "Variable": variable,
            "Overall": summarize_continuous_variable(normalized, variable)["summary"],
        }
        if group_col:
            group_summary = summarize_continuous_by_group(normalized, variable, group_col)
            summary_by_group = dict(zip(group_summary["group"], group_summary["summary"]))
            for group_value in group_values:
                row[_display_group_value(group_value, group_value_labels)] = summary_by_group.get(
                    str(group_value),
                    "Not available",
                )
        rows.append(row)

    for variable in categorical_vars:
        if variable not in normalized.columns:
            continue

        overall_summary = summarize_categorical_variable(
            normalized,
            variable,
            max_levels=max_levels,
            include_missing=include_missing,
        )
        group_summary = (
            summarize_categorical_by_group(
                normalized,
                variable,
                group_col,
                max_levels=max_levels,
                include_missing=include_missing,
            )
            if group_col
            else pd.DataFrame(columns=["group", "variable", "level", "count", "percent", "summary"])
        )

        for _, summary_row in overall_summary.iterrows():
            level = str(summary_row["level"])
            table_row = {
                "Variable": f"{variable} = {level}",
                "Overall": summary_row["summary"],
            }
            for group_value in group_values:
                group_label = str(group_value)
                matching = group_summary[
                    (group_summary["group"] == group_label)
                    & (group_summary["variable"] == variable)
                    & (group_summary["level"].astype(str) == level)
                ]
                table_row[_display_group_value(group_value, group_value_labels)] = (
                    matching["summary"].iloc[0] if not matching.empty else "0 (0.00%)"
                )
            rows.append(table_row)

    columns = ["Variable", "Overall"] + [
        _display_group_value(group_value, group_value_labels)
        for group_value in group_values
    ]
    return pd.DataFrame(rows, columns=columns)


def count_variable_types(
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None = None,
    survival_config=None,
) -> pd.DataFrame:
    normalized = normalize_missing_values(df)
    profile_lookup = _profile_lookup(normalized, profile_df)
    counts: dict[str, int] = {
        "continuous": 0,
        "categorical": 0,
        "datetime": 0,
        "text": 0,
        "id": 0,
        "excluded": 0,
    }

    for column in normalized.columns:
        classification = classify_summary_variable(
            str(column),
            normalized[column],
            profile_lookup.get(str(column)),
            survival_config,
        )
        counts[classification] = counts.get(classification, 0) + 1

    return pd.DataFrame(
        [{"Variable type": variable_type, "Count": count} for variable_type, count in counts.items()]
    )


def _profile_lookup(df: pd.DataFrame, profile_df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if profile_df is None or profile_df.empty:
        profile_df = profile_dataframe(df)

    return {
        str(row["column_name"]): row.to_dict()
        for _, row in profile_df.iterrows()
    }


def _profile_row_for_series(column_name: str, series: pd.Series) -> dict[str, Any]:
    return profile_dataframe(series.to_frame(name=column_name)).iloc[0].to_dict()


def _looks_like_id_column(
    column_name: str,
    unique_ratio: float,
    non_missing_count: int,
    is_id_like: bool,
    numeric_parse_rate: float,
    is_binary_like: bool,
) -> bool:
    if _has_id_name_hint(column_name):
        return True

    if numeric_parse_rate >= 0.95 and not is_binary_like:
        return is_id_like

    return is_id_like or (unique_ratio >= 0.95 and non_missing_count > 10)


def _has_id_name_hint(column_name: str) -> bool:
    normalized_name = column_name.lower().replace("-", "_").replace(" ", "_")
    tokens = {token for token in re.split(r"[^a-z0-9]+", column_name.lower()) if token}
    return "id" in tokens or any(
        hint in normalized_name
        for hint in ID_NAME_HINTS
        if hint != "id"
    )


def _has_continuous_name_hint(column_name: str) -> bool:
    normalized_name = column_name.lower().replace("-", "_").replace(".", "_")
    return any(hint in normalized_name for hint in CONTINUOUS_NAME_HINTS)


def _has_categorical_name_hint(column_name: str) -> bool:
    normalized_name = column_name.lower().replace("-", "_").replace(".", "_")
    return any(hint in normalized_name for hint in CATEGORICAL_NAME_HINTS)


def _has_excluded_name_hint(column_name: str) -> bool:
    normalized_name = column_name.lower().replace("-", "_").replace(".", "_")
    tokens = {token for token in re.split(r"[^a-z0-9]+", column_name.lower()) if token}
    return "inst" in tokens or any(
        hint in normalized_name
        for hint in EXCLUDED_NAME_HINTS
        if hint != "inst"
    )


def _looks_like_free_text(series: pd.Series) -> bool:
    text_values = series.dropna().astype(str)
    if text_values.empty:
        return False

    unique_ratio = text_values.nunique(dropna=True) / len(text_values)
    return text_values.str.len().mean() > TEXT_LENGTH_THRESHOLD or unique_ratio > 0.8


def _collapsed_value_counts(series: pd.Series, max_levels: int) -> dict[str, int]:
    value_counts = series.astype(str).value_counts(sort=False)
    value_counts = value_counts.sort_index(kind="stable").sort_values(ascending=False, kind="stable")

    if len(value_counts) <= max_levels:
        return {str(level): int(count) for level, count in value_counts.items()}

    top_counts = value_counts.iloc[:max_levels]
    other_count = int(value_counts.iloc[max_levels:].sum())
    result = {str(level): int(count) for level, count in top_counts.items()}
    result["Other"] = other_count
    return result


def _group_values(df: pd.DataFrame, group_col: str | None) -> list[str]:
    if not group_col or group_col not in df.columns:
        return []

    values = df[group_col].dropna().astype(str).unique().tolist()
    return sorted(values)


def _display_group_value(value: Any, group_value_labels: dict[str, str]) -> str:
    value_string = str(value)
    return group_value_labels.get(value_string, value_string)


def _format_count_percent(count: int, percent: float) -> str:
    return f"{count} ({round(percent, 2):.2f}%)"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"

    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.2f}"
