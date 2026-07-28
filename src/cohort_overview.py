import re
from numbers import Integral, Real
from typing import Any, Mapping

import pandas as pd

from src.profiling import normalize_missing_values, profile_dataframe
from src.survival_analysis import get_survival_summary


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
    "amount",
    "measure",
    "measurement",
    "dose",
    "count",
]
TEXT_LENGTH_THRESHOLD = 80
COHORT_MEANINGS = {
    "Patient ID": "patient_id",
    "Age": "age",
    "Sex / gender": "sex",
    "Diagnosis": "diagnosis",
    "Treatment / exposure group": "treatment",
    "Outcome other than survival": "outcome",
}


def compute_cohort_overview_metrics(
    df: pd.DataFrame,
    survival_ready_df: pd.DataFrame | None = None,
    time_unit: str = "unknown",
    id_col: str | None = None,
    age_col: str | None = None,
) -> dict[str, Any]:
    source_df = normalize_missing_values(df)
    n_rows = len(source_df)
    n_columns = len(source_df.columns)
    complete_rows = int(source_df.notna().all(axis=1).sum()) if n_columns else n_rows
    total_cells = n_rows * n_columns
    missing_cells = int(source_df.isna().sum().sum()) if total_cells else 0
    has_patient_id = id_col is not None and id_col in source_df.columns
    patient_df = _patient_level_dataframe(source_df, id_col)
    n_patients = (
        int(source_df[id_col].nunique(dropna=True))
        if has_patient_id
        else n_rows
    )
    missing_patient_ids = int(source_df[id_col].isna().sum()) if has_patient_id else None
    age = (
        pd.to_numeric(patient_df[age_col], errors="coerce").dropna()
        if age_col is not None and age_col in patient_df.columns
        else pd.Series(dtype=float)
    )

    metrics: dict[str, Any] = {
        "n_patients": n_patients,
        "patient_count_basis": f"Distinct {id_col}" if has_patient_id else "Dataset rows",
        "missing_patient_ids": missing_patient_ids,
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
        "median_age": round(float(age.median()), 2) if not age.empty else None,
        "age_col": age_col if age_col in source_df.columns else None,
        "time_unit": time_unit,
    }

    if survival_ready_df is None:
        return metrics

    survival_df = survival_ready_df.copy(deep=True)
    if has_patient_id and "_id" in survival_df.columns:
        survival_df = _patient_level_dataframe(survival_df, "_id")
    survival_df["_time"] = pd.to_numeric(survival_df["_time"], errors="coerce")
    survival_df["_event"] = pd.to_numeric(survival_df["_event"], errors="coerce")
    survival_df = survival_df.dropna(subset=["_time", "_event"])
    survival_summary = get_survival_summary(survival_df, time_unit)
    usable_rows = survival_summary["n"]
    events = survival_summary["events"]
    censored = survival_summary["censored"]

    metrics.update(
        {
            "usable_survival_rows": usable_rows,
            "events": events,
            "censored": censored,
            "event_rate": round((events / usable_rows * 100) if usable_rows else 0.0, 2),
            "median_followup": survival_summary["median_followup"],
            "max_followup": survival_summary["max_followup"],
        }
    )
    return metrics


def get_cohort_role_columns(
    df: pd.DataFrame,
    annotations: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    roles = {role: [] for role in COHORT_MEANINGS.values()}
    if not isinstance(annotations, Mapping):
        return roles

    valid_columns = {str(column) for column in df.columns}
    for column, annotation in annotations.items():
        column_name = str(column)
        if column_name not in valid_columns:
            continue
        meaning = getattr(annotation, "resolved_meaning", None)
        if meaning is None and isinstance(annotation, Mapping):
            meaning = annotation.get("meaning")
        role = COHORT_MEANINGS.get(str(meaning))
        if role is not None:
            roles[role].append(column_name)
    return roles


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
        if detected_type == "float" or unique_count > 10 or has_continuous_name_hint:
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
    level_plan = _categorical_level_plan(non_missing, max_levels)
    level_counts = _counts_for_categorical_plan(non_missing, level_plan)
    denominator = total_count if include_missing else len(non_missing)
    rows = [
        {
            "variable": column,
            "level": level,
            "count": int(count),
            "percent": round((int(count) / denominator * 100) if denominator else 0.0, 2),
            "summary": _format_count_percent(
                int(count),
                (int(count) / denominator * 100) if denominator else 0.0,
            ),
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
    survival_ready_df: pd.DataFrame | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    normalized = normalize_missing_values(df)
    analysis_df = _patient_level_dataframe(normalized, id_col)
    continuous_vars = [variable for variable in continuous_vars if variable != group_col]
    categorical_vars = [variable for variable in categorical_vars if variable != group_col]
    group_values = _group_values(analysis_df, group_col)
    group_value_labels = group_value_labels or {}
    group_columns = _unique_group_display_labels(group_values, group_value_labels)
    rows = _baseline_summary_rows(
        analysis_df,
        group_col,
        group_values,
        group_columns,
        survival_ready_df,
        id_col,
    )

    for variable in continuous_vars:
        if variable not in analysis_df.columns:
            continue

        row = {
            "Variable": variable,
            "Overall": summarize_continuous_variable(analysis_df, variable)["summary"],
        }
        if group_col:
            for group_value in group_values:
                group_df = analysis_df[
                    analysis_df[group_col].map(_typed_value_key)
                    == _typed_value_key(group_value)
                ]
                row[group_columns[_typed_value_key(group_value)]] = (
                    summarize_continuous_variable(group_df, variable)["summary"]
                )
        rows.append(row)

    for variable in categorical_vars:
        if variable not in analysis_df.columns:
            continue

        normalized_variable = normalize_missing_values(
            analysis_df[[variable]]
        )[variable]
        level_plan = _categorical_level_plan(
            normalized_variable.dropna(),
            max_levels,
        )
        overall_summary = _summarize_categorical_with_plan(
            normalized_variable,
            variable,
            level_plan,
            include_missing,
        )

        for _, summary_row in overall_summary.iterrows():
            level = str(summary_row["level"])
            table_row = {
                "Variable": f"{variable} = {level}",
                "Overall": summary_row["summary"],
            }
            for group_value in group_values:
                group_df = analysis_df[
                    analysis_df[group_col].map(_typed_value_key)
                    == _typed_value_key(group_value)
                ]
                group_summary = _summarize_categorical_with_plan(
                    normalize_missing_values(group_df[[variable]])[variable],
                    variable,
                    level_plan,
                    include_missing,
                    include_zero_levels=True,
                )
                matching = group_summary[group_summary["level"].astype(str) == level]
                table_row[group_columns[_typed_value_key(group_value)]] = (
                    matching["summary"].iloc[0] if not matching.empty else "0 (0.00%)"
                )
            rows.append(table_row)

    columns = ["Variable", "Overall"] + [
        group_columns[_typed_value_key(group_value)] for group_value in group_values
    ]
    return pd.DataFrame(rows, columns=columns)


def _baseline_summary_rows(
    df: pd.DataFrame,
    group_col: str | None,
    group_values: list[Any],
    group_columns: dict[str, str],
    survival_ready_df: pd.DataFrame | None,
    id_col: str | None,
) -> list[dict[str, str]]:
    count_row = {"Variable": "n", "Overall": str(_patient_count(df, id_col))}
    for group_value in group_values:
        group_df = df[
            df[group_col].map(_typed_value_key) == _typed_value_key(group_value)
        ]
        count_row[group_columns[_typed_value_key(group_value)]] = str(
            _patient_count(group_df, id_col)
        )

    event_row = {"Variable": "Events, n (%)", "Overall": "Not available"}
    followup_row = {
        "Variable": "Observed duration, median [IQR]",
        "Overall": "Not available",
    }
    if survival_ready_df is not None:
        survival_df = survival_ready_df.copy(deep=True)
        if id_col is not None and "_id" in survival_df.columns:
            survival_df = _patient_level_dataframe(survival_df, "_id")
        survival_df["_time"] = pd.to_numeric(survival_df["_time"], errors="coerce")
        survival_df["_event"] = pd.to_numeric(survival_df["_event"], errors="coerce")
        survival_df = survival_df.dropna(subset=["_time", "_event"])
        event_row["Overall"] = _event_summary(survival_df)
        followup_row["Overall"] = _followup_summary(survival_df)

        for group_value in group_values:
            column = group_columns[_typed_value_key(group_value)]
            if "_group" not in survival_df.columns:
                event_row[column] = "Not available"
                followup_row[column] = "Not available"
                continue
            group_survival_df = survival_df[
                survival_df["_group"].map(_typed_value_key)
                == _typed_value_key(group_value)
            ]
            event_row[column] = _event_summary(group_survival_df)
            followup_row[column] = _followup_summary(group_survival_df)
    else:
        for group_value in group_values:
            column = group_columns[_typed_value_key(group_value)]
            event_row[column] = "Not available"
            followup_row[column] = "Not available"

    return [count_row, event_row, followup_row]


def _patient_count(df: pd.DataFrame, id_col: str | None) -> int:
    if id_col is not None and id_col in df.columns:
        return int(df[id_col].nunique(dropna=True))
    return len(df)


def _event_summary(survival_df: pd.DataFrame) -> str:
    total = len(survival_df)
    if not total:
        return "Not available"
    events = int((survival_df["_event"] == 1).sum())
    return _format_count_percent(events, events / total * 100)


def _followup_summary(survival_df: pd.DataFrame) -> str:
    followup = survival_df["_time"].dropna()
    if followup.empty:
        return "Not available"
    return (
        f"{_format_number(followup.median())} "
        f"[{_format_number(followup.quantile(0.25))}, "
        f"{_format_number(followup.quantile(0.75))}]"
    )


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


def _categorical_level_plan(
    series: pd.Series,
    max_levels: int,
) -> dict[str, Any]:
    values_by_key: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for value in series.dropna():
        key = _typed_value_key(value)
        values_by_key.setdefault(key, value)
        counts[key] = counts.get(key, 0) + 1

    ordered_keys = sorted(
        counts,
        key=lambda key: (-counts[key], str(values_by_key[key]), key),
    )
    top_keys = ordered_keys[: max(1, int(max_levels))]
    collapsed_keys = set(ordered_keys[len(top_keys) :])
    labels: dict[str, str] = {}
    used_labels: set[str] = {"Missing"}
    order: list[str] = []
    for key in top_keys:
        raw_label = str(values_by_key[key])
        if raw_label == "Missing":
            raw_label = "Missing (value)"
        label = raw_label
        suffix = 2
        while label in used_labels:
            label = f"{raw_label} [{type(values_by_key[key]).__name__} #{suffix}]"
            suffix += 1
        labels[key] = label
        used_labels.add(label)
        order.append(label)

    collapsed_label = None
    if collapsed_keys:
        collapsed_label = "Other (collapsed)"
        suffix = 2
        while collapsed_label in used_labels:
            collapsed_label = f"Other (collapsed #{suffix})"
            suffix += 1
        order.append(collapsed_label)

    return {
        "labels": labels,
        "top_keys": set(top_keys),
        "collapsed_keys": collapsed_keys,
        "collapsed_label": collapsed_label,
        "order": order,
    }


def _counts_for_categorical_plan(
    series: pd.Series,
    plan: dict[str, Any],
) -> dict[str, int]:
    counts = {str(label): 0 for label in plan["order"]}
    for value in series.dropna():
        key = _typed_value_key(value)
        if key in plan["top_keys"]:
            label = plan["labels"][key]
        else:
            label = plan["collapsed_label"]
            if label is None:
                continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def _summarize_categorical_with_plan(
    series: pd.Series,
    variable: str,
    plan: dict[str, Any],
    include_missing: bool,
    *,
    include_zero_levels: bool = False,
) -> pd.DataFrame:
    total_count = len(series)
    non_missing = series.dropna()
    denominator = total_count if include_missing else len(non_missing)
    counts = _counts_for_categorical_plan(non_missing, plan)
    rows = []
    for level, count in counts.items():
        if count == 0 and not include_zero_levels:
            continue
        percent = count / denominator * 100 if denominator else 0.0
        rows.append(
            {
                "variable": variable,
                "level": level,
                "count": int(count),
                "percent": round(percent, 2),
                "summary": _format_count_percent(int(count), percent),
            }
        )

    missing_count = int(series.isna().sum())
    if include_missing and (missing_count > 0 or include_zero_levels):
        percent = missing_count / total_count * 100 if total_count else 0.0
        rows.append(
            {
                "variable": variable,
                "level": "Missing",
                "count": missing_count,
                "percent": round(percent, 2),
                "summary": _format_count_percent(missing_count, percent),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["variable", "level", "count", "percent", "summary"],
    )


def _patient_level_dataframe(
    df: pd.DataFrame,
    id_col: str | None,
) -> pd.DataFrame:
    if id_col is None or id_col not in df.columns:
        return df.copy(deep=True)
    identified = df.dropna(subset=[id_col]).copy()
    if identified.empty:
        return identified
    keys = identified[id_col].map(_typed_value_key)
    patient_rows = []
    for key in pd.unique(keys):
        patient_records = identified.loc[keys.eq(key)]
        row = {}
        for column in identified.columns:
            non_missing = patient_records[column].dropna()
            row[column] = non_missing.iloc[0] if not non_missing.empty else pd.NA
        patient_rows.append(row)
    return pd.DataFrame(patient_rows, columns=df.columns)


def _group_values(df: pd.DataFrame, group_col: str | None) -> list[Any]:
    if not group_col or group_col not in df.columns:
        return []

    values_by_key: dict[str, Any] = {}
    for value in df[group_col].dropna():
        values_by_key.setdefault(_typed_value_key(value), value)
    return [values_by_key[key] for key in sorted(values_by_key)]


def _display_group_value(value: Any, group_value_labels: dict[str, str]) -> str:
    value_string = str(value)
    typed_key = (
        f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"
    )
    return group_value_labels.get(
        typed_key,
        group_value_labels.get(value_string, value_string),
    )


def _unique_group_display_labels(
    values: list[Any],
    group_value_labels: dict[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for value in values:
        base = _display_group_value(value, group_value_labels)
        label = base
        suffix = 2
        while label in used or label in {"Variable", "Overall"}:
            label = f"{base} [{type(value).__name__} #{suffix}]"
            suffix += 1
        used.add(label)
        result[_typed_value_key(value)] = label
    return result


def _typed_value_key(value: Any) -> str:
    if isinstance(value, bool):
        return f"bool:{value}"
    if isinstance(value, Integral):
        return f"integer:{int(value)}"
    if isinstance(value, Real):
        if pd.isna(value):
            return "missing"
        if float(value).is_integer():
            return f"integer:{int(value)}"
        return f"number:{float(value)!r}"
    return f"{type(value).__name__}:{value!r}"


def _format_count_percent(count: int, percent: float) -> str:
    return f"{count} ({round(percent, 2):.2f}%)"


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Not available"

    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.2f}"
