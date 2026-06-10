import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


MISSING_VALUE_TOKENS = {
    "",
    " ",
    "NA",
    "N/A",
    "na",
    "n/a",
    "NULL",
    "null",
    "None",
    "none",
    "missing",
    "Missing",
}

BOOLEAN_TOKENS = {"true", "false", "yes", "no", "y", "n", "0", "1"}
ID_NAME_HINTS = ("id", "patient_id", "subject_id", "record_id", "case_id", "mrn")


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where common string missing-value markers are converted to NA."""
    normalized = df.copy(deep=True)

    for column in normalized.columns:
        normalized[column] = normalized[column].map(_normalize_value)

    return normalized


def infer_column_type(series: pd.Series) -> str:
    """Infer a simple, deterministic column type label."""
    normalized = normalize_missing_values(series.to_frame(name=series.name)).iloc[:, 0]
    non_missing = normalized.dropna()

    if non_missing.empty:
        return "empty"

    non_missing_count = len(non_missing)
    unique_count = int(non_missing.nunique(dropna=True))
    unique_ratio = unique_count / non_missing_count
    string_values = non_missing.map(str)
    stripped_values = string_values.str.strip()
    lower_values = stripped_values.str.lower()
    column_name = str(series.name or "").lower()

    has_id_hint = _has_id_name_hint(column_name)
    mostly_unique = unique_ratio >= 0.95 and non_missing_count > 10

    if has_id_hint and mostly_unique:
        return "id_like"

    if set(lower_values).issubset(BOOLEAN_TOKENS):
        return "boolean"

    numeric_values = pd.to_numeric(non_missing, errors="coerce")
    numeric_parse_count = int(numeric_values.notna().sum())
    all_numeric = numeric_parse_count == non_missing_count

    if all_numeric:
        if _all_numeric_values_are_integer(numeric_values):
            return "integer"
        return "float"

    date_values = pd.to_datetime(non_missing, errors="coerce", format="mixed")
    date_parse_ratio = float(date_values.notna().sum()) / non_missing_count
    if date_parse_ratio >= 0.8:
        return "date"

    if 0 < numeric_parse_count < non_missing_count:
        return "mixed"

    average_string_length = float(stripped_values.str.len().mean())

    if average_string_length > 30:
        return "text"

    if mostly_unique and _values_look_identifier_like(stripped_values):
        return "id_like"

    if unique_count <= 20 or unique_ratio <= 0.2:
        return "categorical"

    if unique_ratio > 0.5:
        return "text"

    return "mixed"


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return one profiling row per DataFrame column."""
    normalized = normalize_missing_values(df)
    row_count = len(normalized)
    profile_rows = []

    for column in normalized.columns:
        series = normalized[column]
        missing_count = int(series.isna().sum())
        non_missing = series.dropna()
        non_missing_count = int(len(non_missing))
        unique_count = int(non_missing.nunique(dropna=True))
        missing_percent = round((missing_count / row_count * 100) if row_count else 0.0, 2)

        profile_rows.append(
            {
                "column_name": column,
                "detected_type": infer_column_type(series),
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "non_missing_count": non_missing_count,
                "unique_count": unique_count,
                "example_values": _format_example_values(non_missing),
            }
        )

    return pd.DataFrame(
        profile_rows,
        columns=[
            "column_name",
            "detected_type",
            "missing_count",
            "missing_percent",
            "non_missing_count",
            "unique_count",
            "example_values",
        ],
    )


def _normalize_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA

    if isinstance(value, str):
        stripped = value.strip()
        if value in MISSING_VALUE_TOKENS or stripped in MISSING_VALUE_TOKENS:
            return pd.NA

    return value


def _all_numeric_values_are_integer(values: pd.Series) -> bool:
    numeric_values = values.astype(float)
    return bool(np.isclose(numeric_values % 1, 0).all())


def _has_id_name_hint(column_name: str) -> bool:
    tokens = re.split(r"[^a-z0-9]+", column_name)
    token_set = {token for token in tokens if token}
    return any(hint in token_set or hint in column_name for hint in ID_NAME_HINTS)


def _values_look_identifier_like(values: Iterable[object]) -> bool:
    identifier_pattern = re.compile(r"^[A-Za-z0-9_.:-]+$")
    string_values = [str(value).strip() for value in values]

    if not string_values:
        return False

    identifier_like_count = sum(bool(identifier_pattern.match(value)) for value in string_values)
    average_length = sum(len(value) for value in string_values) / len(string_values)

    return identifier_like_count / len(string_values) >= 0.95 and average_length <= 40


def _format_example_values(series: pd.Series) -> str:
    examples = []
    seen = set()

    for value in series:
        key = _hashable_value(value)
        if key in seen:
            continue

        seen.add(key)
        examples.append(str(value))

        if len(examples) == 5:
            break

    return ", ".join(examples)


def _hashable_value(value: object) -> object:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value
