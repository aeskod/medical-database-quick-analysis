from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

import pandas as pd


MEANING_OPTIONS = [
    "Ignore column",
    "Patient ID",
    "Follow-up time / survival time",
    "Event status",
    "Start time",
    "End time",
    "Age",
    "Sex / gender",
    "Diagnosis",
    "Treatment / exposure group",
    "Disease stage",
    "Risk category",
    "Comorbidity / precondition",
    "Medication",
    "Lab value / numeric measurement",
    "Biomarker",
    "Genetic marker",
    "Procedure / surgery",
    "Outcome other than survival",
    "Date",
    "Site / hospital / center",
    "Race / ethnicity",
    "Smoking / lifestyle factor",
    "Other categorical variable",
    "Other numeric variable",
    "Notes / free text",
    "Custom...",
]

USE_FILTER = "filter"
USE_GROUP = "group"
USE_BASELINE = "baseline"
USE_COX = "cox"
USE_CHARTS = "charts"
USE_IGNORE = "ignore"
USE_KEYS = {
    USE_FILTER,
    USE_GROUP,
    USE_BASELINE,
    USE_COX,
    USE_CHARTS,
    USE_IGNORE,
}

USE_COLUMN_LABELS = {
    USE_FILTER: "Use as filter",
    USE_GROUP: "Use as group",
    USE_BASELINE: "Use in baseline table",
    USE_COX: "Use as Cox covariate",
    USE_CHARTS: "Use in charts",
    USE_IGNORE: "Ignore in analysis",
}


@dataclass(frozen=True)
class ColumnAnnotation:
    column_name: str
    meaning: str
    uses: frozenset[str] = frozenset()
    custom_meaning: str = ""

    @property
    def resolved_meaning(self) -> str:
        if self.meaning == "Custom...":
            return self.custom_meaning.strip() or "Custom"
        return self.meaning

    def is_used_as(self, use: str) -> bool:
        if use == USE_IGNORE:
            return USE_IGNORE in self.uses
        return use in self.uses and USE_IGNORE not in self.uses


def build_default_annotations(
    df: pd.DataFrame,
    profile_df: pd.DataFrame,
    survival_config: Any = None,
) -> dict[str, ColumnAnnotation]:
    profile_lookup = _profile_lookup(profile_df)
    annotations: dict[str, ColumnAnnotation] = {}

    for raw_column in df.columns:
        column = str(raw_column)
        profile_row = profile_lookup.get(column, {})
        meaning = _infer_meaning(column, profile_row)
        uses = _default_uses(meaning, profile_row)
        annotations[column] = ColumnAnnotation(
            column_name=column,
            meaning=meaning,
            uses=frozenset(uses),
        )

    return apply_survival_roles(annotations, survival_config)


def sync_annotations(
    existing: Mapping[str, Any] | None,
    df: pd.DataFrame,
    profile_df: pd.DataFrame,
    survival_config: Any = None,
) -> dict[str, ColumnAnnotation]:
    defaults = build_default_annotations(df, profile_df, survival_config)
    if not isinstance(existing, Mapping):
        return defaults

    synchronized: dict[str, ColumnAnnotation] = {}
    for column in defaults:
        existing_annotation = _coerce_annotation(column, existing.get(column))
        synchronized[column] = existing_annotation or defaults[column]

    return apply_survival_roles(
        synchronized,
        survival_config,
        seed_analysis_uses=False,
    )


def apply_survival_roles(
    annotations: Mapping[str, ColumnAnnotation],
    survival_config: Any,
    seed_analysis_uses: bool = True,
) -> dict[str, ColumnAnnotation]:
    result = dict(annotations)
    if survival_config is None:
        return result

    protected_meanings = {
        getattr(survival_config, "time_col", None): "Follow-up time / survival time",
        getattr(survival_config, "event_col", None): "Event status",
        getattr(survival_config, "start_date_col", None): "Start time",
        getattr(survival_config, "event_date_col", None): "Date",
        getattr(survival_config, "last_followup_date_col", None): "End time",
        getattr(survival_config, "id_col", None): "Patient ID",
    }
    for column, meaning in protected_meanings.items():
        if column is None or column not in result:
            continue
        current = result[column]
        uses = set(current.uses)
        if seed_analysis_uses:
            uses.discard(USE_IGNORE)
            if meaning == "Patient ID":
                uses = {USE_IGNORE}
            elif meaning in {"Follow-up time / survival time", "Event status"}:
                uses.discard(USE_BASELINE)
                uses.discard(USE_COX)
                uses.add(USE_CHARTS)
        result[column] = ColumnAnnotation(column, meaning, frozenset(uses))

    group_col = getattr(survival_config, "group_col", None)
    if seed_analysis_uses and group_col is not None and group_col in result:
        current = result[group_col]
        uses = set(current.uses)
        uses.discard(USE_IGNORE)
        uses.add(USE_GROUP)
        result[group_col] = ColumnAnnotation(
            current.column_name,
            current.meaning,
            frozenset(uses),
            current.custom_meaning,
        )

    return result


def annotations_to_dataframe(
    annotations: Mapping[str, ColumnAnnotation],
    profile_df: pd.DataFrame,
) -> pd.DataFrame:
    profile_lookup = _profile_lookup(profile_df)
    rows = []

    for column, annotation in annotations.items():
        profile_row = profile_lookup.get(column, {})
        uses = set(annotation.uses)
        rows.append(
            {
                "Column": column,
                "Type": profile_row.get("detected_type", "unknown"),
                "Missing %": float(profile_row.get("missing_percent") or 0),
                "Example values": profile_row.get("example_values", ""),
                "Meaning": annotation.meaning,
                "Custom meaning": annotation.custom_meaning,
                **{
                    label: use in uses
                    for use, label in USE_COLUMN_LABELS.items()
                },
            }
        )

    return pd.DataFrame(rows)


def annotations_from_dataframe(
    editor_df: pd.DataFrame,
    valid_columns: Iterable[str],
) -> dict[str, ColumnAnnotation]:
    expected_columns = [str(column) for column in valid_columns]
    if "Column" not in editor_df.columns:
        raise ValueError("Annotation table is missing the Column field.")

    edited_columns = editor_df["Column"].astype(str).tolist()
    if len(edited_columns) != len(set(edited_columns)):
        raise ValueError("Each dataset column must appear exactly once in the annotation table.")
    if set(edited_columns) != set(expected_columns):
        raise ValueError("Annotation table columns do not match the uploaded dataset.")

    annotations: dict[str, ColumnAnnotation] = {}
    for _, row in editor_df.iterrows():
        column = str(row["Column"])
        meaning = str(row.get("Meaning", "Ignore column"))
        if meaning not in MEANING_OPTIONS:
            raise ValueError(f"Unsupported meaning for '{column}': {meaning}")

        custom_meaning = _clean_text(row.get("Custom meaning", ""))
        if meaning == "Custom..." and not custom_meaning:
            raise ValueError(f"Enter a custom meaning for '{column}'.")

        uses = {
            use
            for use, label in USE_COLUMN_LABELS.items()
            if _as_bool(row.get(label, False))
        }
        if USE_IGNORE in uses or meaning == "Ignore column":
            uses = {USE_IGNORE}

        annotations[column] = ColumnAnnotation(
            column_name=column,
            meaning=meaning,
            uses=frozenset(uses),
            custom_meaning=custom_meaning,
        )

    return annotations


def get_columns_for_use(
    annotations: Mapping[str, Any] | None,
    use: str,
    valid_columns: Iterable[str] | None = None,
) -> list[str]:
    if use not in USE_KEYS:
        raise ValueError(f"Unsupported annotation use: {use}")
    if not isinstance(annotations, Mapping):
        return []

    valid = None if valid_columns is None else {str(column) for column in valid_columns}
    selected = []
    for column, value in annotations.items():
        column_name = str(column)
        if valid is not None and column_name not in valid:
            continue
        annotation = _coerce_annotation(column_name, value)
        if annotation is not None and annotation.is_used_as(use):
            selected.append(column_name)
    return selected


def get_annotation_summary(
    annotations: Mapping[str, Any] | None,
) -> dict[str, int]:
    return {
        use: len(get_columns_for_use(annotations, use))
        for use in [USE_FILTER, USE_GROUP, USE_BASELINE, USE_COX, USE_CHARTS, USE_IGNORE]
    }


def _profile_lookup(profile_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if profile_df is None or profile_df.empty:
        return {}
    return {
        str(row["column_name"]): row.to_dict()
        for _, row in profile_df.iterrows()
    }


def _infer_meaning(column: str, profile_row: Mapping[str, Any]) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
    tokens = set(normalized.split("_"))
    detected_type = str(profile_row.get("detected_type") or "")

    keyword_meanings = [
        ({"age"}, "Age"),
        ({"sex", "gender"}, "Sex / gender"),
        ({"diagnosis", "diagnostic", "dx", "disease", "cancer"}, "Diagnosis"),
        ({"treatment", "therapy", "exposure", "arm", "drug"}, "Treatment / exposure group"),
        ({"stage", "staging"}, "Disease stage"),
        ({"risk"}, "Risk category"),
        ({"comorbidity", "comorbid", "precondition"}, "Comorbidity / precondition"),
        ({"medication", "medicine", "med"}, "Medication"),
        ({"biomarker"}, "Biomarker"),
        ({"gene", "genetic", "mutation", "genotype"}, "Genetic marker"),
        ({"procedure", "surgery", "operation"}, "Procedure / surgery"),
        ({"outcome", "response"}, "Outcome other than survival"),
        ({"site", "hospital", "center", "centre", "institution", "inst"}, "Site / hospital / center"),
        ({"race", "ethnicity", "ethnic"}, "Race / ethnicity"),
        ({"smoking", "smoker", "lifestyle"}, "Smoking / lifestyle factor"),
        ({"note", "notes", "comment", "comments"}, "Notes / free text"),
    ]
    for keywords, meaning in keyword_meanings:
        if tokens & keywords:
            return meaning

    if bool(profile_row.get("is_id_like")):
        return "Patient ID"
    if detected_type == "date" or "date" in tokens:
        return "Date"
    if detected_type in {"integer", "float"} and not bool(profile_row.get("is_binary_like")):
        return "Other numeric variable"
    if detected_type in {"binary", "boolean", "categorical"} or bool(
        profile_row.get("is_low_cardinality")
    ):
        return "Other categorical variable"
    if detected_type == "text":
        return "Notes / free text"
    return "Ignore column"


def _default_uses(meaning: str, profile_row: Mapping[str, Any]) -> set[str]:
    if meaning in {"Ignore column", "Patient ID", "Notes / free text"}:
        return {USE_IGNORE}

    if meaning == "Date":
        return {USE_FILTER, USE_CHARTS}

    categorical = meaning in {
        "Sex / gender",
        "Diagnosis",
        "Treatment / exposure group",
        "Disease stage",
        "Risk category",
        "Comorbidity / precondition",
        "Medication",
        "Biomarker",
        "Genetic marker",
        "Procedure / surgery",
        "Outcome other than survival",
        "Site / hospital / center",
        "Race / ethnicity",
        "Smoking / lifestyle factor",
        "Other categorical variable",
    } or bool(profile_row.get("is_binary_like"))

    uses = {USE_FILTER, USE_BASELINE, USE_COX, USE_CHARTS}
    if categorical:
        uses.add(USE_GROUP)
    return uses


def _coerce_annotation(column: str, value: Any) -> ColumnAnnotation | None:
    if isinstance(value, ColumnAnnotation):
        return value if value.column_name == column else ColumnAnnotation(
            column,
            value.meaning,
            value.uses,
            value.custom_meaning,
        )
    if not isinstance(value, Mapping):
        return None

    meaning = str(value.get("meaning", "Ignore column"))
    raw_uses = value.get("uses", [])
    uses = frozenset(str(use) for use in raw_uses if str(use) in USE_KEYS)
    return ColumnAnnotation(
        column_name=column,
        meaning=meaning if meaning in MEANING_OPTIONS else "Ignore column",
        uses=uses,
        custom_meaning=_clean_text(value.get("custom_meaning", "")),
    )


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)
