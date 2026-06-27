import re
from typing import Any

import pandas as pd


TIME_KEYWORDS = [
    "time",
    "days",
    "day",
    "months",
    "month",
    "years",
    "year",
    "duration",
    "follow",
    "followup",
    "follow_up",
    "fu",
    "surv",
    "survival",
    "os",
    "pfs",
    "dfs",
    "rfs",
    "tte",
    "ttd",
    "time_to_event",
    "days_to_death",
    "months_to_death",
]

EVENT_KEYWORDS = [
    "status",
    "event",
    "death",
    "dead",
    "deceased",
    "vital",
    "mortality",
    "outcome",
    "censor",
    "censored",
    "relapse",
    "recurrence",
    "progression",
    "failure",
    "endpoint",
    "observed",
]

EVENT_LIKE_VALUES = [
    "0",
    "1",
    "true",
    "false",
    "yes",
    "no",
    "y",
    "n",
    "dead",
    "alive",
    "deceased",
    "living",
    "event",
    "censored",
    "relapsed",
    "progressed",
    "death",
    "no death",
]

ID_KEYWORDS = [
    "id",
    "patient",
    "subject",
    "record",
    "case",
    "sample",
    "participant",
    "person",
    "mrn",
]

GROUP_KEYWORDS = [
    "sex",
    "gender",
    "treatment",
    "arm",
    "group",
    "stage",
    "diagnosis",
    "dx",
    "type",
    "class",
    "risk",
    "site",
    "center",
    "mutation",
    "subtype",
    "therapy",
    "drug",
    "cohort",
    "ecog",
    "karno",
]


def confidence_label(score: int) -> str:
    if score >= 80:
        return "High"
    if score >= 50:
        return "Medium"
    if score >= 25:
        return "Low"
    return "Unlikely"


def score_time_candidate(column_name: str, profile_row: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons = []
    matched_keyword = _matched_keyword(column_name, TIME_KEYWORDS)

    if matched_keyword:
        score += 40
        reasons.append(f"Column name contains '{matched_keyword}'")

    if _numeric_parse_rate(profile_row) >= 0.95:
        score += 25
        reasons.append("Column is numeric")

    if bool(profile_row.get("is_non_negative")):
        score += 10
        reasons.append("Values are non-negative")

    if _unique_count(profile_row) > 10:
        score += 10
        reasons.append(f"Column has {_unique_count(profile_row)} unique values")

    if _missing_percent(profile_row) < 20:
        score += 5
        reasons.append(f"Column has {_missing_percent(profile_row)}% missing values")

    if bool(profile_row.get("is_binary_like")):
        score -= 40
        reasons.append("Column is binary-like")

    if bool(profile_row.get("is_id_like")):
        score -= 40
        reasons.append("Column appears ID-like")

    if profile_row.get("detected_type") in {"text", "categorical"} and _numeric_parse_rate(profile_row) < 0.8:
        score -= 20
        reasons.append("Column is text/categorical rather than numeric")

    min_value = profile_row.get("min_value")
    if min_value is not None and min_value < 0:
        score -= 20
        reasons.append("Column contains negative values")

    return {"score": _clamp_score(score), "reasons": reasons}


def score_event_candidate(column_name: str, profile_row: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons = []
    matched_keyword = _matched_keyword(column_name, EVENT_KEYWORDS)

    if matched_keyword:
        score += 40
        reasons.append(f"Column name contains '{matched_keyword}'")

    if bool(profile_row.get("is_binary_like")):
        score += 30
        reasons.append("Column is binary-like")

    if profile_row.get("detected_type") in {"binary", "boolean"}:
        score += 20
        reasons.append(f"Column is {profile_row.get('detected_type')}")

    unique_count = _unique_count(profile_row)
    if 2 < unique_count <= 6:
        score += 15
        reasons.append(f"Column has {unique_count} unique values")

    matched_values = _matched_event_like_values(profile_row.get("example_values", ""))
    if matched_values:
        score += 15
        reasons.append(f"Example values include event-like values: {', '.join(matched_values)}")

    if _missing_percent(profile_row) < 20:
        score += 5
        reasons.append(f"Column has {_missing_percent(profile_row)}% missing values")

    if bool(profile_row.get("is_id_like")):
        score -= 50
        reasons.append("Column appears ID-like")

    if unique_count > 10:
        score -= 40
        reasons.append(f"Column has {unique_count} unique values")

    if _numeric_parse_rate(profile_row) >= 0.95 and unique_count > 10:
        score -= 30
        reasons.append("Column looks like a continuous numeric measurement")

    if profile_row.get("detected_type") == "date":
        score -= 20
        reasons.append("Column is date-like")

    return {"score": _clamp_score(score), "reasons": reasons}


def score_id_candidate(column_name: str, profile_row: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons = []
    matched_keyword = _matched_keyword(column_name, ID_KEYWORDS)

    if matched_keyword:
        score += 40
        reasons.append(f"Column name contains '{matched_keyword}'")

    if _unique_ratio(profile_row) >= 0.95:
        score += 35
        reasons.append("Column values are mostly unique")

    if _missing_percent(profile_row) == 0:
        score += 10
        reasons.append("Column has no missing values")

    if profile_row.get("detected_type") in {"text", "id_like"}:
        score += 10
        reasons.append(f"Detected type is {profile_row.get('detected_type')}")

    if _unique_count(profile_row) <= 2:
        score -= 40
        reasons.append("Column has too few unique values for an ID")

    if _missing_percent(profile_row) > 20:
        score -= 20
        reasons.append(f"Column has {_missing_percent(profile_row)}% missing values")

    return {"score": _clamp_score(score), "reasons": reasons}


def score_group_candidate(column_name: str, profile_row: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons = []
    matched_keyword = _matched_keyword(column_name, GROUP_KEYWORDS)

    if matched_keyword:
        score += 30
        reasons.append(f"Column name contains '{matched_keyword}'")

    unique_count = _unique_count(profile_row)
    if 2 <= unique_count <= 10:
        score += 35
        reasons.append(f"Column has {unique_count} unique values")

    if 11 <= unique_count <= 20:
        score += 15
        reasons.append(f"Column has {unique_count} unique values")

    if _missing_percent(profile_row) < 20:
        score += 10
        reasons.append(f"Column has {_missing_percent(profile_row)}% missing values")

    if profile_row.get("detected_type") in {"binary", "categorical", "boolean"}:
        score += 10
        reasons.append(f"Detected type is {profile_row.get('detected_type')}")

    if bool(profile_row.get("is_low_cardinality")):
        score += 5
        reasons.append("Column is low-cardinality")

    if bool(profile_row.get("is_id_like")):
        score -= 50
        reasons.append("Column appears ID-like")

    if unique_count > 30:
        score -= 30
        reasons.append(f"Column has {unique_count} unique values")

    if profile_row.get("detected_type") == "text" and unique_count > 20:
        score -= 20
        reasons.append("Text column has too many unique values")

    return {"score": _clamp_score(score), "reasons": reasons}


def suggest_survival_roles(profile_df: pd.DataFrame, top_n: int = 5) -> dict[str, list[dict[str, Any]]]:
    return {
        "time_candidates": _rank_candidates(profile_df, score_time_candidate, top_n),
        "event_candidates": _rank_candidates(profile_df, score_event_candidate, top_n),
        "id_candidates": _rank_candidates(profile_df, score_id_candidate, top_n),
        "group_candidates": _rank_candidates(profile_df, score_group_candidate, top_n),
    }


def _rank_candidates(
    profile_df: pd.DataFrame,
    scoring_function,
    top_n: int,
) -> list[dict[str, Any]]:
    candidates = []

    for _, row in profile_df.iterrows():
        column_name = str(row["column_name"])
        score_result = scoring_function(column_name, row.to_dict())
        score = int(score_result["score"])

        if score < 25:
            continue

        candidates.append(
            {
                "column_name": column_name,
                "score": score,
                "confidence": confidence_label(score),
                "reasons": score_result["reasons"],
            }
        )

    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)[:top_n]


def _clamp_score(score: int) -> int:
    return max(0, min(100, int(score)))


def _matched_keyword(column_name: str, keywords: list[str]) -> str | None:
    normalized_name = _normalize_name(column_name)
    tokens = set(normalized_name.split("_"))

    for keyword in keywords:
        normalized_keyword = _normalize_name(keyword)
        keyword_tokens = normalized_keyword.split("_")
        compact_keyword = normalized_keyword.replace("_", "")
        compact_name = normalized_name.replace("_", "")

        if len(compact_keyword) <= 3:
            if normalized_keyword in tokens:
                return keyword
            continue

        if normalized_keyword in normalized_name or compact_keyword in compact_name:
            return keyword

        if all(token in tokens for token in keyword_tokens):
            return keyword

    return None


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _matched_event_like_values(example_values: object) -> list[str]:
    values = [_normalize_value(value) for value in str(example_values).split(",")]
    return sorted({value for value in values if value in EVENT_LIKE_VALUES})


def _normalize_value(value: str) -> str:
    return value.strip().lower()


def _numeric_parse_rate(profile_row: dict[str, Any]) -> float:
    return float(profile_row.get("numeric_parse_rate") or 0)


def _unique_count(profile_row: dict[str, Any]) -> int:
    return int(profile_row.get("unique_count") or 0)


def _unique_ratio(profile_row: dict[str, Any]) -> float:
    return float(profile_row.get("unique_ratio") or 0)


def _missing_percent(profile_row: dict[str, Any]) -> float:
    return float(profile_row.get("missing_percent") or 0)
