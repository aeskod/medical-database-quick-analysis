from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


CURVE_COLUMNS = ["time", "survival", "ci_lower", "ci_upper", "censored", "group"]
OVERALL_LABEL = "Overall"
GROUP_SUMMARY_COLUMNS = [
    "group",
    "raw_group",
    "n",
    "events",
    "censored",
    "event_rate",
    "median_followup",
    "max_followup",
    "median_survival",
    "time_unit",
]
OVERALL_SUMMARY_COLUMNS = [
    "group",
    "n",
    "events",
    "censored",
    "event_rate",
    "median_followup",
    "max_followup",
    "median_survival",
    "time_unit",
]
PAIRWISE_LOGRANK_COLUMNS = [
    "group_1",
    "group_2",
    "raw_group_1",
    "raw_group_2",
    "test_statistic",
    "p_value",
    "p_value_formatted",
    "p_value_holm",
    "p_value_holm_formatted",
]
AT_RISK_COLUMNS = ["group", "time", "at_risk", "events_up_to_time", "censored_up_to_time"]
SURVIVAL_PROBABILITY_COLUMNS = ["group", "time", "survival_probability"]


def validate_survival_ready_dataframe(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    has_time = "_time" in df.columns
    has_event = "_event" in df.columns

    if not has_time:
        errors.append("_time column is missing.")

    if not has_event:
        errors.append("_event column is missing.")

    if not has_time or not has_event:
        return errors, warnings

    time_series = df["_time"]
    event_series = df["_event"]
    parsed_time = pd.to_numeric(time_series, errors="coerce")
    parsed_event = pd.to_numeric(event_series, errors="coerce")

    time_non_missing = time_series.dropna()
    if not time_non_missing.empty and parsed_time.loc[time_non_missing.index].isna().any():
        errors.append("_time column must contain numeric values.")

    event_non_missing = event_series.dropna()
    parsed_event_non_missing = parsed_event.loc[event_non_missing.index]
    invalid_event_mask = parsed_event_non_missing.isna() | ~parsed_event_non_missing.isin([0, 1])
    if invalid_event_mask.any():
        errors.append("_event column must contain only 0 and 1 values.")

    parsed_non_missing_time = parsed_time.dropna()
    if not parsed_non_missing_time.empty and not np.isfinite(
        parsed_non_missing_time.astype(float)
    ).all():
        errors.append("_time column contains infinite values.")
    if not parsed_non_missing_time.empty and (parsed_non_missing_time < 0).any():
        errors.append("_time column contains negative values.")

    usable_df = _prepare_survival_dataframe(df)
    if usable_df.empty:
        errors.append("No usable rows exist after dropping missing _time/_event values.")

    if not parsed_non_missing_time.empty and (parsed_non_missing_time == 0).any():
        warnings.append("_time column contains zero values.")

    if not usable_df.empty:
        event_count = int((usable_df["_event"] == 1).sum())
        censored_count = int((usable_df["_event"] == 0).sum())

        if event_count == 0:
            warnings.append("_event has no events.")

        if censored_count == 0:
            warnings.append("_event has no censored rows.")

        if len(usable_df) < 10:
            warnings.append("Less than 10 usable rows are available.")

    return errors, warnings


def get_survival_summary(df: pd.DataFrame, time_unit: str = "unknown") -> dict[str, Any]:
    survival_df = _prepare_survival_dataframe(df)
    n = len(survival_df)
    events = int((survival_df["_event"] == 1).sum()) if n else 0
    censored = int((survival_df["_event"] == 0).sum()) if n else 0
    event_rate = round((events / n * 100) if n else 0.0, 2)
    median_followup = _reverse_km_median_followup(survival_df) if n else None
    max_followup = round(float(survival_df["_time"].max()), 2) if n else None

    return {
        "n": n,
        "events": events,
        "censored": censored,
        "event_rate": event_rate,
        "median_followup": median_followup,
        "max_followup": max_followup,
        "time_unit": time_unit,
    }


def fit_km_overall(df: pd.DataFrame, label: str = "Overall") -> dict[str, Any]:
    survival_df = _prepare_survival_dataframe(df)
    if survival_df.empty:
        raise ValueError("Kaplan-Meier estimation requires at least one usable row.")
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=survival_df["_time"],
        event_observed=survival_df["_event"],
        label=label,
    )

    return {
        "label": label,
        "kmf": kmf,
        "curve": _curve_dataframe_from_kmf(kmf, label),
        "median_survival": _to_python_float(kmf.median_survival_time_),
    }


def fit_km_by_group(
    df: pd.DataFrame,
    group_col: str = "_group",
    min_group_size: int = 5,
    max_groups: int = 8,
    group_value_labels: dict | None = None,
    original_group_col: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []

    if group_col not in df.columns:
        return [], [f"{group_col} column is missing; grouped KM analysis was skipped."]

    survival_df = _prepare_survival_dataframe(df)
    if survival_df.empty:
        return [], ["No usable rows exist for grouped KM analysis."]

    if survival_df[group_col].isna().any():
        warnings.append("Rows with missing group values were excluded from grouped analysis.")
        survival_df = survival_df.dropna(subset=[group_col]).copy()

    group_count = len(_iter_typed_groups(survival_df, group_col))
    if group_count == 0:
        return [], warnings + ["No non-missing group values are available for grouped analysis."]

    if group_count > max_groups:
        return [], warnings + ["Selected grouping column has too many groups for a readable KM plot."]

    group_results = []
    used_labels: set[str] = set()
    for group_value, group_df in _iter_typed_groups(survival_df, group_col):
        label = format_group_label(group_value, original_group_col, group_value_labels)
        label = _make_unique_group_label(label, group_value, used_labels)
        used_labels.add(label)
        if len(group_df) < min_group_size:
            warnings.append(f"Group {label} has fewer than {min_group_size} rows. Interpret cautiously.")

        group_results.append(fit_km_overall(group_df, label=label))

    return group_results, warnings


def suggest_timepoints(max_time: float, time_unit: str = "unknown") -> list[float]:
    if max_time is None or pd.isna(max_time) or max_time <= 0:
        return []

    fixed_timepoints = {
        "days": [30, 90, 180, 365.25, 730.5, 1095.75, 1826.25],
        "weeks": [4, 13, 26, 52, 104, 156, 260],
        "months": [1, 3, 6, 12, 24, 36, 60],
        "years": [1, 3, 5, 10],
    }

    if time_unit in fixed_timepoints:
        selected = [timepoint for timepoint in fixed_timepoints[time_unit] if timepoint <= max_time]
        if selected:
            return _normalize_timepoints(selected)

    return _normalize_timepoints(
        [float(max_time) * fraction for fraction in [0.25, 0.5, 0.75]]
    )


def survival_probability_at_times(kmf: KaplanMeierFitter, timepoints: list[float]) -> pd.DataFrame:
    normalized = _normalize_timepoints(timepoints)
    max_followup = float(kmf.timeline[-1])
    rows = []
    for timepoint in normalized:
        probability = (
            round(float(kmf.predict(timepoint)), 4)
            if timepoint <= max_followup
            else np.nan
        )
        rows.append(
            {"time": timepoint, "survival_probability": probability}
        )
    return pd.DataFrame(rows, columns=["time", "survival_probability"])


def survival_probabilities_at_years(
    kmf: KaplanMeierFitter,
    time_unit: str,
) -> dict[int, float | None]:
    units_per_year = {
        "days": 365.25,
        "weeks": 52.0,
        "months": 12.0,
        "years": 1.0,
    }
    if time_unit not in units_per_year:
        return {year: None for year in (1, 3, 5)}

    max_followup = float(kmf.timeline[-1])
    return {
        year: (
            round(float(kmf.predict(year * units_per_year[time_unit])), 4)
            if year * units_per_year[time_unit] <= max_followup
            else None
        )
        for year in (1, 3, 5)
    }


def format_group_label(
    raw_value,
    group_col: str | None = None,
    group_value_labels: dict | None = None,
) -> str:
    raw_label = str(raw_value)
    if group_col and isinstance(group_value_labels, dict):
        column_labels = group_value_labels.get(group_col, {})
        if isinstance(column_labels, dict):
            for lookup_key in [
                _typed_group_key(raw_value),
                raw_label,
                _group_label_key(raw_value),
            ]:
                if lookup_key in column_labels and str(column_labels[lookup_key]).strip():
                    return str(column_labels[lookup_key]).strip()

    if group_col:
        return f"{group_col} = {raw_label}"

    return raw_label


def format_survival_time(value, time_unit: str = "unknown") -> str:
    if value is None:
        return "Not reached"

    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value) or not np.isfinite(float(numeric_value)):
        return "Not reached"

    suffix = "" if time_unit == "unknown" else f" {time_unit}"
    return f"{float(numeric_value):.2f}{suffix}"


def format_p_value(p: float) -> str:
    numeric_p = pd.to_numeric(pd.Series([p]), errors="coerce").iloc[0]
    if pd.isna(numeric_p) or not np.isfinite(float(numeric_p)):
        return "p = N/A"
    if float(numeric_p) < 0.001:
        return "p < 0.001"
    return f"p = {float(numeric_p):.4f}"


def compute_overall_survival_summary_table(
    survival_ready_df: pd.DataFrame,
    time_unit: str = "unknown",
) -> pd.DataFrame:
    survival_df = _prepare_survival_dataframe(survival_ready_df)
    if survival_df.empty:
        return pd.DataFrame(columns=OVERALL_SUMMARY_COLUMNS)

    summary = get_survival_summary(survival_df, time_unit=time_unit)
    median_survival = fit_km_overall(survival_df, label=OVERALL_LABEL)["median_survival"]

    return pd.DataFrame(
        [
            {
                "group": OVERALL_LABEL,
                "n": summary["n"],
                "events": summary["events"],
                "censored": summary["censored"],
                "event_rate": summary["event_rate"],
                "median_followup": summary["median_followup"],
                "max_followup": summary["max_followup"],
                "median_survival": median_survival,
                "time_unit": time_unit,
            }
        ],
        columns=OVERALL_SUMMARY_COLUMNS,
    )


def compute_group_survival_summary(
    survival_ready_df: pd.DataFrame,
    group_col: str = "_group",
    time_unit: str = "unknown",
    group_value_labels: dict | None = None,
    original_group_col: str | None = None,
) -> pd.DataFrame:
    if group_col not in survival_ready_df.columns:
        return pd.DataFrame(columns=GROUP_SUMMARY_COLUMNS)

    survival_df = _prepare_survival_dataframe(survival_ready_df).dropna(subset=[group_col]).copy()
    if survival_df.empty:
        return pd.DataFrame(columns=GROUP_SUMMARY_COLUMNS)

    rows = []
    display_group_col = original_group_col
    used_labels: set[str] = set()
    for raw_group, group_df in _iter_typed_groups(survival_df, group_col):
        n = len(group_df)
        events = int((group_df["_event"] == 1).sum())
        censored = int((group_df["_event"] == 0).sum())
        group_label = format_group_label(raw_group, display_group_col, group_value_labels)
        group_label = _make_unique_group_label(group_label, raw_group, used_labels)
        used_labels.add(group_label)
        km_result = fit_km_overall(group_df, label=group_label)

        rows.append(
            {
                "group": group_label,
                "raw_group": raw_group,
                "n": n,
                "events": events,
                "censored": censored,
                "event_rate": round((events / n * 100) if n else 0.0, 2),
                "median_followup": _reverse_km_median_followup(group_df) if n else None,
                "max_followup": round(float(group_df["_time"].max()), 2) if n else None,
                "median_survival": km_result["median_survival"],
                "time_unit": time_unit,
            }
        )

    return pd.DataFrame(rows, columns=GROUP_SUMMARY_COLUMNS)


def run_logrank_test(
    survival_ready_df: pd.DataFrame,
    group_col: str = "_group",
    max_groups: int = 8,
) -> dict:
    warnings: list[str] = []

    if group_col not in survival_ready_df.columns:
        return {
            "available": False,
            "reason": "A grouping column is required for the log-rank test.",
            "warnings": warnings,
        }

    survival_df = _prepare_survival_dataframe(survival_ready_df).dropna(subset=[group_col]).copy()
    if survival_df.empty:
        return {
            "available": False,
            "reason": "At least two groups with usable rows are required.",
            "warnings": warnings,
        }

    group_count = len(_iter_typed_groups(survival_df, group_col))
    if group_count < 2:
        return {
            "available": False,
            "reason": "At least two groups with usable rows are required.",
            "warnings": warnings,
        }
    if group_count > max_groups:
        return {
            "available": False,
            "reason": (
                f"The log-rank test is limited to {max_groups} groups; "
                f"{group_count} were found."
            ),
            "warnings": ["Selected grouping column has too many groups."],
        }

    events = int((survival_df["_event"] == 1).sum())
    if events == 0:
        return {
            "available": False,
            "reason": "The log-rank test requires at least one observed event.",
            "warnings": ["No events in dataset."],
        }
    result = multivariate_logrank_test(
        event_durations=survival_df["_time"],
        groups=survival_df[group_col].map(_typed_group_key),
        event_observed=survival_df["_event"],
    )

    return {
        "available": True,
        "method": "multivariate_logrank_test",
        "test_statistic": _to_python_float(result.test_statistic),
        "p_value": _to_python_float(result.p_value),
        "degrees_of_freedom": group_count - 1,
        "n_groups": group_count,
        "warnings": warnings,
    }


def run_pairwise_logrank_tests(
    survival_ready_df: pd.DataFrame,
    group_col: str = "_group",
    group_value_labels: dict | None = None,
    original_group_col: str | None = None,
    max_groups: int = 8,
) -> pd.DataFrame:
    if group_col not in survival_ready_df.columns:
        return pd.DataFrame(columns=PAIRWISE_LOGRANK_COLUMNS)

    survival_df = _prepare_survival_dataframe(survival_ready_df).dropna(subset=[group_col]).copy()
    grouped_frames = _iter_typed_groups(survival_df, group_col)
    if len(grouped_frames) < 3 or len(grouped_frames) > max_groups:
        return pd.DataFrame(columns=PAIRWISE_LOGRANK_COLUMNS)
    if int((survival_df["_event"] == 1).sum()) == 0:
        return pd.DataFrame(columns=PAIRWISE_LOGRANK_COLUMNS)

    rows = []
    display_group_col = original_group_col
    used_labels: set[str] = set()
    labeled_groups = []
    for raw_group, group_df in grouped_frames:
        label = format_group_label(raw_group, display_group_col, group_value_labels)
        label = _make_unique_group_label(label, raw_group, used_labels)
        used_labels.add(label)
        labeled_groups.append((raw_group, label, group_df))
    for group_a, group_b in combinations(labeled_groups, 2):
        raw_a, label_a, group_a_df = group_a
        raw_b, label_b, group_b_df = group_b
        pair_has_events = int(group_a_df["_event"].sum() + group_b_df["_event"].sum()) > 0
        if pair_has_events:
            result = logrank_test(
                group_a_df["_time"],
                group_b_df["_time"],
                event_observed_A=group_a_df["_event"],
                event_observed_B=group_b_df["_event"],
            )
            p_value = _to_python_float(result.p_value)
            test_statistic = _to_python_float(result.test_statistic)
        else:
            p_value = np.nan
            test_statistic = np.nan
        rows.append(
            {
                "group_1": label_a,
                "group_2": label_b,
                "raw_group_1": raw_a,
                "raw_group_2": raw_b,
                "test_statistic": test_statistic,
                "p_value": p_value,
                "p_value_formatted": format_p_value(p_value),
                "p_value_holm": np.nan,
                "p_value_holm_formatted": "p = N/A",
            }
        )

    result_df = pd.DataFrame(rows, columns=PAIRWISE_LOGRANK_COLUMNS)
    valid_p_values = result_df["p_value"].dropna()
    if not valid_p_values.empty:
        ordered_indices = valid_p_values.sort_values().index.tolist()
        comparison_count = len(ordered_indices)
        running_max = 0.0
        for rank, row_index in enumerate(ordered_indices):
            adjusted = min(
                1.0,
                float(result_df.at[row_index, "p_value"])
                * (comparison_count - rank),
            )
            running_max = max(running_max, adjusted)
            result_df.at[row_index, "p_value_holm"] = running_max
            result_df.at[row_index, "p_value_holm_formatted"] = format_p_value(
                running_max
            )
    return result_df


def compute_number_at_risk_table(
    survival_ready_df: pd.DataFrame,
    timepoints: list[float],
    group_col: str | None = "_group",
    group_value_labels: dict | None = None,
    original_group_col: str | None = None,
) -> pd.DataFrame:
    survival_df = _prepare_survival_dataframe(survival_ready_df)
    if survival_df.empty:
        return pd.DataFrame(columns=AT_RISK_COLUMNS)

    normalized_timepoints = _normalize_timepoints_with_zero(timepoints)
    if not normalized_timepoints:
        return pd.DataFrame(columns=AT_RISK_COLUMNS)

    if group_col is not None and group_col in survival_df.columns:
        used_labels: set[str] = set()
        grouped_frames = []
        for raw_group, group_df in _iter_typed_groups(survival_df, group_col):
            label = format_group_label(
                raw_group,
                original_group_col,
                group_value_labels,
            )
            label = _make_unique_group_label(label, raw_group, used_labels)
            used_labels.add(label)
            grouped_frames.append((label, group_df))
    else:
        grouped_frames = [(OVERALL_LABEL, survival_df)]

    rows = []
    for group_label, group_df in grouped_frames:
        for timepoint in normalized_timepoints:
            rows.append(
                {
                    "group": group_label,
                    "time": timepoint,
                    "at_risk": int((group_df["_time"] >= timepoint).sum()),
                    "events_up_to_time": int(
                        ((group_df["_time"] <= timepoint) & (group_df["_event"] == 1)).sum()
                    ),
                    "censored_up_to_time": int(
                        ((group_df["_time"] <= timepoint) & (group_df["_event"] == 0)).sum()
                    ),
                }
            )

    return pd.DataFrame(rows, columns=AT_RISK_COLUMNS)


def pivot_at_risk_table(at_risk_df: pd.DataFrame) -> pd.DataFrame:
    if at_risk_df.empty:
        return pd.DataFrame()

    wide = at_risk_df.pivot_table(
        index="group",
        columns="time",
        values="at_risk",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"group": "Group"})
    wide.columns = [
        column if column == "Group" else _format_timepoint_column(column)
        for column in wide.columns
    ]
    return wide


def survival_probability_table_by_group(
    km_results: list[dict[str, Any]],
    timepoints: list[float],
) -> pd.DataFrame:
    rows = []
    for result in km_results:
        label = str(result.get("label", OVERALL_LABEL))
        kmf = result.get("kmf")
        if kmf is None:
            continue

        max_followup = float(kmf.timeline[-1])
        for timepoint in _normalize_timepoints(timepoints):
            try:
                probability_value = (
                    round(float(kmf.predict(float(timepoint))), 4)
                    if timepoint <= max_followup
                    else np.nan
                )
            except Exception:
                probability_value = np.nan

            rows.append(
                {
                    "group": label,
                    "time": timepoint,
                    "survival_probability": probability_value,
                }
            )

    return pd.DataFrame(rows, columns=SURVIVAL_PROBABILITY_COLUMNS)


def pivot_survival_probability_table(prob_df: pd.DataFrame) -> pd.DataFrame:
    if prob_df.empty:
        return pd.DataFrame()

    wide = prob_df.pivot_table(
        index="group",
        columns="time",
        values="survival_probability",
        aggfunc="last",
        dropna=False,
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"group": "Group"})
    wide.columns = [
        column if column == "Group" else _format_timepoint_column(column)
        for column in wide.columns
    ]
    return wide


def generate_survival_interpretation_warnings(
    survival_ready_df: pd.DataFrame,
    group_col: str | None = "_group",
    min_group_size: int = 10,
    min_events_per_group: int = 5,
    max_groups: int = 8,
    check_curve_crossing: bool = True,
) -> list[str]:
    warnings: list[str] = []
    survival_df = _prepare_survival_dataframe(survival_ready_df)

    if survival_df.empty:
        return ["Fewer than 10 usable rows are available."]

    event_count = int((survival_df["_event"] == 1).sum())
    censored_count = int((survival_df["_event"] == 0).sum())
    if event_count == 0:
        warnings.append("No events in dataset.")
    if censored_count == 0:
        warnings.append("No censored observations in dataset.")
    if len(survival_df) < 10:
        warnings.append("Fewer than 10 usable rows are available.")

    if group_col is None or group_col not in survival_df.columns:
        return warnings

    grouped_df = survival_df.dropna(subset=[group_col]).copy()
    if grouped_df.empty:
        return warnings

    typed_group = grouped_df[group_col].map(_typed_group_key)
    group_count = int(typed_group.nunique(dropna=True))
    if group_count > max_groups:
        warnings.append("Selected grouping column has too many groups.")

    group_sizes = grouped_df.groupby(typed_group, dropna=True).size()
    if (group_sizes < min_group_size).any():
        warnings.append(f"One or more groups have fewer than {min_group_size} rows.")

    group_events = grouped_df.groupby(typed_group, dropna=True)["_event"].sum()
    if (group_events < min_events_per_group).any():
        warnings.append(f"One or more groups have fewer than {min_events_per_group} events.")

    group_censored = grouped_df.groupby(typed_group, dropna=True)["_event"].apply(
        lambda values: int((values == 0).sum())
    )
    if (group_censored == 0).any():
        warnings.append("One or more groups have no censored observations.")

    if check_curve_crossing and 2 <= group_count <= max_groups:
        group_results, _ = fit_km_by_group(
            grouped_df,
            group_col=group_col,
            min_group_size=1,
            max_groups=max_groups,
        )
        if detect_curve_crossing(combine_curve_results(group_results)):
            warnings.append(
                "Survival curves may cross. The log-rank test can be less reliable when hazards are not proportional."
            )

    return warnings


def detect_curve_crossing(curve_df: pd.DataFrame) -> bool:
    required_columns = {"time", "survival", "group"}
    if curve_df.empty or not required_columns.issubset(curve_df.columns):
        return False

    groups = list(curve_df["group"].dropna().unique())
    if len(groups) < 2:
        return False

    pivoted = (
        curve_df.pivot_table(index="time", columns="group", values="survival", aggfunc="last")
        .sort_index()
        .ffill()
    )

    for group_a, group_b in combinations(pivoted.columns, 2):
        difference = (pivoted[group_a] - pivoted[group_b]).dropna()
        signs = np.sign(difference[difference != 0])
        if len(signs) < 2:
            continue
        if (signs.shift(1).dropna() * signs.iloc[1:] < 0).any():
            return True

    return False


def combine_curve_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=CURVE_COLUMNS)

    return pd.concat([result["curve"] for result in results], ignore_index=True)[CURVE_COLUMNS]


def _prepare_survival_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "_time" not in df.columns or "_event" not in df.columns:
        return pd.DataFrame(columns=list(df.columns) + ["_time", "_event"])

    survival_df = df.copy(deep=True)
    survival_df["_time"] = pd.to_numeric(survival_df["_time"], errors="coerce")
    survival_df["_event"] = pd.to_numeric(survival_df["_event"], errors="coerce")
    survival_df = survival_df.dropna(subset=["_time", "_event"]).copy()
    survival_df = survival_df[survival_df["_event"].isin([0, 1])].copy()
    survival_df = survival_df[np.isfinite(survival_df["_time"].astype(float))].copy()
    survival_df = survival_df[survival_df["_time"] >= 0].copy()
    survival_df["_event"] = survival_df["_event"].astype(int)
    return survival_df


def _normalize_timepoints_with_zero(timepoints: list[float]) -> list[float]:
    return _normalize_timepoints([0.0] + list(timepoints))


def _normalize_timepoints(timepoints: list[float]) -> list[float]:
    normalized: list[float] = []
    for timepoint in timepoints:
        numeric_timepoint = pd.to_numeric(pd.Series([timepoint]), errors="coerce").iloc[0]
        if (
            pd.isna(numeric_timepoint)
            or not np.isfinite(float(numeric_timepoint))
            or float(numeric_timepoint) < 0
        ):
            continue
        numeric_timepoint = round(float(numeric_timepoint), 6)
        if numeric_timepoint not in normalized:
            normalized.append(numeric_timepoint)
    return sorted(normalized)


def _group_label_key(value: Any) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric_value):
        float_value = float(numeric_value)
        if float_value.is_integer():
            return str(int(float_value))
        return str(float_value)

    return str(value)


def _typed_group_key(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{value!r}"


def _iter_typed_groups(
    dataframe: pd.DataFrame,
    group_col: str,
) -> list[tuple[Any, pd.DataFrame]]:
    grouped = dataframe.dropna(subset=[group_col]).copy()
    if grouped.empty:
        return []
    keys = grouped[group_col].map(_typed_group_key)
    result: list[tuple[Any, pd.DataFrame]] = []
    for key in sorted(keys.unique()):
        mask = keys.eq(key)
        raw_value = grouped.loc[mask, group_col].iloc[0]
        result.append((raw_value, grouped.loc[mask].copy()))
    return result


def _make_unique_group_label(
    label: str,
    raw_value: Any,
    used_labels: set[str],
) -> str:
    if label not in used_labels:
        return label
    type_name = type(raw_value).__name__
    candidate = f"{label} [raw: {raw_value!s}; type: {type_name}]"
    suffix = 2
    while candidate in used_labels:
        candidate = (
            f"{label} [raw: {raw_value!s}; type: {type_name}; #{suffix}]"
        )
        suffix += 1
    return candidate


def _format_timepoint_column(value: Any) -> str:
    numeric = float(value)
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def _reverse_km_median_followup(dataframe: pd.DataFrame) -> float | None:
    """Estimate potential follow-up with the reverse Kaplan-Meier method."""
    if dataframe.empty:
        return None
    reverse_km = KaplanMeierFitter()
    reverse_km.fit(
        durations=dataframe["_time"],
        event_observed=1 - dataframe["_event"].astype(int),
    )
    median = _to_python_float(reverse_km.median_survival_time_)
    if not np.isfinite(median):
        return None
    return round(median, 2)


def _curve_dataframe_from_kmf(kmf: KaplanMeierFitter, label: str) -> pd.DataFrame:
    survival_function = kmf.survival_function_.reset_index()
    confidence_interval = kmf.confidence_interval_survival_function_.reset_index()

    return pd.DataFrame(
        {
            "time": survival_function.iloc[:, 0].astype(float),
            "survival": survival_function.iloc[:, 1].astype(float),
            "ci_lower": confidence_interval.iloc[:, 1].astype(float),
            "ci_upper": confidence_interval.iloc[:, 2].astype(float),
            "censored": kmf.event_table["censored"].astype(int).to_numpy(),
            "group": label,
        }
    )


def _to_python_float(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)
