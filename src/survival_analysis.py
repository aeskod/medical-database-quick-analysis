from typing import Any

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


CURVE_COLUMNS = ["time", "survival", "ci_lower", "ci_upper", "group"]


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
    median_followup = round(float(survival_df["_time"].median()), 2) if n else None
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

    group_count = int(survival_df[group_col].nunique(dropna=True))
    if group_count == 0:
        return [], warnings + ["No non-missing group values are available for grouped analysis."]

    if group_count > max_groups:
        return [], warnings + ["Selected grouping column has too many groups for a readable KM plot."]

    group_results = []
    for group_value, group_df in survival_df.groupby(group_col, sort=True, dropna=True):
        label = str(group_value)
        if len(group_df) < min_group_size:
            warnings.append(f"Group {label} has fewer than {min_group_size} rows. Interpret cautiously.")

        group_results.append(fit_km_overall(group_df, label=label))

    return group_results, warnings


def suggest_timepoints(max_time: float, time_unit: str = "unknown") -> list[float]:
    if max_time is None or pd.isna(max_time) or max_time <= 0:
        return []

    fixed_timepoints = {
        "days": [30, 90, 180, 365, 730, 1095, 1825],
        "months": [1, 3, 6, 12, 24, 36, 60],
        "years": [1, 3, 5, 10],
    }

    if time_unit in fixed_timepoints:
        selected = [timepoint for timepoint in fixed_timepoints[time_unit] if timepoint <= max_time]
        if selected:
            return [round(float(timepoint), 2) for timepoint in selected]

    return [round(float(max_time) * fraction, 2) for fraction in [0.25, 0.5, 0.75]]


def survival_probability_at_times(kmf: KaplanMeierFitter, timepoints: list[float]) -> pd.DataFrame:
    probabilities = kmf.predict(timepoints)

    return pd.DataFrame(
        {
            "time": [round(float(timepoint), 2) for timepoint in timepoints],
            "survival_probability": [round(float(probability), 4) for probability in probabilities],
        }
    )


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
    survival_df["_event"] = survival_df["_event"].astype(int)
    return survival_df


def _curve_dataframe_from_kmf(kmf: KaplanMeierFitter, label: str) -> pd.DataFrame:
    survival_function = kmf.survival_function_.reset_index()
    confidence_interval = kmf.confidence_interval_survival_function_.reset_index()

    return pd.DataFrame(
        {
            "time": survival_function.iloc[:, 0].astype(float),
            "survival": survival_function.iloc[:, 1].astype(float),
            "ci_lower": confidence_interval.iloc[:, 1].astype(float),
            "ci_upper": confidence_interval.iloc[:, 2].astype(float),
            "group": label,
        }
    )


def _to_python_float(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)
