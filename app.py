from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st

from src.data_quality import (
    build_data_quality_report,
    determine_quality_status,
    issues_to_dataframe,
)
from src.cohort_overview import (
    build_baseline_table,
    compute_cohort_overview_metrics,
    count_variable_types,
    get_default_baseline_variables,
)
from src.data_loading import read_dataset
from src.profiling import normalize_missing_values, profile_dataframe
from src.role_suggestions import suggest_survival_roles
from src.survival_analysis import (
    combine_curve_results,
    fit_km_by_group,
    fit_km_overall,
    get_survival_summary,
    suggest_timepoints,
    survival_probability_at_times,
    validate_survival_ready_dataframe,
)
from src.survival_mapping import (
    SurvivalConfig,
    create_survival_ready_dataframe,
    validate_survival_config,
)
from src.survival_plots import plot_km_curve


def main() -> None:
    st.set_page_config(page_title="Medical Dataset Explorer", layout="wide")

    st.title("Medical Dataset Explorer")

    upload_tab, data_quality_tab, cohort_tab, charts_tab, survival_tab = st.tabs(
        ["Upload", "Data Quality", "Cohort Overview", "Charts", "Survival Analysis"]
    )

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a dataset",
            type=["csv", "tsv", "xlsx"],
            help="Supported formats: CSV, TSV, XLSX",
        )
        st.caption("Supported formats: CSV, TSV, XLSX")

        if uploaded_file is None:
            st.info("Upload a CSV, TSV, or Excel file to begin.")
        else:
            try:
                df = read_dataset(uploaded_file)
            except ValueError as exc:
                st.error(str(exc))
            else:
                _sync_uploaded_dataset_state(uploaded_file.name, df)
                st.success("Dataset loaded successfully:")
                st.markdown(
                    "\n".join(
                        [
                            f"- File: {uploaded_file.name}",
                            f"- Rows: {len(df)}",
                            f"- Columns: {len(df.columns)}",
                        ]
                    )
                )

                st.subheader("Preview")
                st.dataframe(df.head(20), width="stretch")

                st.subheader("Column profile")
                profile = st.session_state["profile_df"]
                st.dataframe(profile, width="stretch")

                st.subheader("Survival setup")
                _render_survival_setup(df, profile)

    with data_quality_tab:
        _render_data_quality_tab()

    with cohort_tab:
        _render_cohort_overview_tab()

    with charts_tab:
        st.info("Charts will be implemented in a later step.")

    with survival_tab:
        _render_survival_analysis_tab()


def _sync_uploaded_dataset_state(file_name: str, df: pd.DataFrame) -> None:
    dataset_signature = (file_name, len(df), len(df.columns), tuple(str(column) for column in df.columns))
    previous_signature = st.session_state.get("uploaded_dataset_signature")

    if previous_signature != dataset_signature:
        st.session_state.pop("survival_config", None)
        st.session_state.pop("survival_ready_df", None)
        st.session_state.pop("data_quality_report", None)
        st.session_state.pop("cohort_group_col", None)
        st.session_state.pop("cohort_continuous_vars", None)
        st.session_state.pop("cohort_categorical_vars", None)

    st.session_state["uploaded_dataset_signature"] = dataset_signature
    st.session_state["uploaded_df"] = df.copy(deep=True)
    st.session_state["profile_df"] = profile_dataframe(df)


def _render_data_quality_tab() -> None:
    st.subheader("Data Quality")

    uploaded_df = st.session_state.get("uploaded_df")
    profile_df = st.session_state.get("profile_df")

    if uploaded_df is None:
        st.info("No dataset uploaded yet.\n\nUpload a dataset first to run data-quality checks.")
        return

    report = build_data_quality_report(
        df=uploaded_df,
        profile_df=profile_df,
        survival_config=st.session_state.get("survival_config"),
        survival_ready_df=st.session_state.get("survival_ready_df"),
    )
    st.session_state["data_quality_report"] = report

    _render_quality_status(report["issues"])
    _render_quality_summary_cards(report)
    _render_quality_issues(report["issues"])
    _render_missingness_section(report)
    _render_duplicate_checks(report)
    _render_survival_quality_section(report)
    _render_group_quality_section(report)
    _render_sensitive_columns_section(report)


def _render_quality_status(issues: list[Any]) -> None:
    status = determine_quality_status(issues)

    if status == "error":
        st.error("Blocking issues found. Resolve these before interpreting survival results.")
    elif status == "warning":
        st.warning("Warnings found. Review the issues below before interpreting survival results.")
    else:
        st.success("No major issues detected.")


def _render_quality_summary_cards(report: dict[str, Any]) -> None:
    overview = report["overview"]
    duplicate_rows = report["duplicate_rows"]
    survival_quality = report["survival_quality"]

    first_row = st.columns(4)
    first_row[0].metric("Rows", overview["n_rows"])
    first_row[1].metric("Columns", overview["n_columns"])
    first_row[2].metric("Missing cells", overview["missing_cells"])
    first_row[3].metric("Duplicate rows", duplicate_rows["duplicate_row_count"])

    second_row = st.columns(2)
    if survival_quality.get("has_mapping"):
        second_row[0].metric("Usable survival rows", survival_quality["usable_survival_rows"])
        second_row[1].metric("Excluded survival rows", survival_quality["excluded_rows"])
    else:
        second_row[0].metric("Usable survival rows", "Not available")
        second_row[1].metric("Excluded survival rows", "Not available")


def _render_quality_issues(issues: list[Any]) -> None:
    st.subheader("Issues")

    if not issues:
        st.success("No major issues detected.")
        return

    issue_df = issues_to_dataframe(issues)
    severity_order = {"error": 0, "warning": 1, "info": 2}
    issue_df["_sort"] = issue_df["severity"].map(severity_order).fillna(3)
    issue_df = issue_df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    st.dataframe(issue_df, hide_index=True, width="stretch")


def _render_missingness_section(report: dict[str, Any]) -> None:
    st.subheader("Missingness by column")
    missingness_by_column = report["missingness_by_column"]
    st.dataframe(missingness_by_column, hide_index=True, width="stretch")

    if not missingness_by_column.empty:
        chart_data = missingness_by_column.set_index("column_name")["missing_percent"]
        st.bar_chart(chart_data)

    st.subheader("Rows with missing values")
    missingness_by_row = report["missingness_by_row"].head(50)
    if missingness_by_row.empty:
        st.success("No rows with missing values.")
    else:
        st.dataframe(missingness_by_row, hide_index=True, width="stretch")


def _render_duplicate_checks(report: dict[str, Any]) -> None:
    st.subheader("Duplicate checks")
    duplicate_rows = report["duplicate_rows"]
    duplicate_ids = report["duplicate_ids"]

    duplicate_row_summary = pd.DataFrame(
        [
            {"Metric": "Duplicate row count", "Value": duplicate_rows["duplicate_row_count"]},
            {"Metric": "Duplicate row percent", "Value": f"{duplicate_rows['duplicate_row_percent']}%"},
            {"Metric": "Duplicate row groups", "Value": duplicate_rows["duplicate_group_count"]},
            {
                "Metric": "Example duplicate row indices",
                "Value": ", ".join(str(index) for index in duplicate_rows["example_duplicate_indices"]) or "None",
            },
        ]
    )
    st.markdown("**Exact duplicate rows**")
    st.dataframe(duplicate_row_summary, hide_index=True, width="stretch")

    st.markdown("**Patient ID duplicates**")
    if not duplicate_ids["checked"]:
        st.info("Patient ID duplicate check was not run because no patient ID column is selected.")
        return

    duplicate_id_summary = pd.DataFrame(
        [
            {"Metric": "Selected ID column", "Value": duplicate_ids["id_col"]},
            {"Metric": "Duplicate ID row count", "Value": duplicate_ids["duplicate_id_row_count"]},
            {"Metric": "Duplicate ID value count", "Value": duplicate_ids["duplicate_id_value_count"]},
            {"Metric": "Duplicate ID percent", "Value": f"{duplicate_ids['duplicate_id_percent']}%"},
            {
                "Metric": "Example duplicate IDs",
                "Value": ", ".join(duplicate_ids["duplicate_id_examples"]) or "None",
            },
        ]
    )
    st.dataframe(duplicate_id_summary, hide_index=True, width="stretch")


def _render_survival_quality_section(report: dict[str, Any]) -> None:
    st.subheader("Survival-specific checks")
    survival_quality = report["survival_quality"]

    if not survival_quality.get("has_mapping"):
        st.info("Survival-specific checks will appear after survival mapping is confirmed.")
        return

    survival_summary = pd.DataFrame(
        [
            {"Metric": "Time column", "Value": survival_quality["time_col"]},
            {"Metric": "Event column", "Value": survival_quality["event_col"]},
            {"Metric": "Raw rows", "Value": survival_quality["raw_rows"]},
            {"Metric": "Usable survival rows", "Value": survival_quality["usable_survival_rows"]},
            {"Metric": "Excluded rows", "Value": survival_quality["excluded_rows"]},
            {"Metric": "Events", "Value": survival_quality["events"]},
            {"Metric": "Censored", "Value": survival_quality["censored"]},
            {"Metric": "Event rate", "Value": f"{survival_quality['event_rate']}%"},
            {"Metric": "Missing time values", "Value": survival_quality["time_missing_count"]},
            {"Metric": "Missing event values", "Value": survival_quality["event_missing_count"]},
            {"Metric": "Negative time values", "Value": survival_quality["negative_time_count"]},
            {"Metric": "Zero time values", "Value": survival_quality["zero_time_count"]},
            {
                "Metric": "Unmapped event values",
                "Value": ", ".join(survival_quality["unmapped_event_values"]) or "None",
            },
        ]
    )
    st.dataframe(survival_summary, hide_index=True, width="stretch")

    exclusion_breakdown = report["survival_exclusion_breakdown"]
    if not exclusion_breakdown.empty:
        st.markdown("**Survival exclusion breakdown**")
        st.dataframe(exclusion_breakdown, hide_index=True, width="stretch")


def _render_group_quality_section(report: dict[str, Any]) -> None:
    group_quality = report["group_quality"]
    if group_quality is None:
        return

    st.subheader("Grouping quality")
    st.dataframe(group_quality, hide_index=True, width="stretch")


def _render_sensitive_columns_section(report: dict[str, Any]) -> None:
    st.subheader("Possible identifier / sensitive columns")
    sensitive_columns = report["sensitive_column_candidates"]

    if sensitive_columns.empty:
        st.success("No obvious identifier-like columns detected.")
        return

    st.warning(
        "Possible identifier or sensitive columns detected. "
        "This is not a privacy audit. Review these columns before using real patient data."
    )
    st.dataframe(sensitive_columns, hide_index=True, width="stretch")


def _render_cohort_overview_tab() -> None:
    st.subheader("Cohort Overview")

    uploaded_df = st.session_state.get("uploaded_df")
    profile_df = st.session_state.get("profile_df")
    survival_config = st.session_state.get("survival_config")
    survival_ready_df = st.session_state.get("survival_ready_df")

    if uploaded_df is None:
        st.info("No dataset uploaded yet.\n\nUpload a dataset first to view cohort summaries.")
        return

    time_unit = getattr(survival_config, "time_unit", "unknown") if survival_config is not None else "unknown"
    metrics = compute_cohort_overview_metrics(
        uploaded_df,
        survival_ready_df=survival_ready_df,
        time_unit=time_unit,
    )
    default_variables = get_default_baseline_variables(uploaded_df, profile_df, survival_config)
    variable_type_summary = count_variable_types(uploaded_df, profile_df, survival_config)

    _render_cohort_summary_cards(metrics)
    _render_cohort_mapping_summary(survival_config)

    st.subheader("Variable type summary")
    st.dataframe(variable_type_summary, hide_index=True, width="stretch")

    st.subheader("Variables to include")
    group_col = _render_cohort_group_selector(uploaded_df, default_variables, survival_config)
    continuous_vars, categorical_vars, max_levels, include_missing = _render_baseline_table_controls(
        uploaded_df,
        default_variables,
    )

    overlapping_vars = sorted(set(continuous_vars) & set(categorical_vars))
    if overlapping_vars:
        st.warning(
            "Variables selected as both continuous and categorical will be summarized as categorical: "
            + ", ".join(overlapping_vars)
        )
        continuous_vars = [column for column in continuous_vars if column not in overlapping_vars]

    st.subheader("Baseline characteristics")
    if not continuous_vars and not categorical_vars:
        st.info("No variables selected for baseline table.")
        return

    baseline_table = build_baseline_table(
        uploaded_df,
        continuous_vars,
        categorical_vars,
        group_col=group_col,
        max_levels=max_levels,
        include_missing=include_missing,
        group_value_labels=_group_value_labels_for_column(group_col),
    )
    st.dataframe(baseline_table, hide_index=True, width="stretch")
    st.download_button(
        "Download baseline table as CSV",
        data=baseline_table.to_csv(index=False).encode("utf-8"),
        file_name="baseline_characteristics.csv",
        mime="text/csv",
    )


def _render_cohort_summary_cards(metrics: dict[str, Any]) -> None:
    first_row = st.columns(4)
    first_row[0].metric("Rows", metrics["n_rows"])
    first_row[1].metric("Columns", metrics["n_columns"])
    first_row[2].metric(
        "Complete rows",
        f"{metrics['complete_rows']} ({_format_number(metrics['complete_rows_percent'])}%)",
    )
    first_row[3].metric("Missing cells %", f"{_format_number(metrics['missing_cells_percent'])}%")

    second_row = st.columns(5)
    second_row[0].metric("Usable survival rows", _metric_value(metrics["usable_survival_rows"]))
    second_row[1].metric("Events", _metric_value(metrics["events"]))
    second_row[2].metric("Censored", _metric_value(metrics["censored"]))
    second_row[3].metric(
        "Event rate",
        "Not available" if metrics["event_rate"] is None else f"{_format_number(metrics['event_rate'])}%",
    )
    second_row[4].metric(
        "Median follow-up",
        _format_time_value(metrics["median_followup"], metrics["time_unit"]),
    )


def _render_cohort_mapping_summary(survival_config: SurvivalConfig | None) -> None:
    if survival_config is None:
        st.info(
            "Survival mapping has not been confirmed yet. Cohort summaries are still available, "
            "but survival-specific metrics are disabled."
        )
        return

    st.markdown("**Survival mapping**")
    mapping_summary = pd.DataFrame(
        [
            {"Field": "time", "Value": survival_config.time_col},
            {"Field": "event", "Value": survival_config.event_col},
            {
                "Field": "event values",
                "Value": ", ".join(_format_value(value) for value in survival_config.event_values),
            },
            {
                "Field": "censored values",
                "Value": ", ".join(_format_value(value) for value in survival_config.censor_values),
            },
            {"Field": "group", "Value": survival_config.group_col or "None"},
            {"Field": "time unit", "Value": survival_config.time_unit},
        ]
    )
    st.dataframe(mapping_summary, hide_index=True, width="stretch")


def _render_cohort_group_selector(
    df: pd.DataFrame,
    default_variables: dict[str, list[str]],
    survival_config: SurvivalConfig | None,
) -> str | None:
    all_columns = [str(column) for column in df.columns]
    categorical_columns = [
        column
        for column in default_variables.get("categorical", [])
        if column in all_columns
    ]
    remaining_columns = [
        column
        for column in all_columns
        if column not in categorical_columns
    ]
    options = ["No grouping"] + categorical_columns + remaining_columns
    default_group = getattr(survival_config, "group_col", None) if survival_config is not None else None
    default_option = default_group if default_group in options else "No grouping"

    if st.session_state.get("cohort_group_col") not in options:
        st.session_state["cohort_group_col"] = default_option

    selected = st.selectbox(
        "Group by",
        options,
        index=options.index(st.session_state["cohort_group_col"]),
        key="cohort_group_col",
    )
    return None if selected == "No grouping" else str(selected)


def _render_baseline_table_controls(
    df: pd.DataFrame,
    default_variables: dict[str, list[str]],
) -> tuple[list[str], list[str], int, bool]:
    all_columns = [str(column) for column in df.columns]
    continuous_default = [
        column
        for column in default_variables.get("continuous", [])
        if column in all_columns
    ]
    categorical_default = [
        column
        for column in default_variables.get("categorical", [])
        if column in all_columns
    ]

    _sanitize_multiselect_state("cohort_continuous_vars", all_columns)
    _sanitize_multiselect_state("cohort_categorical_vars", all_columns)

    continuous_vars = st.multiselect(
        "Continuous variables",
        all_columns,
        default=continuous_default,
        key="cohort_continuous_vars",
    )
    categorical_vars = st.multiselect(
        "Categorical variables",
        all_columns,
        default=categorical_default,
        key="cohort_categorical_vars",
    )

    control_row = st.columns(2)
    max_levels = int(
        control_row[0].slider(
            "Maximum categorical levels",
            min_value=3,
            max_value=30,
            value=10,
        )
    )
    include_missing = bool(control_row[1].checkbox("Include missing rows", value=True))

    return list(continuous_vars), list(categorical_vars), max_levels, include_missing


def _sanitize_multiselect_state(key: str, valid_options: list[str]) -> None:
    existing_value = st.session_state.get(key)
    if isinstance(existing_value, list):
        st.session_state[key] = [
            value
            for value in existing_value
            if value in valid_options
        ]


def _group_value_labels_for_column(group_col: str | None) -> dict[str, str]:
    if not group_col:
        return {}

    all_labels = st.session_state.get("group_value_labels")
    if not isinstance(all_labels, dict):
        return {}

    column_labels = all_labels.get(group_col, {})
    if not isinstance(column_labels, dict):
        return {}

    return {
        str(raw_value): str(display_value)
        for raw_value, display_value in column_labels.items()
    }


def _metric_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return "Not available"
    return value


def _render_survival_analysis_tab() -> None:
    st.subheader("Survival Analysis")

    config = st.session_state.get("survival_config")
    survival_ready_df = st.session_state.get("survival_ready_df")

    if config is None or survival_ready_df is None:
        st.info(
            "No survival mapping has been confirmed yet.\n\n"
            "Go to the Upload tab and confirm the required survival columns first."
        )
        return

    _render_current_mapping(config)

    errors, warnings = validate_survival_ready_dataframe(survival_ready_df)
    for warning in warnings:
        st.warning(warning)

    if errors:
        st.error("Survival analysis cannot be run because:")
        for error in errors:
            st.markdown(f"- {error}")
        return

    st.success("Survival-ready data validated.")

    summary = get_survival_summary(survival_ready_df, config.time_unit)
    overall_result = fit_km_overall(survival_ready_df)
    _render_survival_summary_metrics(
        summary,
        overall_result["median_survival"],
        config.time_unit,
    )

    st.markdown("**Plot controls**")
    show_ci = st.checkbox("Show confidence interval", value=True)
    has_group = "_group" in survival_ready_df.columns
    use_group = st.checkbox(
        "Use grouping column if available",
        value=has_group,
        disabled=not has_group,
    )

    plot_results = [overall_result]
    plot_title = "Kaplan-Meier Survival Curve"
    group_warnings: list[str] = []

    if use_group and has_group:
        group_results, group_warnings = fit_km_by_group(survival_ready_df)
        if group_results:
            plot_results = group_results
            plot_title = "Grouped Kaplan-Meier Survival Curve"

    curve_df = combine_curve_results(plot_results)
    fig = plot_km_curve(
        curve_df,
        title=plot_title,
        time_unit=config.time_unit,
        show_ci=show_ci,
    )
    st.plotly_chart(fig, width="stretch")

    for warning in group_warnings:
        st.warning(warning)

    st.subheader("Survival probability at selected times")
    suggested_timepoints = suggest_timepoints(summary["max_followup"], config.time_unit)
    default_timepoint_text = ", ".join(_format_number(timepoint) for timepoint in suggested_timepoints)
    timepoint_text = st.text_input("Time points", default_timepoint_text)
    timepoints, parse_warning = _parse_timepoints(timepoint_text, suggested_timepoints)

    if parse_warning:
        st.warning(parse_warning)

    if not timepoints:
        st.info("No time points are available for this follow-up range.")
        return

    probability_df = survival_probability_at_times(overall_result["kmf"], timepoints)
    st.dataframe(probability_df, width="stretch")


def _render_current_mapping(config: SurvivalConfig) -> None:
    st.markdown("**Current survival mapping**")
    mapping_df = pd.DataFrame(
        [
            {"Field": "Time column", "Value": config.time_col},
            {"Field": "Event column", "Value": config.event_col},
            {"Field": "Event values", "Value": ", ".join(_format_value(value) for value in config.event_values)},
            {"Field": "Censor values", "Value": ", ".join(_format_value(value) for value in config.censor_values)},
            {"Field": "Patient ID column", "Value": config.id_col or "Row number"},
            {"Field": "Group column", "Value": config.group_col or "None"},
            {"Field": "Time unit", "Value": config.time_unit},
        ]
    )
    st.dataframe(mapping_df, hide_index=True, width="stretch")


def _render_survival_summary_metrics(
    summary: dict[str, Any],
    median_survival: float,
    time_unit: str,
) -> None:
    first_row = st.columns(4)
    first_row[0].metric("Usable rows", summary["n"])
    first_row[1].metric("Events", summary["events"])
    first_row[2].metric("Censored", summary["censored"])
    first_row[3].metric("Event rate", f"{summary['event_rate']}%")

    second_row = st.columns(3)
    second_row[0].metric("Median follow-up", _format_time_value(summary["median_followup"], time_unit))
    second_row[1].metric("Max follow-up", _format_time_value(summary["max_followup"], time_unit))
    second_row[2].metric("Median survival", _format_time_value(median_survival, time_unit, not_reached=True))


def _parse_timepoints(
    timepoint_text: str,
    fallback_timepoints: list[float],
) -> tuple[list[float], str | None]:
    try:
        parsed_timepoints = [
            float(value.strip())
            for value in timepoint_text.split(",")
            if value.strip()
        ]
    except ValueError:
        return fallback_timepoints, "Could not parse time points; using suggested defaults."

    parsed_timepoints = [timepoint for timepoint in parsed_timepoints if timepoint >= 0]
    if not parsed_timepoints and fallback_timepoints:
        return fallback_timepoints, "No valid non-negative time points were entered; using suggested defaults."

    return parsed_timepoints, None


def _format_time_value(value: Any, time_unit: str, not_reached: bool = False) -> str:
    if value is None or pd.isna(value):
        return "Not available"

    numeric_value = float(value)
    if not_reached and numeric_value == float("inf"):
        return "Not reached"

    suffix = "" if time_unit == "unknown" else f" {time_unit}"
    return f"{_format_number(numeric_value)}{suffix}"


def _format_number(value: Any) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.2f}"


def _render_survival_setup(df: pd.DataFrame, profile: pd.DataFrame) -> None:
    role_suggestions = suggest_survival_roles(profile)
    all_columns = [str(column) for column in df.columns]

    time_col = _select_required_role_column(
        "Follow-up / survival time column",
        role_suggestions["time_candidates"],
        all_columns,
        "time_col",
    )

    event_col = _select_required_role_column(
        "Event/status column",
        role_suggestions["event_candidates"],
        all_columns,
        "event_col",
    )

    event_values: list[Any] = []
    censor_values: list[Any] = []
    missing_event_handling = "exclude"

    if event_col:
        event_values, censor_values, missing_event_handling = _render_event_value_mapping(df, event_col)

    id_col = _select_optional_role_column(
        "Patient ID column",
        role_suggestions["id_candidates"],
        all_columns,
        "No patient ID / use row number",
        "id_col",
    )

    group_col = _select_optional_role_column(
        "Optional grouping column",
        role_suggestions["group_candidates"],
        all_columns,
        "No grouping",
        "group_col",
    )

    time_unit = _render_time_unit_selector(time_col)

    if st.button("Confirm survival mapping", type="primary"):
        config = SurvivalConfig(
            time_col=time_col,
            event_col=event_col,
            event_values=event_values,
            censor_values=censor_values,
            id_col=id_col,
            group_col=group_col,
            time_unit=time_unit,
            missing_event_handling=missing_event_handling,
        )
        errors, warnings = validate_survival_config(df, config)

        for warning in warnings:
            st.warning(warning)

        if errors:
            for error in errors:
                st.error(error)
            return

        survival_ready_df = create_survival_ready_dataframe(df, config)
        st.session_state["survival_config"] = config
        st.session_state["survival_ready_df"] = survival_ready_df

        event_count = int((survival_ready_df["_event"] == 1).sum())
        censored_count = int((survival_ready_df["_event"] == 0).sum())
        st.success(
            "Survival mapping confirmed.\n\n"
            f"Usable rows: {len(survival_ready_df)}\n\n"
            f"Events: {event_count}\n\n"
            f"Censored: {censored_count}"
        )
        st.json(_json_safe_config(config))


def _select_required_role_column(
    label: str,
    candidates: list[dict[str, Any]],
    all_columns: list[str],
    key_prefix: str,
) -> str:
    st.markdown(f"**{label}**")
    _render_candidate_recommendations(candidates)

    if candidates:
        candidate_columns = [candidate["column_name"] for candidate in candidates]
        options = candidate_columns + ["__search_all__"]
        selected = st.radio(
            label,
            options,
            format_func=lambda option: _format_candidate_option(option, candidates),
            key=f"{key_prefix}_recommended",
            label_visibility="collapsed",
        )

        if selected != "__search_all__":
            return str(selected)

    return str(
        st.selectbox(
            f"Search all columns for {label.lower()}",
            all_columns,
            key=f"{key_prefix}_all_columns",
        )
    )


def _select_optional_role_column(
    label: str,
    candidates: list[dict[str, Any]],
    all_columns: list[str],
    none_label: str,
    key_prefix: str,
) -> str | None:
    st.markdown(f"**{label}**")
    _render_candidate_recommendations(candidates)

    options = ["__none__"] + [candidate["column_name"] for candidate in candidates] + ["__search_all__"]
    selected = st.radio(
        label,
        options,
        format_func=lambda option: _format_optional_candidate_option(
            option,
            candidates,
            none_label,
        ),
        key=f"{key_prefix}_recommended",
        label_visibility="collapsed",
    )

    if selected == "__none__":
        return None

    if selected == "__search_all__":
        fallback_options = [none_label] + all_columns
        fallback_selected = st.selectbox(
            f"Search all columns for {label.lower()}",
            fallback_options,
            key=f"{key_prefix}_all_columns",
        )
        return None if fallback_selected == none_label else str(fallback_selected)

    return str(selected)


def _render_candidate_recommendations(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        st.caption("No recommended candidates found. Use the search-all-columns selector.")
        return

    st.caption("Recommended columns")
    for candidate in candidates:
        reasons = "; ".join(candidate["reasons"]) if candidate["reasons"] else "No specific reasons"
        st.caption(
            f"{candidate['column_name']} - {candidate['confidence']} confidence "
            f"({candidate['score']}). Reasons: {reasons}"
        )


def _format_candidate_option(option: str, candidates: list[dict[str, Any]]) -> str:
    if option == "__search_all__":
        return "Search all columns"

    candidate = _candidate_by_column(option, candidates)
    if candidate is None:
        return option

    return f"{candidate['column_name']} - {candidate['confidence']} confidence ({candidate['score']})"


def _format_optional_candidate_option(
    option: str,
    candidates: list[dict[str, Any]],
    none_label: str,
) -> str:
    if option == "__none__":
        return none_label
    return _format_candidate_option(option, candidates)


def _candidate_by_column(column_name: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate["column_name"] == column_name:
            return candidate
    return None


def _render_event_value_mapping(
    df: pd.DataFrame,
    event_col: str,
) -> tuple[list[Any], list[Any], str]:
    unique_values = _unique_non_missing_values(df, event_col)
    st.markdown(f"Values found in `{event_col}`:")

    if not unique_values:
        st.caption("No non-missing values found.")
        return [], [], "exclude"

    st.caption(", ".join(_format_value(value) for value in unique_values))

    default_event_values = _default_event_values(unique_values)
    event_values = st.multiselect(
        "Which value(s) mean the event occurred?",
        unique_values,
        default=default_event_values,
        format_func=_format_value,
        key=f"event_values_{event_col}",
    )
    event_keys = {_canonical_value(value) for value in event_values}
    default_censor_values = [
        value for value in unique_values if _canonical_value(value) not in event_keys
    ]
    censor_values = st.multiselect(
        "Censored values",
        unique_values,
        default=default_censor_values,
        format_func=_format_value,
        key=f"censor_values_{event_col}",
    )
    missing_choice = st.radio(
        "Missing values",
        ["Exclude from survival analysis", "Treat as censored"],
        index=0,
        key=f"missing_event_handling_{event_col}",
    )
    missing_event_handling = "treat_as_censored" if missing_choice == "Treat as censored" else "exclude"

    return event_values, censor_values, missing_event_handling


def _render_time_unit_selector(time_col: str) -> str:
    options = ["days", "months", "years", "unknown"]
    suggested_unit = _suggest_time_unit(time_col)
    return str(
        st.selectbox(
            "Time unit",
            options,
            index=options.index(suggested_unit),
            key=f"time_unit_{time_col}",
        )
    )


def _unique_non_missing_values(df: pd.DataFrame, column: str) -> list[Any]:
    normalized = normalize_missing_values(df[[column]])
    values = list(pd.unique(normalized[column].dropna()))
    return values


def _default_event_values(unique_values: list[Any]) -> list[Any]:
    numeric_values = pd.to_numeric(pd.Series(unique_values), errors="coerce")
    if len(unique_values) == 2 and numeric_values.notna().all():
        numeric_set = set(numeric_values.astype(float))
        target_value = 1.0 if numeric_set == {0.0, 1.0} else float(numeric_values.max())

        for value in unique_values:
            parsed_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if parsed_value == target_value:
                return [value]

    event_markers = {
        "true",
        "yes",
        "y",
        "dead",
        "deceased",
        "event",
        "relapsed",
        "progressed",
        "death",
    }
    canonical_values = {_canonical_value(value): value for value in unique_values}
    defaults = [
        original_value
        for canonical, original_value in canonical_values.items()
        if canonical in event_markers
    ]

    if defaults:
        return defaults

    return []


def _suggest_time_unit(time_col: str) -> str:
    normalized = str(time_col).lower()

    if any(keyword in normalized for keyword in ["days", "day"]):
        return "days"

    if any(keyword in normalized for keyword in ["months", "month", "os_months"]):
        return "months"

    if any(keyword in normalized for keyword in ["years", "year"]):
        return "years"

    return "unknown"


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"

    if isinstance(value, str):
        return value.strip().lower()

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, (int, float)) and not pd.isna(value):
        float_value = float(value)
        if float_value.is_integer():
            return str(int(float_value))
        return str(float_value)

    return str(value).strip().lower()


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    return str(value)


def _json_safe_config(config: SurvivalConfig) -> dict[str, Any]:
    config_dict = asdict(config)
    config_dict["event_values"] = [_json_safe_value(value) for value in config.event_values]
    config_dict["censor_values"] = [_json_safe_value(value) for value in config.censor_values]
    return config_dict


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


if __name__ == "__main__":
    main()
