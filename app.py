from dataclasses import replace
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.column_annotations import (
    MEANING_OPTIONS,
    USE_BASELINE,
    USE_CHARTS,
    USE_COLUMN_LABELS,
    USE_COX,
    USE_FILTER,
    USE_GROUP,
    USE_IGNORE,
    annotations_from_dataframe,
    annotations_to_dataframe,
    apply_survival_roles,
    build_default_annotations,
    get_annotation_summary,
    get_columns_for_use,
    sync_annotations,
)
from src.charts import (
    CHART_TYPE_LABELS,
    CHART_TYPE_OPTIONS,
    build_chart,
    explain_chart_recommendation,
    get_chart_variable_type,
    plot_missingness_heatmap,
    summarize_chart_variables,
)
from src.data_quality import (
    build_data_quality_report,
    determine_quality_status,
    issues_to_dataframe,
)
from src.cohort_overview import (
    build_baseline_table,
    compute_cohort_overview_metrics,
    count_variable_types,
    get_cohort_role_columns,
    get_default_baseline_variables,
    summarize_categorical_variable,
    summarize_continuous_variable,
)
from src.data_loading import (
    DatasetMetadata,
    read_dataset_with_metadata,
)
from src.exports import (
    build_html_report,
    build_pdf_report,
    deserialize_analysis_configuration,
    serialize_analysis_configuration,
)
from src.profiling import normalize_missing_values, profile_dataframe
from src.role_suggestions import suggest_survival_roles
from src.survival_analysis import (
    combine_curve_results,
    compute_group_survival_summary,
    compute_number_at_risk_table,
    compute_overall_survival_summary_table,
    fit_km_by_group,
    fit_km_overall,
    format_p_value,
    format_survival_time,
    generate_survival_interpretation_warnings,
    get_survival_summary,
    pivot_at_risk_table,
    pivot_survival_probability_table,
    run_logrank_test,
    run_pairwise_logrank_tests,
    suggest_timepoints,
    survival_probabilities_at_years,
    survival_probability_table_by_group,
    validate_survival_ready_dataframe,
)
from src.survival_mapping import (
    SurvivalConfig,
    create_cleaned_mapped_dataframe,
    create_survival_ready_dataframe,
    validate_survival_config,
)
from src.survival_plots import plot_km_curve
from src.upload_state import dataset_content_signature, uploaded_file_content_digest


DATASET_DERIVED_SESSION_KEYS = {
    "survival_config",
    "survival_ready_df",
    "data_quality_report",
    "cohort_group_col",
    "cohort_continuous_vars",
    "cohort_categorical_vars",
    "chart_type_label",
    "chart_x_col",
    "chart_x_col_not_applicable",
    "chart_y_col",
    "chart_y_col_not_applicable",
    "chart_color_col",
    "chart_color_col_not_applicable",
    "survival_analysis_group_col",
    "survival_timepoints",
    "group_value_labels",
    "column_annotations",
    "annotation_editor_version",
    "annotation_status_message",
    "configuration_status_message",
    "analysis_configuration_upload",
    "pending_survival_setup_config",
    "combined_report_html",
    "combined_report_pdf",
    "show_more_preview_rows",
    "survival_time_source",
    "active_survival_filter_columns",
    "active_survival_filters",
    "loaded_example_dataset",
}

DATASET_DERIVED_SESSION_PREFIXES = (
    "time_col_",
    "event_col_",
    "start_date_col_",
    "event_date_col_",
    "last_followup_date_col_",
    "id_col_",
    "group_col_",
    "event_values_",
    "censor_values_",
    "unmapped_event_handling_",
    "missing_event_handling_",
    "time_unit_",
    "survival_group_label_",
    "survival_filter_",
    "column_annotation_editor_",
)


PAGE_OPTIONS = [
    "Dataset",
    "Setup",
    "Data Quality",
    "Cohort Overview",
    "Charts",
    "Survival Analysis",
    "Export",
]
ANALYSIS_GOAL_EXPLORE = "Explore and review the dataset"
ANALYSIS_GOAL_SURVIVAL = "Run survival analysis"
EXAMPLE_DATASETS = {
    "Lung cancer cohort (recommended)": "lung.csv",
    "Rossi recidivism cohort": "rossi.csv",
}


def main() -> None:
    st.set_page_config(page_title="Medical Dataset Explorer", layout="wide")

    st.title("Medical Dataset Explorer")
    st.caption(
        "Review patient-level datasets, check data quality, and run exploratory "
        "survival analysis when follow-up data are available."
    )
    page = _render_sidebar_navigation()

    if page == "Dataset":
        _render_dataset_page()
    elif page == "Setup":
        _render_setup_page()
    elif page == "Data Quality":
        _render_data_quality_tab()
    elif page == "Cohort Overview":
        _render_cohort_overview_tab()
    elif page == "Charts":
        _render_charts_tab()
    elif page == "Survival Analysis":
        _render_survival_analysis_tab()
    elif page == "Export":
        _render_export_page()


def _render_sidebar_navigation() -> str:
    if st.session_state.get("main_tab") not in PAGE_OPTIONS:
        st.session_state["main_tab"] = "Dataset"

    st.sidebar.markdown("### Workspace")
    uploaded_df = st.session_state.get("uploaded_df")
    metadata = st.session_state.get("dataset_metadata")
    if uploaded_df is None:
        st.sidebar.caption("No dataset loaded")
        st.sidebar.markdown("○ Dataset\n\n○ Survival mapping\n\n○ Data quality")
    else:
        file_name = getattr(metadata, "file_name", "Current dataset")
        st.sidebar.markdown(f"**{escape(str(file_name))}**")
        st.sidebar.caption(f"{len(uploaded_df):,} rows · {len(uploaded_df.columns):,} columns")
        if st.session_state.get("survival_config"):
            mapping_status = "✓ Confirmed"
        elif st.session_state.get("analysis_goal") == ANALYSIS_GOAL_SURVIVAL:
            mapping_status = "○ Not configured"
        else:
            mapping_status = "○ Optional"
        quality_report = st.session_state.get("data_quality_report")
        if quality_report is None:
            quality_status = "○ Not reviewed"
        else:
            status = determine_quality_status(quality_report.get("issues", []))
            quality_status = {
                "error": "! Blocking issues",
                "warning": "! Review warnings",
                "success": "✓ Reviewed",
            }.get(status, "✓ Reviewed")
        active_filters = st.session_state.get("active_survival_filters", {})
        st.sidebar.markdown(
            "✓ Dataset loaded  \n"
            f"{mapping_status} survival mapping  \n"
            f"{quality_status} data quality  \n"
            f"{len(active_filters)} active filter{'s' if len(active_filters) != 1 else ''}"
        )

    st.sidebar.markdown("---")
    return st.sidebar.radio(
        "Go to",
        PAGE_OPTIONS,
        key="main_tab",
        label_visibility="collapsed",
    )


def _render_dataset_page() -> None:
    st.header("Dataset")
    st.write("Start by uploading a patient-level table, or load a bundled example.")

    uploaded_file = st.file_uploader(
        "Upload a dataset",
        type=["csv", "tsv", "txt", "xlsx"],
        help="Supported formats: CSV, TSV, TXT, XLSX",
    )
    st.caption("Supported formats: CSV, TSV, TXT, XLSX")

    with st.expander("Try an example dataset"):
        example_label = st.selectbox("Example dataset", list(EXAMPLE_DATASETS))
        st.caption(
            "The lung cohort is the clearest first walkthrough: time and event columns "
            "are detected automatically."
        )
        if st.button("Load example dataset"):
            _load_example_dataset(EXAMPLE_DATASETS[example_label])
            st.rerun()

    if uploaded_file is not None:
        try:
            content_digest = uploaded_file_content_digest(uploaded_file)
            load_result = read_dataset_with_metadata(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
        else:
            dataset_replaced = _sync_uploaded_dataset_state(
                uploaded_file.name,
                load_result.dataframe,
                content_digest=content_digest,
            )
            st.session_state["dataset_metadata"] = load_result.metadata
            if dataset_replaced:
                st.info(
                    "A different dataset was detected. Previous survival mapping, "
                    "annotations, and analysis selections were reset."
                )
            st.success("Dataset loaded successfully.")

    df = st.session_state.get("uploaded_df")
    profile = st.session_state.get("profile_df")
    metadata = st.session_state.get("dataset_metadata")
    if df is None or profile is None or metadata is None:
        st.info("Upload a CSV, TSV, TXT, or Excel file to begin.")
        return

    _render_dataset_summary(metadata, df)
    goal_options = [ANALYSIS_GOAL_EXPLORE, ANALYSIS_GOAL_SURVIVAL]
    saved_goal = st.session_state.get("analysis_goal", ANALYSIS_GOAL_EXPLORE)
    if st.session_state.get("analysis_goal_selector") not in goal_options:
        st.session_state["analysis_goal_selector"] = saved_goal
    analysis_goal = st.radio(
        "What would you like to do?",
        goal_options,
        horizontal=True,
        key="analysis_goal_selector",
        on_change=_sync_analysis_goal,
    )
    st.session_state["analysis_goal"] = analysis_goal
    if analysis_goal == ANALYSIS_GOAL_SURVIVAL:
        st.caption("Next, confirm the follow-up time and event meaning on Setup.")
    else:
        st.caption("Survival mapping is optional. You can go directly to data quality and charts.")

    st.subheader("Preview")
    _render_dataset_preview(df, profile)

    with st.expander("Column profile and type detection"):
        st.caption(
            "Technical profiling details are useful when a column type or recommendation "
            "looks incorrect."
        )
        st.dataframe(profile, width="stretch")

    next_page = "Setup" if analysis_goal == ANALYSIS_GOAL_SURVIVAL else "Data Quality"
    next_label = (
        "Continue to Survival Setup"
        if analysis_goal == ANALYSIS_GOAL_SURVIVAL
        else "Continue to Data Quality"
    )
    st.button(
        next_label,
        type="primary",
        on_click=_navigate_to_main_tab,
        args=(next_page,),
    )


def _load_example_dataset(file_name: str) -> None:
    example_path = Path(__file__).resolve().parent / "datasets" / file_name
    content = example_path.read_bytes()
    uploaded = BytesIO(content)
    uploaded.name = file_name
    load_result = read_dataset_with_metadata(uploaded)
    _sync_uploaded_dataset_state(
        file_name,
        load_result.dataframe,
        content_digest=uploaded_file_content_digest(uploaded),
    )
    st.session_state["dataset_metadata"] = load_result.metadata
    st.session_state["analysis_goal"] = ANALYSIS_GOAL_SURVIVAL
    st.session_state["analysis_goal_selector"] = ANALYSIS_GOAL_SURVIVAL
    st.session_state["loaded_example_dataset"] = file_name
    if file_name == "lung.csv":
        st.session_state["time_unit_time"] = "days"


def _sync_analysis_goal() -> None:
    selected = st.session_state.get("analysis_goal_selector")
    if selected in {ANALYSIS_GOAL_EXPLORE, ANALYSIS_GOAL_SURVIVAL}:
        st.session_state["analysis_goal"] = selected


def _render_setup_page() -> None:
    st.header("Setup")
    st.caption("Confirm only the information needed for your analysis. Advanced column settings are optional.")
    df = st.session_state.get("uploaded_df")
    profile = st.session_state.get("profile_df")
    if df is None or profile is None:
        st.info("Load a dataset before configuring analysis roles.")
        st.button(
            "Go to Dataset",
            type="primary",
            on_click=_navigate_to_main_tab,
            args=("Dataset",),
        )
        return

    goal = st.session_state.get("analysis_goal", ANALYSIS_GOAL_EXPLORE)
    if goal == ANALYSIS_GOAL_SURVIVAL or st.session_state.get("survival_config") is not None:
        st.subheader("Survival mapping")
        st.write(
            "Confirm how follow-up time and event status are represented. "
            "Recommended columns are preselected, but their clinical meaning must be verified."
        )
        _render_survival_setup(df, profile)
    else:
        st.info(
            "Survival setup is skipped for the current goal. Data Quality, Cohort Overview, "
            "and Charts are ready to use."
        )
        action_columns = st.columns(2)
        action_columns[0].button(
            "Continue to Data Quality",
            type="primary",
            on_click=_navigate_to_main_tab,
            args=("Data Quality",),
        )
        if action_columns[1].button("Enable survival analysis"):
            st.session_state["analysis_goal"] = ANALYSIS_GOAL_SURVIVAL
            st.rerun()

    with st.expander("Advanced: column meanings and analysis uses"):
        _render_column_annotations(df, profile)


def _render_export_page() -> None:
    st.header("Export")
    st.caption("Download the current dataset, reusable configuration, or a combined report.")
    df = st.session_state.get("uploaded_df")
    profile = st.session_state.get("profile_df")
    if df is None or profile is None:
        st.info("Load a dataset before exporting results.")
        st.button(
            "Go to Dataset",
            type="primary",
            on_click=_navigate_to_main_tab,
            args=("Dataset",),
        )
        return
    _render_export_section(df, profile)


def _render_dataset_summary(metadata: DatasetMetadata, df: pd.DataFrame) -> None:
    summary_columns = st.columns(4)
    summary_columns[0].metric("File", metadata.file_name)
    summary_columns[1].metric("Size", _format_file_size(metadata.file_size_bytes))
    summary_columns[2].metric("Rows", len(df))
    summary_columns[3].metric("Columns", len(df.columns))

    delimiter = (
        "Not applicable"
        if metadata.format_name == "Excel"
        else _format_delimiter(metadata.delimiter)
    )
    encoding = metadata.encoding or "Not applicable"
    st.markdown(
        f"**Detected format:** {metadata.format_name}  \n"
        f"**Detected delimiter:** {delimiter}  \n"
        f"**Detected encoding:** {encoding}"
    )


def _render_dataset_preview(df: pd.DataFrame, profile: pd.DataFrame) -> None:
    initial_row_limit = 20
    expanded_row_limit = 100
    can_expand = len(df) > initial_row_limit
    show_more = st.checkbox(
        "Show more rows",
        value=False,
        key="show_more_preview_rows",
        disabled=not can_expand,
        help=f"Show up to {expanded_row_limit} rows instead of {initial_row_limit}.",
    )
    row_limit = expanded_row_limit if show_more and can_expand else initial_row_limit
    preview, missing_mask = _prepare_dataset_preview(df, profile, row_limit)
    styles = pd.DataFrame(
        "",
        index=preview.index,
        columns=preview.columns,
    )
    styles[missing_mask] = "background-color: rgba(255, 75, 75, 0.25)"
    styled_preview = preview.style.apply(lambda _: styles, axis=None)

    st.dataframe(
        styled_preview,
        hide_index=True,
        width="stretch",
        height=min(800, 38 + 35 * max(len(preview), 1)),
    )
    st.caption(f"Showing {len(preview)} of {len(df)} rows. Column types appear in the headers.")
    if bool(missing_mask.to_numpy().any()):
        st.caption("Missing values are highlighted in red.")


def _prepare_dataset_preview(
    df: pd.DataFrame,
    profile: pd.DataFrame,
    row_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preview = df.head(max(0, row_limit)).copy(deep=True)
    detected_types = profile["detected_type"].astype(str).tolist()
    preview.columns = [
        f"{column} · {detected_types[index] if index < len(detected_types) else 'unknown'}"
        for index, column in enumerate(preview.columns)
    ]
    missing_mask = normalize_missing_values(preview).isna()
    return preview, missing_mask


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _format_delimiter(delimiter: str | None) -> str:
    return {
        None: "Not detected (single column)",
        ",": "Comma (,)",
        "\t": "Tab",
        ";": "Semicolon (;)",
        "|": "Pipe (|)",
    }.get(delimiter, repr(delimiter))


def _sync_uploaded_dataset_state(
    file_name: str,
    df: pd.DataFrame,
    *,
    content_digest: str | None = None,
) -> bool:
    """Synchronize upload state and report whether an existing dataset was replaced."""
    dataset_signature = dataset_content_signature(
        file_name,
        df,
        content_digest=content_digest,
    )
    previous_signature = st.session_state.get("uploaded_dataset_signature")
    dataset_changed = previous_signature != dataset_signature
    dataset_replaced = previous_signature is not None and dataset_changed

    if dataset_changed:
        _invalidate_dataset_derived_state()

    st.session_state["uploaded_dataset_signature"] = dataset_signature
    st.session_state["uploaded_df"] = df.copy(deep=True)
    profile_df = profile_dataframe(df)
    st.session_state["profile_df"] = profile_df
    st.session_state["column_annotations"] = sync_annotations(
        st.session_state.get("column_annotations"),
        df,
        profile_df,
        st.session_state.get("survival_config"),
    )
    return dataset_replaced


def _invalidate_dataset_derived_state() -> None:
    for key in list(st.session_state):
        if key in DATASET_DERIVED_SESSION_KEYS or key.startswith(
            DATASET_DERIVED_SESSION_PREFIXES
        ):
            st.session_state.pop(key, None)


def _render_data_quality_tab() -> None:
    st.header("Data Quality")
    st.caption("Start with the issues that can change interpretation; open detailed diagnostics only when needed.")

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
        annotations=st.session_state.get("column_annotations"),
    )
    st.session_state["data_quality_report"] = report

    _render_quality_status(report["issues"])
    _render_quality_summary_cards(report)
    _render_quality_issues(report["issues"])
    with st.expander("Detailed diagnostics"):
        _render_missingness_section(report, uploaded_df)
        _render_duplicate_checks(report)
        _render_clinical_value_checks(report)
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
    duplicate_ids = report["duplicate_ids"]
    age_quality = report["age_quality"]
    survival_quality = report["survival_quality"]

    first_row = st.columns(3)
    first_row[0].metric("Missing cells", f"{overview['missing_percent']}%")
    first_row[1].metric(
        "Duplicate IDs",
        duplicate_ids["duplicate_id_value_count"] if duplicate_ids["checked"] else "Not checked",
    )
    first_row[2].metric(
        "Invalid ages",
        age_quality["invalid_age_count"] if age_quality["checked"] else "Not detected",
    )

    second_row = st.columns(3)
    if survival_quality.get("has_mapping"):
        second_row[0].metric(
            "Invalid event values",
            survival_quality["unmapped_event_value_count"],
        )
        second_row[1].metric("Zero follow-up rows", survival_quality["zero_time_count"])
        second_row[2].metric(
            "Analysis-ready rows",
            f"{survival_quality['usable_survival_rows']} / {survival_quality['raw_rows']}",
        )
    else:
        second_row[0].metric("Invalid event values", "Not mapped")
        second_row[1].metric("Zero follow-up rows", "Not mapped")
        second_row[2].metric("Analysis-ready rows", "Not mapped")


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


def _render_missingness_section(report: dict[str, Any], df: pd.DataFrame) -> None:
    st.subheader("Missingness by column")
    missingness_by_column = report["missingness_by_column"]
    st.dataframe(missingness_by_column, hide_index=True, width="stretch")

    if not df.empty and len(df.columns):
        st.plotly_chart(plot_missingness_heatmap(df), width="stretch")
        if len(df) > 250:
            st.caption("Heatmap shows the first 250 dataset rows; the table covers all rows.")

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
        ],
        dtype=str,
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
        ],
        dtype=str,
    )
    st.dataframe(duplicate_id_summary, hide_index=True, width="stretch")


def _render_clinical_value_checks(report: dict[str, Any]) -> None:
    st.subheader("Age and date checks")
    age_quality = report["age_quality"]
    date_quality = report["date_quality"]

    st.markdown("**Age range (0–120)**")
    if age_quality["checked"]:
        st.dataframe(age_quality["details"], hide_index=True, width="stretch")
    else:
        st.info("No age column was identified. Annotate a column as Age to check it.")

    st.markdown("**Date parsing**")
    if date_quality["checked"]:
        st.dataframe(date_quality["parsing"], hide_index=True, width="stretch")
    else:
        st.info("No date columns were identified.")

    st.markdown("**Date consistency**")
    if date_quality["consistency"].empty:
        st.info(
            "Select survival date columns, or annotate start and end dates, "
            "to run chronology checks."
        )
    else:
        st.dataframe(date_quality["consistency"], hide_index=True, width="stretch")


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
            {
                "Metric": "Unmapped value handling",
                "Value": _format_event_handling(
                    survival_quality["unmapped_event_handling"]
                ),
            },
            {
                "Metric": "Missing event handling",
                "Value": _format_event_handling(
                    survival_quality["missing_event_handling"]
                ),
            },
            {"Metric": "Missing time values", "Value": survival_quality["time_missing_count"]},
            {"Metric": "Missing event values", "Value": survival_quality["event_missing_count"]},
            {"Metric": "Negative time values", "Value": survival_quality["negative_time_count"]},
            {"Metric": "Zero time values", "Value": survival_quality["zero_time_count"]},
            {
                "Metric": "Unmapped event values",
                "Value": ", ".join(survival_quality["unmapped_event_values"]) or "None",
            },
        ],
        dtype=str,
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
    st.header("Cohort Overview")
    st.caption("Understand who is in the current cohort before building a detailed baseline table.")

    uploaded_df = st.session_state.get("uploaded_df")
    profile_df = st.session_state.get("profile_df")
    survival_config = st.session_state.get("survival_config")
    survival_ready_df = st.session_state.get("survival_ready_df")
    annotations = st.session_state.get("column_annotations")

    if uploaded_df is None:
        st.info("No dataset uploaded yet.\n\nUpload a dataset first to view cohort summaries.")
        return

    time_unit = getattr(survival_config, "time_unit", "unknown") if survival_config is not None else "unknown"
    role_columns = get_cohort_role_columns(uploaded_df, annotations)
    configured_id_col = getattr(survival_config, "id_col", None)
    id_col = (
        configured_id_col
        if configured_id_col in uploaded_df.columns
        else next(iter(role_columns["patient_id"]), None)
    )
    age_col = next(iter(role_columns["age"]), None)
    metrics = compute_cohort_overview_metrics(
        uploaded_df,
        survival_ready_df=survival_ready_df,
        time_unit=time_unit,
        id_col=id_col,
        age_col=age_col,
    )
    inferred_variables = get_default_baseline_variables(uploaded_df, profile_df, survival_config)
    default_variables = _annotated_baseline_variables(
        uploaded_df,
        inferred_variables,
        annotations,
    )
    variable_type_summary = count_variable_types(uploaded_df, profile_df, survival_config)

    _render_cohort_summary_cards(metrics)
    _render_key_cohort_characteristics(
        uploaded_df,
        profile_df,
        role_columns,
    )
    _render_cohort_mapping_summary(survival_config)

    with st.expander("Advanced: variable type summary"):
        st.dataframe(variable_type_summary, hide_index=True, width="stretch")

    with st.expander("Build a baseline characteristics table"):
        st.markdown("**Variables to include**")
        group_col = _render_cohort_group_selector(
            uploaded_df,
            annotations,
            survival_config,
        )
        continuous_vars, categorical_vars, max_levels, include_missing = _render_baseline_table_controls(
            default_variables,
        )

        overlapping_vars = sorted(set(continuous_vars) & set(categorical_vars))
        if overlapping_vars:
            st.warning(
                "Variables selected as both continuous and categorical will be summarized as categorical: "
                + ", ".join(overlapping_vars)
            )
            continuous_vars = [column for column in continuous_vars if column not in overlapping_vars]

        if group_col is not None and (
            group_col in continuous_vars or group_col in categorical_vars
        ):
            st.caption(
                f"`{group_col}` defines the Table 1 columns and is not repeated as a characteristic."
            )
            continuous_vars = [column for column in continuous_vars if column != group_col]
            categorical_vars = [column for column in categorical_vars if column != group_col]

        st.markdown("**Baseline characteristics**")
        if not continuous_vars and not categorical_vars:
            st.info("No characteristics selected; Table 1 still includes cohort and survival rows.")

        grouped_survival_df = (
            _survival_dataframe_for_group(
                uploaded_df,
                survival_config,
                survival_ready_df,
                group_col,
            )
            if survival_config is not None and survival_ready_df is not None
            else survival_ready_df
        )

        baseline_table = build_baseline_table(
            uploaded_df,
            continuous_vars,
            categorical_vars,
            group_col=group_col,
            max_levels=max_levels,
            include_missing=include_missing,
            group_value_labels=_group_value_labels_for_column(group_col),
            survival_ready_df=grouped_survival_df,
            id_col=id_col,
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
    first_row[0].metric("Total patients", metrics["n_patients"])
    first_row[1].metric("Rows", metrics["n_rows"])
    first_row[2].metric("Events", _metric_value(metrics["events"]))
    first_row[3].metric("Censored", _metric_value(metrics["censored"]))

    second_row = st.columns(4)
    second_row[0].metric(
        "Event rate",
        "Not available" if metrics["event_rate"] is None else f"{_format_number(metrics['event_rate'])}%",
    )
    second_row[1].metric(
        "Median follow-up",
        _format_time_value(metrics["median_followup"], metrics["time_unit"]),
    )
    second_row[2].metric(
        "Median age",
        (
            "Not available"
            if metrics["median_age"] is None
            else _format_number(metrics["median_age"])
        ),
    )
    second_row[3].metric(
        "Complete rows",
        f"{metrics['complete_rows']} ({_format_number(metrics['complete_rows_percent'])}%)",
    )

    st.caption(
        f"Patient count: {metrics['patient_count_basis']}. "
        f"Missing cells: {_format_number(metrics['missing_cells_percent'])}%."
    )
    if metrics["missing_patient_ids"]:
        st.warning(
            f"{metrics['missing_patient_ids']} row(s) have no patient ID and are excluded "
            "from the distinct patient count."
        )


def _render_key_cohort_characteristics(
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None,
    role_columns: dict[str, list[str]],
) -> None:
    st.subheader("Key cohort characteristics")
    st.caption("These summaries follow the meanings saved in Column annotations.")
    rendered = False

    for column in role_columns["age"]:
        rendered = True
        with st.expander(f"Age: {column}", expanded=True):
            summary = summarize_continuous_variable(df, column)
            summary_row = st.columns(4)
            summary_row[0].metric("Valid values", summary["n"])
            summary_row[1].metric("Median", _metric_value(summary["median"]))
            summary_row[2].metric("Mean", _metric_value(summary["mean"]))
            summary_row[3].metric(
                "Range",
                (
                    "Not available"
                    if summary["min"] is None
                    else f"{_format_number(summary['min'])}–{_format_number(summary['max'])}"
                ),
            )
            result = build_chart(
                df,
                chart_type="histogram",
                x_col=column,
                profile_df=profile_df,
            )
            if result["fig"] is not None:
                st.plotly_chart(result["fig"], width="stretch")
            for warning in result["warnings"]:
                st.warning(warning)

    categorical_roles = [
        ("sex", "Sex / gender"),
        ("diagnosis", "Diagnosis"),
        ("treatment", "Treatment"),
        ("outcome", "Outcome"),
    ]
    for role, label in categorical_roles:
        for column in role_columns[role]:
            rendered = True
            with st.expander(f"{label}: {column}"):
                summary = summarize_categorical_variable(
                    df,
                    column,
                    max_levels=20,
                    include_missing=True,
                )
                st.dataframe(
                    summary[["level", "count", "percent"]].rename(
                        columns={"level": "Level", "count": "n", "percent": "%"}
                    ),
                    hide_index=True,
                    width="stretch",
                )
                result = build_chart(
                    df,
                    chart_type="bar",
                    x_col=column,
                    profile_df=profile_df,
                    max_category_levels=20,
                    include_missing=True,
                )
                if result["fig"] is not None:
                    st.plotly_chart(result["fig"], width="stretch")
                for warning in result["warnings"]:
                    st.warning(warning)

    if not rendered:
        st.info(
            "No age, sex/gender, diagnosis, treatment, or outcome columns are annotated. "
            "Update Column annotations in Setup to add these summaries."
        )


def _render_cohort_mapping_summary(survival_config: SurvivalConfig | None) -> None:
    if survival_config is None:
        st.info(
            "Survival mapping has not been confirmed yet. Cohort summaries are still available, "
            "but survival-specific metrics are disabled."
        )
        return

    st.markdown("**Survival mapping**")
    if survival_config.time_source == "dates":
        mapping_rows = [
            {"Field": "time", "Value": "Derived from dates"},
            {"Field": "start date", "Value": survival_config.start_date_col},
            {"Field": "event date", "Value": survival_config.event_date_col},
            {
                "Field": "last follow-up date",
                "Value": survival_config.last_followup_date_col,
            },
            {
                "Field": "missing event dates",
                "Value": (
                    "Censor at last follow-up"
                    if survival_config.missing_event_handling == "treat_as_censored"
                    else "Exclude"
                ),
            },
            {"Field": "group", "Value": survival_config.group_col or "None"},
            {"Field": "time unit", "Value": survival_config.time_unit},
        ]
    else:
        mapping_rows = [
            {"Field": "time", "Value": survival_config.time_col},
            {"Field": "event", "Value": survival_config.event_col},
            {
                "Field": "event values",
                "Value": ", ".join(
                    _format_value(value)
                    for value in survival_config.event_values
                ),
            },
            {
                "Field": "censored values",
                "Value": ", ".join(
                    _format_value(value)
                    for value in survival_config.censor_values
                ),
            },
            {
                "Field": "unmapped values",
                "Value": _format_event_handling(
                    getattr(survival_config, "unmapped_event_handling", "exclude")
                ),
            },
            {
                "Field": "missing event values",
                "Value": _format_event_handling(
                    survival_config.missing_event_handling
                ),
            },
            {"Field": "group", "Value": survival_config.group_col or "None"},
            {"Field": "time unit", "Value": survival_config.time_unit},
        ]
    mapping_summary = pd.DataFrame(
        mapping_rows
    )
    st.dataframe(mapping_summary, hide_index=True, width="stretch")


def _render_cohort_group_selector(
    df: pd.DataFrame,
    annotations: Any,
    survival_config: SurvivalConfig | None,
) -> str | None:
    all_columns = [str(column) for column in df.columns]
    group_columns = get_columns_for_use(annotations, USE_GROUP, all_columns)
    options = ["No grouping"] + group_columns
    default_group = getattr(survival_config, "group_col", None) if survival_config is not None else None
    default_option = default_group if default_group in options else "No grouping"

    if st.session_state.get("cohort_group_col") not in options:
        st.session_state["cohort_group_col"] = default_option

    selected = st.selectbox(
        "Group by",
        options,
        key="cohort_group_col",
    )
    if not group_columns:
        st.caption("No columns are annotated for grouping. Update Column annotations in Setup.")
    return None if selected == "No grouping" else str(selected)


def _render_baseline_table_controls(
    default_variables: dict[str, list[str]],
) -> tuple[list[str], list[str], int, bool]:
    all_columns = list(
        dict.fromkeys(
            default_variables.get("continuous", [])
            + default_variables.get("categorical", [])
        )
    )
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
    if not all_columns:
        st.caption(
            "No columns are annotated for the baseline table. "
            "Update Column annotations in Setup."
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


def _annotated_baseline_variables(
    df: pd.DataFrame,
    inferred_variables: dict[str, list[str]],
    annotations: Any,
) -> dict[str, list[str]]:
    selected = get_columns_for_use(annotations, USE_BASELINE, df.columns)
    inferred_continuous = set(inferred_variables.get("continuous", []))
    continuous = [column for column in selected if column in inferred_continuous]
    categorical = [column for column in selected if column not in inferred_continuous]
    return {
        "continuous": continuous,
        "categorical": categorical,
        "excluded": [
            str(column)
            for column in df.columns
            if str(column) not in selected
        ],
    }


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


def _render_charts_tab() -> None:
    st.header("Charts")
    st.caption("Choose variables first. Auto mode recommends a compatible chart from their detected types.")

    uploaded_df = st.session_state.get("uploaded_df")
    profile_df = st.session_state.get("profile_df")
    annotations = st.session_state.get("column_annotations")

    if uploaded_df is None:
        st.info("No dataset uploaded yet.\n\nUpload a dataset first to create charts.")
        return

    all_columns = get_columns_for_use(
        annotations,
        USE_CHARTS,
        uploaded_df.columns,
    )
    if not all_columns:
        st.warning(
            "No columns are annotated for charts. "
            "Update Column annotations in Setup."
        )
        return

    chart_df = uploaded_df.loc[:, all_columns]
    variable_types = {
        column: get_chart_variable_type(column, chart_df, profile_df)
        for column in all_columns
    }
    chartable_variables = [
        column
        for column, variable_type in variable_types.items()
        if variable_type in {"numeric", "categorical"}
    ]
    if not chartable_variables:
        st.warning("No chartable variables found.")

    variable_options = ["None"] + all_columns
    chart_type_labels = list(CHART_TYPE_OPTIONS.keys())
    _set_default_session_value("chart_type_label", "Auto", chart_type_labels)
    _set_default_session_value(
        "chart_x_col",
        _default_chart_x_column(variable_types) or "None",
        variable_options,
    )
    _set_default_session_value("chart_y_col", "None", variable_options)
    _set_default_session_value("chart_color_col", "None", variable_options)

    control_row = st.columns(2)
    chart_type_label = control_row[0].selectbox(
        "Chart type",
        chart_type_labels,
        key="chart_type_label",
    )
    chart_type = CHART_TYPE_OPTIONS[chart_type_label]
    axis_controls_disabled = chart_type in {"correlation_heatmap", "missingness_bar"}
    if axis_controls_disabled:
        x_choice = control_row[1].selectbox(
            "X variable",
            ["N/A"],
            key="chart_x_col_not_applicable",
            disabled=True,
        )
    else:
        x_choice = control_row[1].selectbox(
            "X variable",
            variable_options,
            key="chart_x_col",
        )

    variable_row = st.columns(2)
    if axis_controls_disabled:
        y_choice = variable_row[0].selectbox(
            "Y variable",
            ["N/A"],
            key="chart_y_col_not_applicable",
            disabled=True,
        )
        color_choice = variable_row[1].selectbox(
            "Color/group variable",
            ["N/A"],
            key="chart_color_col_not_applicable",
            disabled=True,
        )
    else:
        y_choice = variable_row[0].selectbox(
            "Y variable",
            variable_options,
            key="chart_y_col",
        )
        color_choice = variable_row[1].selectbox(
            "Color/group variable",
            variable_options,
            key="chart_color_col",
        )
    if axis_controls_disabled:
        st.caption(f"{chart_type_label} uses the full dataset, so X, Y, and color variables are disabled.")

    with st.expander("Advanced chart options"):
        option_row = st.columns(3)
        max_category_levels = int(
            option_row[0].slider(
                "Maximum categorical levels",
                min_value=3,
                max_value=50,
                value=20,
            )
        )
        include_missing = bool(
            option_row[1].checkbox(
                "Include missing values in categorical charts",
                value=True,
            )
        )
        normalize = bool(
            option_row[2].checkbox(
                "Normalize stacked bar chart to percentages",
                value=False,
            )
        )

    x_col = None if axis_controls_disabled else _none_option_to_value(x_choice)
    y_col = None if axis_controls_disabled else _none_option_to_value(y_choice)
    color_col = None if axis_controls_disabled else _none_option_to_value(color_choice)

    result = build_chart(
        chart_df,
        chart_type=chart_type,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        profile_df=profile_df,
        max_category_levels=max_category_levels,
        include_missing=include_missing,
        normalize=normalize,
    )

    resolved_chart_type = result["chart_type"]
    if chart_type == "auto":
        st.info(
            f"Suggested chart: {CHART_TYPE_LABELS.get(resolved_chart_type, resolved_chart_type)}\n\n"
            f"Reason: {explain_chart_recommendation(resolved_chart_type, x_col, y_col, chart_df, profile_df)}"
        )

    _render_chart_survival_mapping_note(
        [x_col, y_col, color_col],
        st.session_state.get("survival_config"),
    )

    for warning in result["warnings"]:
        st.warning(warning)

    fig = result["fig"]
    if fig is None:
        st.warning("Select compatible variables to create a chart.")
        return

    summaries = summarize_chart_variables(
        chart_df,
        [x_col, y_col, color_col],
        profile_df,
    )
    if summaries:
        chart_column, summary_column = st.columns([3, 1])
        with chart_column:
            _render_plotly_chart_with_image_export(
                fig,
                key="chart_plot_image_format",
                filename="dashboard_chart",
            )
            st.download_button(
                "Download chart as HTML",
                data=fig.to_html(include_plotlyjs="cdn"),
                file_name="chart.html",
                mime="text/html",
            )
        with summary_column:
            st.subheader("Summary statistics")
            for column, statistics in summaries.items():
                st.caption(column)
                st.dataframe(
                    pd.DataFrame(
                        statistics.items(),
                        columns=["Statistic", "Value"],
                    ),
                    hide_index=True,
                    width="stretch",
                )
    else:
        _render_plotly_chart_with_image_export(
            fig,
            key="chart_plot_image_format",
            filename="dashboard_chart",
        )
        st.download_button(
            "Download chart as HTML",
            data=fig.to_html(include_plotlyjs="cdn"),
            file_name="chart.html",
            mime="text/html",
        )


def _default_chart_x_column(variable_types: dict[str, str]) -> str | None:
    for variable_type in ["numeric", "categorical"]:
        for column, detected_type in variable_types.items():
            if detected_type == variable_type:
                return column
    return None


def _set_default_session_value(key: str, default_value: str, valid_options: list[str]) -> None:
    if st.session_state.get(key) not in valid_options:
        st.session_state[key] = default_value if default_value in valid_options else valid_options[0]


def _render_plotly_chart_with_image_export(
    fig: Any,
    *,
    key: str,
    filename: str,
) -> None:
    image_format = st.radio(
        "Plot image download format",
        ["PNG", "SVG"],
        horizontal=True,
        key=key,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": image_format.lower(),
                "filename": filename,
            },
        },
    )
    st.caption(
        f"Download this plot as {image_format} with the camera button in the plot toolbar."
    )


def _none_option_to_value(value: str) -> str | None:
    return None if value == "None" else value


def _render_chart_survival_mapping_note(
    selected_columns: list[str | None],
    survival_config: SurvivalConfig | None,
) -> None:
    if survival_config is None:
        return

    survival_roles = {
        survival_config.time_col: "time",
        survival_config.event_col: "event",
        survival_config.start_date_col: "start date",
        survival_config.event_date_col: "event date",
        survival_config.last_followup_date_col: "last follow-up date",
        survival_config.id_col: "patient ID",
        survival_config.group_col: "group",
    }
    selected_survival_columns = [
        f"{column} ({survival_roles[column]})"
        for column in selected_columns
        if column is not None and column in survival_roles and survival_roles[column] is not None
    ]
    if selected_survival_columns:
        st.info("Selected variable is used in survival mapping: " + ", ".join(selected_survival_columns))


def _render_survival_analysis_tab() -> None:
    st.header("Survival Analysis")
    st.caption("Review the current cohort, then interpret the Kaplan–Meier curve and group comparison.")

    config = st.session_state.get("survival_config")
    survival_ready_df = st.session_state.get("survival_ready_df")
    uploaded_df = st.session_state.get("uploaded_df")
    profile_df = st.session_state.get("profile_df")
    annotations = st.session_state.get("column_annotations")

    if config is None or survival_ready_df is None:
        st.info(
            "No survival mapping has been confirmed yet.\n\n"
            "Go to Setup and confirm the required survival columns first."
        )
        st.button(
            "Go to Setup",
            type="primary",
            on_click=_navigate_to_main_tab,
            args=("Setup",),
        )
        return

    with st.expander("Current survival mapping"):
        _render_current_mapping(config)

    errors, validation_warnings = validate_survival_ready_dataframe(survival_ready_df)
    if errors:
        st.error("Survival analysis cannot be run because:")
        for error in errors:
            st.markdown(f"- {error}")
        return

    filtered_uploaded_df = _render_survival_filters(uploaded_df, profile_df, annotations)

    st.markdown("**Plot controls**")
    control_columns = st.columns(2)
    show_ci = control_columns[0].checkbox("Show confidence interval", value=True)
    show_censors = control_columns[1].checkbox("Show censor marks", value=True)
    group_columns = get_columns_for_use(
        annotations,
        USE_GROUP,
        uploaded_df.columns if uploaded_df is not None else [],
    )
    group_options = ["No grouping"] + group_columns
    default_group = config.group_col if config.group_col in group_columns else "No grouping"
    _set_default_session_value(
        "survival_analysis_group_col",
        default_group,
        group_options,
    )
    selected_group_option = st.selectbox(
        "Group / stratification variable",
        group_options,
        key="survival_analysis_group_col",
    )
    analysis_group_col = (
        None if selected_group_option == "No grouping" else str(selected_group_option)
    )
    survival_ready_df = _survival_dataframe_for_group(
        filtered_uploaded_df,
        config,
        survival_ready_df,
        analysis_group_col,
    )
    filtered_errors, filtered_warnings = validate_survival_ready_dataframe(survival_ready_df)
    if filtered_errors:
        st.warning("No analyzable rows remain after applying the current filters.")
        return
    st.success("Survival-ready data validated.")

    has_group = analysis_group_col is not None and "_group" in survival_ready_df.columns
    use_group = has_group
    if not group_columns:
        st.caption("No columns are annotated for grouping. Update Column annotations in Setup.")

    summary = get_survival_summary(survival_ready_df, config.time_unit)
    overall_result = fit_km_overall(survival_ready_df)

    group_value_labels = st.session_state.get("group_value_labels")
    if use_group and has_group and analysis_group_col is not None:
        _render_survival_group_label_editor(survival_ready_df, analysis_group_col)
        group_value_labels = st.session_state.get("group_value_labels")

    plot_results = [overall_result]
    plot_title = "Kaplan-Meier Survival Curve"
    group_warnings: list[str] = []
    group_results: list[dict[str, Any]] = []

    if use_group and has_group:
        group_results, group_warnings = fit_km_by_group(
            survival_ready_df,
            group_value_labels=group_value_labels,
            original_group_col=analysis_group_col,
        )
        if group_results:
            plot_results = group_results
            plot_title = "Grouped Kaplan-Meier Survival Curve"

    logrank_result = run_logrank_test(survival_ready_df) if use_group and has_group else None
    logrank_warnings = logrank_result.get("warnings", []) if logrank_result is not None else []
    interpretation_warnings = generate_survival_interpretation_warnings(
        survival_ready_df,
        group_col="_group" if use_group and has_group else None,
    )
    _render_survival_warning_section(
        validation_warnings
        + filtered_warnings
        + group_warnings
        + interpretation_warnings
        + list(logrank_warnings)
    )

    _render_survival_summary_metrics(
        summary,
        overall_result["median_survival"],
        config.time_unit,
        survival_probabilities_at_years(overall_result["kmf"], config.time_unit),
        logrank_result,
    )

    curve_df = combine_curve_results(plot_results)
    fig = plot_km_curve(
        curve_df,
        title=plot_title,
        time_unit=config.time_unit,
        show_ci=show_ci,
        show_censors=show_censors,
    )
    _render_plotly_chart_with_image_export(
        fig,
        key="survival_plot_image_format",
        filename="kaplan_meier_curve",
    )

    overall_summary_table = compute_overall_survival_summary_table(survival_ready_df, config.time_unit)
    with st.expander("Overall survival summary table"):
        st.dataframe(
            _format_survival_summary_display_table(overall_summary_table, config.time_unit),
            hide_index=True,
            width="stretch",
        )
        _render_dataframe_download(
            "Download overall survival summary as CSV",
            overall_summary_table,
            "overall_survival_summary.csv",
        )

    if use_group and has_group:
        st.subheader("Group-wise survival summary")
        group_summary = compute_group_survival_summary(
            survival_ready_df,
            time_unit=config.time_unit,
            group_value_labels=group_value_labels,
            original_group_col=analysis_group_col,
        )
        if group_summary.empty:
            st.info("Grouped summary is unavailable because no grouping column was selected.")
        else:
            st.dataframe(
                _format_survival_summary_display_table(group_summary, config.time_unit),
                hide_index=True,
                width="stretch",
            )
            _render_dataframe_download(
                "Download group-wise survival summary as CSV",
                group_summary,
                "group_survival_summary.csv",
            )
    else:
        st.info("Grouped summary is unavailable because no grouping column was selected.")

    if use_group and has_group and logrank_result is not None:
        _render_logrank_section(logrank_result)

        pairwise_df = run_pairwise_logrank_tests(
            survival_ready_df,
            group_value_labels=group_value_labels,
            original_group_col=analysis_group_col,
        )
        if not pairwise_df.empty:
            st.subheader("Pairwise log-rank tests")
            st.warning(
                "Pairwise tests are exploratory and are not adjusted for multiple comparisons in this version."
            )
            st.dataframe(pairwise_df, hide_index=True, width="stretch")
            _render_dataframe_download(
                "Download pairwise log-rank tests as CSV",
                pairwise_df,
                "pairwise_logrank_tests.csv",
            )

    st.subheader("Selected time points")
    suggested_timepoints = suggest_timepoints(summary["max_followup"], config.time_unit)
    default_timepoint_text = ", ".join(_format_number(timepoint) for timepoint in suggested_timepoints)
    timepoint_text = st.text_input(
        "Time points",
        default_timepoint_text,
        key="survival_timepoints",
    )
    timepoints, parse_warning = _parse_timepoints(timepoint_text, suggested_timepoints)

    if parse_warning:
        st.warning(parse_warning)

    if not timepoints:
        st.info("No time points are available for this follow-up range.")
        return

    st.subheader("Number at risk")
    overall_at_risk = compute_number_at_risk_table(survival_ready_df, timepoints, group_col=None)
    if use_group and has_group:
        group_at_risk = compute_number_at_risk_table(
            survival_ready_df,
            timepoints,
            group_col="_group",
            group_value_labels=group_value_labels,
            original_group_col=analysis_group_col,
        )
        at_risk_df = pd.concat([overall_at_risk, group_at_risk], ignore_index=True)
    else:
        at_risk_df = overall_at_risk

    st.dataframe(pivot_at_risk_table(at_risk_df), hide_index=True, width="stretch")
    _render_dataframe_download(
        "Download number-at-risk table as CSV",
        at_risk_df,
        "number_at_risk.csv",
    )

    st.subheader("Survival probability at selected time points")
    probability_results = (
        [overall_result] + group_results
        if use_group and has_group and group_results
        else [overall_result]
    )
    probability_df = survival_probability_table_by_group(probability_results, timepoints)
    st.dataframe(pivot_survival_probability_table(probability_df), hide_index=True, width="stretch")
    _render_dataframe_download(
        "Download survival probabilities as CSV",
        probability_df,
        "survival_probabilities.csv",
    )


def _survival_dataframe_for_group(
    uploaded_df: pd.DataFrame | None,
    config: SurvivalConfig,
    stored_survival_df: pd.DataFrame,
    group_col: str | None,
) -> pd.DataFrame:
    if uploaded_df is None:
        if group_col is None:
            return stored_survival_df.drop(columns=["_group"], errors="ignore").copy(deep=True)
        return stored_survival_df.copy(deep=True)
    if group_col is not None and group_col not in uploaded_df.columns:
        return stored_survival_df.copy(deep=True)

    analysis_config = replace(config, group_col=group_col)
    return create_survival_ready_dataframe(uploaded_df, analysis_config)


def _render_survival_filters(
    uploaded_df: pd.DataFrame | None,
    profile_df: pd.DataFrame | None,
    annotations: Any,
) -> pd.DataFrame | None:
    if uploaded_df is None:
        st.session_state["active_survival_filters"] = {}
        return None

    filter_columns = get_columns_for_use(annotations, USE_FILTER, uploaded_df.columns)
    if not filter_columns:
        st.caption("No columns are annotated as filters. Update Column annotations in Setup.")
        st.session_state["active_survival_filters"] = {}
        return uploaded_df

    detected_types = (
        profile_df.set_index("column_name")["detected_type"].astype(str).to_dict()
        if profile_df is not None and not profile_df.empty
        else {}
    )
    _sanitize_multiselect_state("active_survival_filter_columns", filter_columns)
    selections: dict[str, tuple[str, Any]] = {}
    with st.expander("Cohort filters"):
        selected_columns = st.multiselect(
            "Add filter variables",
            filter_columns,
            default=[],
            key="active_survival_filter_columns",
            help="Only selected variables will show filter controls.",
        )
        if not selected_columns:
            st.caption("No filters selected; the full mapped cohort is included.")

        for column in selected_columns:
            series = uploaded_df[column]
            detected_type = detected_types.get(column, "")
            if detected_type in {"integer", "float"}:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if numeric.empty or numeric.min() == numeric.max():
                    st.caption(f"{column}: no variable numeric range")
                    continue
                lower, upper = float(numeric.min()), float(numeric.max())
                selected = st.slider(
                    f"Filter by {column}",
                    lower,
                    upper,
                    (lower, upper),
                    key=f"survival_filter_{column}",
                )
                if tuple(selected) != (lower, upper):
                    selections[column] = ("numeric", selected)
            elif detected_type == "date":
                dates = pd.to_datetime(series, errors="coerce").dropna()
                if dates.empty or dates.min().date() == dates.max().date():
                    st.caption(f"{column}: no variable date range")
                    continue
                bounds = (dates.min().date(), dates.max().date())
                selected = st.date_input(
                    f"Filter by {column}",
                    bounds,
                    min_value=bounds[0],
                    max_value=bounds[1],
                    key=f"survival_filter_{column}",
                )
                if len(selected) == 2 and tuple(selected) != bounds:
                    selections[column] = ("date", selected)
            else:
                options = sorted(series.dropna().unique().tolist(), key=str)
                selected = st.multiselect(
                    f"Filter by {column}",
                    options,
                    default=[],
                    help="Leave empty to include all values.",
                    key=f"survival_filter_{column}",
                )
                if selected:
                    selections[column] = ("categorical", selected)

    st.session_state["active_survival_filters"] = selections
    if selections:
        st.caption("Active filters: " + ", ".join(selections))
    else:
        st.caption("Current cohort: all mapped rows")
    return _apply_survival_filters(uploaded_df, selections)


def _apply_survival_filters(
    df: pd.DataFrame,
    selections: dict[str, tuple[str, Any]],
) -> pd.DataFrame:
    filtered = df.copy(deep=True)
    for column, (filter_type, value) in selections.items():
        if column not in filtered:
            continue
        if filter_type == "numeric":
            series = pd.to_numeric(filtered[column], errors="coerce")
            filtered = filtered[series.between(*value)]
        elif filter_type == "date":
            series = pd.to_datetime(filtered[column], errors="coerce").dt.date
            filtered = filtered[series.between(*value)]
        elif filter_type == "categorical":
            filtered = filtered[filtered[column].isin(value)]
    return filtered.copy(deep=True)


def _render_current_mapping(config: SurvivalConfig) -> None:
    st.markdown("**Current survival mapping**")
    if config.time_source == "dates":
        mapping_rows = [
            {"Field": "Time", "Value": "Derived from dates"},
            {"Field": "Start date column", "Value": config.start_date_col},
            {"Field": "Event date column", "Value": config.event_date_col},
            {
                "Field": "Last follow-up date column",
                "Value": config.last_followup_date_col,
            },
            {
                "Field": "Missing event dates",
                "Value": (
                    "Censor at last follow-up"
                    if config.missing_event_handling == "treat_as_censored"
                    else "Exclude"
                ),
            },
            {"Field": "Patient ID column", "Value": config.id_col or "Row number"},
            {"Field": "Group column", "Value": config.group_col or "None"},
            {"Field": "Time unit", "Value": config.time_unit},
        ]
    else:
        mapping_rows = [
            {"Field": "Time column", "Value": config.time_col},
            {"Field": "Event column", "Value": config.event_col},
            {
                "Field": "Event values",
                "Value": ", ".join(_format_value(value) for value in config.event_values),
            },
            {
                "Field": "Censor values",
                "Value": ", ".join(_format_value(value) for value in config.censor_values),
            },
            {
                "Field": "Unmapped values",
                "Value": _format_event_handling(
                    getattr(config, "unmapped_event_handling", "exclude")
                ),
            },
            {
                "Field": "Missing event values",
                "Value": _format_event_handling(config.missing_event_handling),
            },
            {"Field": "Patient ID column", "Value": config.id_col or "Row number"},
            {"Field": "Group column", "Value": config.group_col or "None"},
            {"Field": "Time unit", "Value": config.time_unit},
        ]
    mapping_df = pd.DataFrame(
        mapping_rows
    )
    st.dataframe(mapping_df, hide_index=True, width="stretch")


def _render_survival_summary_metrics(
    summary: dict[str, Any],
    median_survival: float,
    time_unit: str,
    yearly_probabilities: dict[int, float | None],
    logrank_result: dict[str, Any] | None,
) -> None:
    first_row = st.columns(4)
    first_row[0].metric("Current cohort", summary["n"])
    first_row[1].metric("Events", summary["events"])
    first_row[2].metric("Censored", summary["censored"])
    first_row[3].metric("Event rate", f"{summary['event_rate']}%")

    second_row = st.columns(4)
    second_row[0].metric("Median survival", format_survival_time(median_survival, time_unit))
    for column, year in zip(second_row[1:], (1, 3, 5)):
        probability = yearly_probabilities.get(year)
        column.metric(
            f"{year}-year survival",
            "Not available" if probability is None else f"{probability:.1%}",
        )

    third_row = st.columns(3 if logrank_result is not None else 2)
    third_row[0].metric("Median follow-up", _format_time_value(summary["median_followup"], time_unit))
    third_row[1].metric("Max follow-up", _format_time_value(summary["max_followup"], time_unit))
    if logrank_result is not None:
        third_row[2].metric(
            "Log-rank p-value",
            (
                format_p_value(logrank_result["p_value"])
                if logrank_result.get("available")
                else "Not available"
            ),
        )


def _render_survival_group_label_editor(survival_ready_df: pd.DataFrame, group_col: str) -> None:
    if "_group" not in survival_ready_df.columns:
        return

    raw_group_values = sorted(
        [value for value in survival_ready_df["_group"].dropna().unique()],
        key=lambda value: str(value),
    )
    if not raw_group_values:
        return

    all_labels = st.session_state.get("group_value_labels")
    if not isinstance(all_labels, dict):
        all_labels = {}

    existing_column_labels = all_labels.get(group_col, {})
    if not isinstance(existing_column_labels, dict):
        existing_column_labels = {}

    updated_column_labels = {}
    with st.expander("Group value labels"):
        for raw_value in raw_group_values:
            raw_key = str(raw_value)
            entered_label = st.text_input(
                f"Raw value: {raw_key}",
                value=str(existing_column_labels.get(raw_key, "")),
                key=f"survival_group_label_{group_col}_{raw_key}",
            )
            if entered_label.strip():
                updated_column_labels[raw_key] = entered_label.strip()

    if updated_column_labels:
        all_labels[group_col] = updated_column_labels
    else:
        all_labels.pop(group_col, None)
    st.session_state["group_value_labels"] = all_labels


def _render_survival_warning_section(warnings: list[str]) -> None:
    st.subheader("Survival analysis warnings")
    unique_warnings = _unique_messages(warnings)
    if not unique_warnings:
        st.success("No major survival-analysis warnings detected.")
        return

    for warning in unique_warnings:
        st.warning(warning)


def _render_logrank_section(logrank_result: dict[str, Any]) -> None:
    st.subheader("Log-rank test")
    st.caption("The log-rank test compares survival curves between groups. It does not adjust for other variables.")

    if not logrank_result.get("available"):
        st.info(str(logrank_result.get("reason", "Log-rank test is unavailable.")))
        return

    columns = st.columns(4)
    columns[0].metric("Test statistic", f"{float(logrank_result['test_statistic']):.2f}")
    columns[1].metric("p-value", format_p_value(logrank_result["p_value"]))
    columns[2].metric("Degrees of freedom", logrank_result["degrees_of_freedom"])
    columns[3].metric("Groups", logrank_result["n_groups"])


def _format_survival_summary_display_table(summary_df: pd.DataFrame, time_unit: str) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    display_df = summary_df.copy(deep=True)
    if "event_rate" in display_df.columns:
        display_df["event_rate"] = display_df["event_rate"].apply(lambda value: f"{float(value):.2f}%")

    for column in ["median_followup", "max_followup"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(lambda value: _format_time_value(value, time_unit))

    if "median_survival" in display_df.columns:
        display_df["median_survival"] = display_df["median_survival"].apply(
            lambda value: format_survival_time(value, time_unit)
        )

    return display_df


def _render_dataframe_download(label: str, df: pd.DataFrame, file_name: str) -> None:
    if df.empty:
        return

    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


def _unique_messages(messages: list[str]) -> list[str]:
    seen = set()
    unique = []
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        unique.append(message)
    return unique


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
    pending_config = st.session_state.pop("pending_survival_setup_config", None)
    if isinstance(pending_config, SurvivalConfig):
        _seed_survival_setup_widget_state(pending_config, role_suggestions)

    _render_control_anchor("survival-time-setup")
    time_source_label = st.radio(
        "Survival time setup",
        ["Use a follow-up duration column", "Derive from date columns"],
        horizontal=True,
        key="survival_time_source",
    )
    time_source = "dates" if time_source_label == "Derive from date columns" else "duration"
    time_col: str | None = None
    event_col: str | None = None
    start_date_col: str | None = None
    event_date_col: str | None = None
    last_followup_date_col: str | None = None
    event_values: list[Any] = []
    censor_values: list[Any] = []
    missing_event_handling = "exclude"
    unmapped_event_handling = "exclude"

    if time_source == "duration":
        _render_control_anchor("follow-up-time")
        time_col = _select_required_role_column(
            "Follow-up / survival time column",
            role_suggestions["time_candidates"],
            all_columns,
            "time_col",
        )
        _render_control_anchor("event-status")
        event_col = _select_required_role_column(
            "Event/status column",
            role_suggestions["event_candidates"],
            all_columns,
            "event_col",
        )
        if event_col:
            _render_control_anchor("event-values")
            (
                event_values,
                censor_values,
                missing_event_handling,
                unmapped_event_handling,
            ) = _render_event_value_mapping(df, event_col)
        time_unit = _render_time_unit_selector(time_col or "")
    else:
        st.caption(
            "Duration is calculated in days: event date minus start date, or last follow-up "
            "minus start date when the event date is missing."
        )
        date_columns = profile.loc[
            profile["detected_type"] == "date",
            "column_name",
        ].astype(str).tolist()
        date_options = date_columns + [
            column for column in all_columns if column not in date_columns
        ]
        _render_control_anchor("start-date")
        start_date_col = st.selectbox(
            "Start date column",
            date_options,
            index=_suggest_date_column_index(date_options, ("start", "diagnosis", "enroll", "index")),
            key="start_date_col_dates",
        )
        _render_control_anchor("event-date")
        event_date_col = st.selectbox(
            "Event date column",
            date_options,
            index=_suggest_date_column_index(date_options, ("event", "death", "relapse", "progress")),
            key="event_date_col_dates",
        )
        _render_control_anchor("last-follow-up-date")
        last_followup_date_col = st.selectbox(
            "Last follow-up date column",
            date_options,
            index=_suggest_date_column_index(
                date_options,
                ("last_follow", "last follow", "last_contact", "last contact", "last_seen"),
            ),
            key="last_followup_date_col_dates",
        )
        missing_date_choice = st.radio(
            "How should missing event dates be interpreted?",
            [
                "Unknown / exclude from survival analysis",
                "Censored / event did not occur before last follow-up",
            ],
            index=0,
            key=f"missing_event_handling_{event_date_col}",
            help="Exclusion is the conservative default.",
        )
        missing_event_handling = (
            "treat_as_censored"
            if missing_date_choice.startswith("Censored")
            else "exclude"
        )
        time_unit = "days"
        st.caption("Time unit: days")

    _render_control_anchor("patient-id")
    id_col = _select_optional_role_column(
        "Patient ID column",
        role_suggestions["id_candidates"],
        all_columns,
        "No patient ID / use row number",
        "id_col",
    )

    _render_control_anchor("group-column")
    group_col = _select_optional_role_column(
        "Optional grouping column",
        role_suggestions["group_candidates"],
        all_columns,
        "No grouping",
        "group_col",
    )

    config = SurvivalConfig(
        time_col=time_col,
        event_col=event_col,
        event_values=event_values,
        censor_values=censor_values,
        id_col=id_col,
        group_col=group_col,
        time_unit=time_unit,
        missing_event_handling=missing_event_handling,
        unmapped_event_handling=unmapped_event_handling,
        time_source=time_source,
        start_date_col=start_date_col,
        event_date_col=event_date_col,
        last_followup_date_col=last_followup_date_col,
    )
    errors, warnings = validate_survival_config(df, config)
    mapping_is_current = st.session_state.get("survival_config") == config

    if st.button(
        "Confirm survival mapping",
        type="primary",
        disabled=bool(errors) or mapping_is_current,
    ):
        survival_ready_df = create_survival_ready_dataframe(df, config)
        st.session_state["survival_config"] = config
        st.session_state["survival_ready_df"] = survival_ready_df
        st.session_state.pop("combined_report_html", None)
        st.session_state.pop("combined_report_pdf", None)
        st.session_state["column_annotations"] = apply_survival_roles(
            sync_annotations(
                st.session_state.get("column_annotations"),
                df,
                profile,
                config,
            ),
            config,
            seed_analysis_uses=True,
        )
        st.session_state["annotation_editor_version"] = (
            int(st.session_state.get("annotation_editor_version", 0)) + 1
        )

        event_count = int((survival_ready_df["_event"] == 1).sum())
        censored_count = int((survival_ready_df["_event"] == 0).sum())
        st.success(
            "Survival mapping confirmed.\n\n"
            f"Usable rows: {len(survival_ready_df)}\n\n"
            f"Events: {event_count}\n\n"
            f"Censored: {censored_count}"
        )

    mapping_is_current = st.session_state.get("survival_config") == config
    _render_mapping_validation_summary(
        df,
        config,
        errors,
        warnings,
        mapping_is_current=mapping_is_current,
    )
    if not errors and mapping_is_current:
        st.button(
            "Continue to Data Quality",
            type="primary",
            on_click=_navigate_to_main_tab,
            args=("Data Quality",),
        )


def _navigate_to_main_tab(tab_label: str) -> None:
    st.session_state["main_tab"] = tab_label


def _render_control_anchor(anchor: str) -> None:
    st.markdown(f"<span id='{anchor}'></span>", unsafe_allow_html=True)


def _render_mapping_validation_summary(
    df: pd.DataFrame,
    config: SurvivalConfig,
    errors: list[str],
    warnings: list[str],
    *,
    mapping_is_current: bool,
) -> None:
    _render_control_anchor("validation-summary")
    st.markdown("#### Validation summary")
    if errors:
        st.error(
            f"{len(errors)} blocking issue{'s' if len(errors) != 1 else ''}. "
            "Fix these before confirming the mapping."
        )
        st.markdown("**Errors**")
        for message in errors:
            _render_validation_issue(message)
    else:
        st.success(
            "Mapping is valid and confirmed."
            if mapping_is_current
            else "Mapping is valid and ready to confirm."
        )

    if warnings:
        st.markdown("**Warnings**")
        for message in warnings:
            _render_validation_issue(message)

    status, patient_message = _assess_patient_level_structure(df, config.id_col)
    st.markdown("**Dataset suitability**")
    getattr(st, status)(patient_message)
    if errors:
        st.warning(
            "This dataset is not currently suitable for survival analysis. "
            "Cohort and chart exploration remain available while the blocking issues are fixed."
        )
    st.caption(
        "Best fit: one row per patient, subject, admission, or case with individual follow-up "
        "and event status. Aggregate trial summaries and repeated-visit tables need reshaping "
        "before basic survival analysis."
    )


def _render_validation_issue(message: str) -> None:
    anchor = _validation_fix_anchor(message)
    st.markdown(
        f"- {escape(message)} <a href='#{anchor}'>Fix</a>",
        unsafe_allow_html=True,
    )


def _validation_fix_anchor(message: str) -> str:
    normalized = message.lower()
    if "patient id" in normalized:
        return "patient-id"
    if "group column" in normalized:
        return "group-column"
    if "start date" in normalized:
        return "start-date"
    if "last follow-up date" in normalized:
        return "last-follow-up-date"
    if "event date" in normalized:
        return "event-date"
    if (
        "event value" in normalized
        or "unmapped values" in normalized
        or "event count" in normalized
        or "censor" in normalized
        or "usable rows" in normalized
    ):
        return "event-values"
    if "event column" in normalized:
        return "event-status"
    if "time source" in normalized:
        return "survival-time-setup"
    return "follow-up-time"


def _assess_patient_level_structure(
    df: pd.DataFrame,
    id_col: str | None,
) -> tuple[str, str]:
    if id_col is None or id_col not in df.columns:
        return (
            "info",
            "Patient-level structure is unverified. Select a patient ID to check the "
            "one-row-per-patient assumption; until then, counts treat each row as a case.",
        )

    ids = normalize_missing_values(df[[id_col]])[id_col]
    non_missing_ids = ids.dropna()
    missing_count = int(ids.isna().sum())
    repeated_ids = non_missing_ids[non_missing_ids.duplicated(keep=False)]
    if not repeated_ids.empty:
        repeated_count = int(repeated_ids.nunique())
        return (
            "warning",
            "Poor fit for basic patient-level analysis: "
            f"{repeated_count} {id_col} value{'s' if repeated_count != 1 else ''} "
            "repeat across "
            f"{len(repeated_ids)} rows. This looks like repeated visits or records; "
            "reduce to one analysis row per patient or use a repeated-measures method.",
        )
    if missing_count:
        return (
            "warning",
            f"Patient-level structure is uncertain: {id_col} is missing in "
            f"{missing_count} row{'s' if missing_count != 1 else ''}, which cannot "
            "be checked for uniqueness.",
        )
    return (
        "success",
        f"Patient-level structure check passed: {id_col} is unique across "
        f"{len(non_missing_ids)} non-missing rows.",
    )


def _seed_survival_setup_widget_state(
    config: SurvivalConfig,
    role_suggestions: dict[str, list[dict[str, Any]]],
) -> None:
    st.session_state["survival_time_source"] = (
        "Derive from date columns"
        if config.time_source == "dates"
        else "Use a follow-up duration column"
    )

    def seed_role(prefix: str, value: str | None, candidates: list[dict[str, Any]]) -> None:
        candidate_columns = [candidate["column_name"] for candidate in candidates]
        if value is None:
            st.session_state[f"{prefix}_recommended"] = "__none__"
        elif value in candidate_columns:
            st.session_state[f"{prefix}_recommended"] = value
        else:
            st.session_state[f"{prefix}_recommended"] = "__search_all__"
            st.session_state[f"{prefix}_all_columns"] = value

    if config.time_source == "dates":
        st.session_state["start_date_col_dates"] = config.start_date_col
        st.session_state["event_date_col_dates"] = config.event_date_col
        st.session_state["last_followup_date_col_dates"] = config.last_followup_date_col
        if config.event_date_col:
            st.session_state[f"missing_event_handling_{config.event_date_col}"] = (
                "Censored / event did not occur before last follow-up"
                if config.missing_event_handling == "treat_as_censored"
                else "Unknown / exclude from survival analysis"
            )
    else:
        seed_role("time_col", config.time_col, role_suggestions["time_candidates"])
        seed_role("event_col", config.event_col, role_suggestions["event_candidates"])
        if config.event_col:
            st.session_state[f"event_values_{config.event_col}"] = config.event_values
            st.session_state[f"censor_values_{config.event_col}"] = config.censor_values
            st.session_state[f"unmapped_event_handling_{config.event_col}"] = {
                "treat_as_censored": "Treat as censored",
                "treat_as_event": "Treat as events",
            }.get(config.unmapped_event_handling, "Exclude from survival analysis")
            st.session_state[f"missing_event_handling_{config.event_col}"] = (
                "Treat as censored"
                if config.missing_event_handling == "treat_as_censored"
                else "Exclude from survival analysis"
            )
        if config.time_col:
            st.session_state[f"time_unit_{config.time_col}"] = config.time_unit

    seed_role("id_col", config.id_col, role_suggestions["id_candidates"])
    seed_role("group_col", config.group_col, role_suggestions["group_candidates"])


def _render_column_annotations(df: pd.DataFrame, profile: pd.DataFrame) -> None:
    st.subheader("Column annotations")
    config = st.session_state.get("survival_config")

    annotations = sync_annotations(
        st.session_state.get("column_annotations"),
        df,
        profile,
        config,
    )
    st.session_state["column_annotations"] = annotations

    st.caption(
        "Meaning describes what a column represents. Analysis uses are independent, so a column "
        "can be available for filters, grouping, the baseline table, and charts."
    )
    editor_df = annotations_to_dataframe(annotations, profile)
    editor_version = int(st.session_state.get("annotation_editor_version", 0))

    with st.form(f"column_annotation_form_{editor_version}"):
        edited_df = st.data_editor(
            editor_df,
            hide_index=True,
            width="stretch",
            height=min(800, 38 + 35 * max(len(editor_df), 1)),
            disabled=["Column", "Type", "Missing %", "Example values"],
            column_config={
                "Column": st.column_config.TextColumn("Column"),
                "Type": st.column_config.TextColumn("Detected type"),
                "Missing %": st.column_config.NumberColumn("Missing %", format="%.2f"),
                "Example values": st.column_config.TextColumn("Example values"),
                "Meaning": st.column_config.SelectboxColumn(
                    "Meaning",
                    options=MEANING_OPTIONS,
                    required=True,
                ),
                "Custom meaning": st.column_config.TextColumn(
                    "Custom meaning",
                    help="Required only when Meaning is Custom...",
                ),
                **{
                    label: (
                        None
                        if use == USE_COX
                        else st.column_config.CheckboxColumn(label)
                    )
                    for use, label in USE_COLUMN_LABELS.items()
                },
            },
            key=f"column_annotation_editor_{editor_version}",
        )
        save_annotations = st.form_submit_button(
            "Save annotations",
            type="primary",
        )
        reset_annotations = st.form_submit_button("Reset to suggested annotations")

    if save_annotations:
        try:
            parsed_annotations = annotations_from_dataframe(edited_df, df.columns)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["column_annotations"] = sync_annotations(
                parsed_annotations,
                df,
                profile,
                config,
            )
            _invalidate_annotation_consumer_state()
            st.session_state["annotation_editor_version"] = editor_version + 1
            st.session_state["annotation_status_message"] = "Column annotations saved."
            st.rerun()

    if reset_annotations:
        st.session_state["column_annotations"] = build_default_annotations(df, profile, config)
        _invalidate_annotation_consumer_state()
        st.session_state["annotation_editor_version"] = editor_version + 1
        st.session_state["annotation_status_message"] = "Suggested annotations restored."
        st.rerun()

    status_message = st.session_state.pop("annotation_status_message", None)
    if status_message:
        st.success(status_message)

    summary = get_annotation_summary(st.session_state["column_annotations"])
    summary_columns = st.columns(5)
    summary_columns[0].metric("Filters", summary[USE_FILTER])
    summary_columns[1].metric("Groups", summary[USE_GROUP])
    summary_columns[2].metric("Baseline", summary[USE_BASELINE])
    summary_columns[3].metric("Charts", summary[USE_CHARTS])
    summary_columns[4].metric("Ignored", summary[USE_IGNORE])


def _render_export_section(df: pd.DataFrame, profile: pd.DataFrame) -> None:
    st.download_button(
        "Download current dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="current_dataset.csv",
        mime="text/csv",
    )

    with st.expander("Load a saved mapping and annotation configuration"):
        uploaded_config = st.file_uploader(
            "Configuration JSON",
            type=["json"],
            key="analysis_configuration_upload",
            help="Configuration is validated against the currently uploaded dataset before use.",
        )
        if st.button(
            "Apply uploaded configuration",
            disabled=uploaded_config is None,
        ):
            try:
                config, loaded_annotations = deserialize_analysis_configuration(
                    uploaded_config.getvalue()
                )
                errors, warnings = validate_survival_config(df, config)
                if errors:
                    raise ValueError(" ".join(errors))
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["survival_config"] = config
                st.session_state["survival_ready_df"] = create_survival_ready_dataframe(
                    df,
                    config,
                )
                st.session_state["column_annotations"] = sync_annotations(
                    loaded_annotations,
                    df,
                    profile,
                    config,
                )
                st.session_state["pending_survival_setup_config"] = config
                st.session_state["analysis_goal"] = ANALYSIS_GOAL_SURVIVAL
                st.session_state["annotation_editor_version"] = (
                    int(st.session_state.get("annotation_editor_version", 0)) + 1
                )
                _invalidate_annotation_consumer_state()
                st.session_state["configuration_status_message"] = (
                    "Mapping and annotations loaded."
                    + (f" Review: {' '.join(warnings)}" if warnings else "")
                )
                st.rerun()

    status_message = st.session_state.pop("configuration_status_message", None)
    if status_message:
        st.success(status_message)

    config = st.session_state.get("survival_config")
    annotations = st.session_state.get("column_annotations")
    if config is None or not isinstance(annotations, dict):
        st.info(
            "The current dataset can be downloaded now. Confirm or load a survival "
            "mapping to enable cleaned data, reusable configuration, and combined reports."
        )
        return

    ready_df = create_survival_ready_dataframe(df, config)
    cleaned_df = create_cleaned_mapped_dataframe(df, config)
    config_bytes = serialize_analysis_configuration(config, annotations)

    data_column, config_column = st.columns(2)
    data_column.download_button(
        "Download cleaned mapped data as CSV",
        data=cleaned_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_mapped_data.csv",
        mime="text/csv",
    )
    config_column.download_button(
        "Save mapping and annotations as JSON",
        data=config_bytes,
        file_name="analysis_configuration.json",
        mime="application/json",
    )

    st.markdown("**Combined report**")
    if st.button("Prepare combined HTML and PDF reports"):
        quality_report = build_data_quality_report(
            df,
            profile,
            config,
            ready_df,
            annotations,
        )
        st.session_state["combined_report_html"] = build_html_report(
            df,
            config,
            annotations,
            quality_report,
            ready_df,
        )
        st.session_state["combined_report_pdf"] = build_pdf_report(
            df,
            config,
            annotations,
            quality_report,
            ready_df,
        )

    html_report = st.session_state.get("combined_report_html")
    pdf_report = st.session_state.get("combined_report_pdf")
    if html_report and pdf_report:
        report_columns = st.columns(2)
        report_columns[0].download_button(
            "Download combined report as HTML",
            data=html_report,
            file_name="medical_dataset_report.html",
            mime="text/html",
        )
        report_columns[1].download_button(
            "Download combined report as PDF",
            data=pdf_report,
            file_name="medical_dataset_report.pdf",
            mime="application/pdf",
        )
    st.caption("Exports may contain sensitive source data. Store and share them appropriately.")


def _invalidate_annotation_consumer_state() -> None:
    for key in [
        "cohort_group_col",
        "cohort_continuous_vars",
        "cohort_categorical_vars",
        "chart_x_col",
        "chart_y_col",
        "chart_color_col",
        "survival_analysis_group_col",
        "active_survival_filter_columns",
        "active_survival_filters",
        "combined_report_html",
        "combined_report_pdf",
    ]:
        st.session_state.pop(key, None)


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

    with st.expander("Why these columns were recommended"):
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
) -> tuple[list[Any], list[Any], str, str]:
    unique_values = _unique_non_missing_values(df, event_col)
    st.markdown(f"Values found in `{event_col}`:")

    if not unique_values:
        st.caption("No non-missing values found.")
        return [], [], "exclude", "exclude"

    st.caption(", ".join(_format_value(value) for value in unique_values))

    default_event_values = _default_event_values(unique_values)
    event_values = st.multiselect(
        "Which value(s) mean the event occurred?",
        unique_values,
        default=default_event_values,
        format_func=_format_value,
        key=f"event_values_{event_col}",
    )
    default_censor_values = _default_censor_values(
        unique_values,
        event_values,
    )
    censor_values = st.multiselect(
        "Which value(s) explicitly mean censored?",
        unique_values,
        default=default_censor_values,
        format_func=_format_value,
        key=f"censor_values_{event_col}",
    )

    mapped_keys = {
        _canonical_value(value)
        for value in list(event_values) + list(censor_values)
    }
    unmapped_values = [
        value
        for value in unique_values
        if _canonical_value(value) not in mapped_keys
    ]
    if unmapped_values:
        st.warning(
            "Currently unmapped: "
            + ", ".join(_format_value(value) for value in unmapped_values)
        )
    else:
        st.caption("All non-missing values are explicitly mapped.")

    with st.expander("Advanced event handling"):
        unmapped_choice = st.radio(
            "Unmapped non-missing values",
            [
                "Exclude from survival analysis",
                "Treat as censored",
                "Treat as events",
            ],
            index=0,
            key=f"unmapped_event_handling_{event_col}",
            help=(
                "Exclusion is the conservative default. Treating unknown values as censored "
                "or as events can materially change survival estimates."
            ),
        )
        unmapped_event_handling = {
            "Treat as censored": "treat_as_censored",
            "Treat as events": "treat_as_event",
        }.get(unmapped_choice, "exclude")

        missing_choice = st.radio(
            "Missing event values",
            ["Exclude from survival analysis", "Treat as censored"],
            index=0,
            key=f"missing_event_handling_{event_col}",
        )
        missing_event_handling = "treat_as_censored" if missing_choice == "Treat as censored" else "exclude"

    return (
        event_values,
        censor_values,
        missing_event_handling,
        unmapped_event_handling,
    )


def _render_time_unit_selector(time_col: str) -> str:
    options = ["days", "months", "years", "unknown"]
    suggested_unit = (
        "days"
        if st.session_state.get("loaded_example_dataset") == "lung.csv"
        and time_col == "time"
        else _suggest_time_unit(time_col)
    )
    return str(
        st.selectbox(
            "Time unit",
            options,
            index=options.index(suggested_unit),
            key=f"time_unit_{time_col}",
        )
    )


def _suggest_date_column_index(columns: list[str], markers: tuple[str, ...]) -> int:
    for index, column in enumerate(columns):
        normalized = column.lower().replace("-", "_")
        if any(marker in normalized for marker in markers):
            return index
    return 0


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


def _default_censor_values(
    unique_values: list[Any],
    event_values: list[Any],
) -> list[Any]:
    event_keys = {_canonical_value(value) for value in event_values}
    numeric_values = pd.to_numeric(pd.Series(unique_values), errors="coerce")

    if (
        len(unique_values) == 2
        and numeric_values.notna().all()
        and len(event_keys) == 1
    ):
        return [
            value
            for value in unique_values
            if _canonical_value(value) not in event_keys
        ]

    censor_markers = {
        "0",
        "false",
        "no",
        "n",
        "alive",
        "living",
        "censored",
        "censor",
        "no event",
        "event free",
        "event-free",
        "disease free",
        "disease-free",
        "no death",
    }
    return [
        value
        for value in unique_values
        if (
            _canonical_value(value) in censor_markers
            and _canonical_value(value) not in event_keys
        )
    ]


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


def _format_event_handling(value: str) -> str:
    return {
        "exclude": "Exclude",
        "treat_as_censored": "Treat as censored",
        "treat_as_event": "Treat as events",
    }.get(value, str(value))


if __name__ == "__main__":
    main()
