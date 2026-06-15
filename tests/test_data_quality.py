import pandas as pd

from src.data_quality import (
    QualityIssue,
    build_data_quality_report,
    check_duplicate_patient_ids,
    check_duplicate_rows,
    compute_dataset_overview,
    compute_group_quality,
    compute_missingness_by_column,
    compute_survival_quality,
    detect_sensitive_column_candidates,
    determine_quality_status,
)
from src.survival_mapping import SurvivalConfig, create_survival_ready_dataframe


def test_compute_dataset_overview():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", None, "z"]})

    overview = compute_dataset_overview(df)

    assert overview["n_rows"] == 3
    assert overview["n_columns"] == 2
    assert overview["total_cells"] == 6
    assert overview["missing_cells"] == 2
    assert overview["complete_rows"] == 1


def test_compute_missingness_by_column_sorts_descending():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", None, None]})

    missingness = compute_missingness_by_column(df)

    assert missingness.iloc[0]["column_name"] == "b"
    assert missingness.loc[missingness["column_name"] == "a", "missing_count"].iloc[0] == 1
    assert missingness.loc[missingness["column_name"] == "b", "missing_count"].iloc[0] == 2


def test_check_duplicate_rows_counts_all_rows_in_duplicate_groups():
    df = pd.DataFrame({"id": ["A", "B", "B"], "time": [1, 2, 2], "status": [0, 1, 1]})

    duplicate_rows = check_duplicate_rows(df)

    assert duplicate_rows["duplicate_row_count"] == 2
    assert duplicate_rows["duplicate_group_count"] == 1


def test_check_duplicate_patient_ids():
    df = pd.DataFrame({"patient_id": ["P1", "P2", "P2", "P3"], "time": [10, 20, 30, 40]})

    duplicate_ids = check_duplicate_patient_ids(df, "patient_id")

    assert duplicate_ids["duplicate_id_row_count"] == 2
    assert duplicate_ids["duplicate_id_value_count"] == 1
    assert "P2" in duplicate_ids["duplicate_id_examples"]


def test_check_duplicate_patient_ids_when_no_id_selected():
    df = pd.DataFrame({"patient_id": ["P1", "P2"], "time": [10, 20]})

    duplicate_ids = check_duplicate_patient_ids(df, None)

    assert duplicate_ids["checked"] is False
    assert duplicate_ids["duplicate_id_row_count"] is None


def test_compute_survival_quality_good_mapping():
    df = pd.DataFrame({"time": [10, 20, 30, 40], "status": [1, 0, 1, 0]})
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
    )

    survival_quality = compute_survival_quality(df, config)

    assert survival_quality["raw_rows"] == 4
    assert survival_quality["usable_survival_rows"] == 4
    assert survival_quality["excluded_rows"] == 0
    assert survival_quality["events"] == 2
    assert survival_quality["censored"] == 2
    assert survival_quality["negative_time_count"] == 0
    assert survival_quality["unmapped_event_value_count"] == 0


def test_compute_survival_quality_bad_values():
    df = pd.DataFrame({"time": [10, -5, None, 0], "status": [1, 0, "Unknown", 1]})
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
    )

    survival_quality = compute_survival_quality(df, config)

    assert survival_quality["negative_time_count"] == 1
    assert survival_quality["time_missing_count"] == 1
    assert survival_quality["zero_time_count"] == 1
    assert "Unknown" in survival_quality["unmapped_event_values"]
    assert any(issue.severity == "error" for issue in survival_quality["issues"])


def test_compute_group_quality():
    survival_ready_df = pd.DataFrame(
        {
            "_time": [10, 20, 30, 40, 50, 60],
            "_event": [1, 0, 1, 0, 1, 0],
            "_group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    group_quality, issues = compute_group_quality(survival_ready_df)

    assert len(group_quality) == 2
    assert set(group_quality["group"]) == {"A", "B"}
    assert group_quality.loc[group_quality["group"] == "A", "events"].iloc[0] == 2
    assert group_quality.loc[group_quality["group"] == "B", "censored"].iloc[0] == 2
    assert any("fewer than 5 rows" in issue.message for issue in issues)


def test_compute_group_quality_too_many_groups_warning():
    survival_ready_df = pd.DataFrame(
        {
            "_time": list(range(1, 12)),
            "_event": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "_group": [f"G{index}" for index in range(11)],
        }
    )

    _, issues = compute_group_quality(survival_ready_df)

    assert any("too many groups" in issue.message for issue in issues)


def test_detect_sensitive_column_candidates():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "email": ["a@example.com", "b@example.com"],
            "notes": ["Patient reported severe pain after treatment" * 3, "No issue" * 20],
        }
    )

    candidates = detect_sensitive_column_candidates(df)
    risk_by_column = dict(zip(candidates["column_name"], candidates["risk_level"]))

    assert risk_by_column["email"] == "high"
    assert risk_by_column["notes"] == "medium"
    assert risk_by_column["patient_id"] in {"low", "medium"}


def test_determine_quality_status():
    warning_issue = QualityIssue(
        severity="warning",
        category="missingness",
        message="Column X has missing values.",
    )
    error_issue = QualityIssue(
        severity="error",
        category="survival_time",
        message="Negative survival time.",
    )

    assert determine_quality_status([warning_issue]) == "warning"
    assert determine_quality_status([warning_issue, error_issue]) == "error"
    assert determine_quality_status([]) == "success"


def test_build_data_quality_report_combines_checks():
    df = pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003", "P003", "P004", "P005"],
            "time": [300, 500, 200, 200, -10, 0],
            "status": ["Alive", "Dead", "Alive", "Alive", "Dead", "Unknown"],
            "age": [55, 63, 72, 72, None, 70],
            "treatment": ["A", "B", "A", "A", "B", "A"],
            "email": ["a@example.com", None, "c@example.com", "c@example.com", "d@example.com", "e@example.com"],
        }
    )
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=["Dead"],
        censor_values=["Alive"],
        id_col="patient_id",
        group_col="treatment",
        time_unit="days",
    )
    survival_ready_df = create_survival_ready_dataframe(df, config)

    report = build_data_quality_report(df, survival_config=config, survival_ready_df=survival_ready_df)

    assert report["duplicate_rows"]["duplicate_row_count"] == 2
    assert report["duplicate_ids"]["duplicate_id_row_count"] == 2
    assert report["survival_quality"]["negative_time_count"] == 1
    assert "Unknown" in report["survival_quality"]["unmapped_event_values"]
    assert report["survival_quality"]["excluded_rows"] == 2
    assert not report["sensitive_column_candidates"].empty
