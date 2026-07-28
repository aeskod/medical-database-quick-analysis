import pandas as pd

from src.cohort_overview import (
    build_baseline_table,
    classify_summary_variable,
    compute_cohort_overview_metrics,
    get_cohort_role_columns,
    get_default_baseline_variables,
    summarize_categorical_by_group,
    summarize_categorical_variable,
    summarize_continuous_by_group,
    summarize_continuous_variable,
)
from src.profiling import profile_dataframe
from src.survival_mapping import SurvivalConfig, create_survival_ready_dataframe


def test_compute_cohort_overview_metrics_with_survival_ready_data_does_not_mutate_inputs():
    df = pd.DataFrame(
        {
            "time": [10, 20, 30, None],
            "status": [1, 0, 1, 0],
            "age": [55, 60, None, 70],
        }
    )
    survival_ready_df = pd.DataFrame({"_time": [10, 20, 30], "_event": [1, 0, 1]})
    original_df = df.copy(deep=True)
    original_survival_df = survival_ready_df.copy(deep=True)

    metrics = compute_cohort_overview_metrics(df, survival_ready_df, time_unit="days")

    assert metrics["n_rows"] == 4
    assert metrics["n_columns"] == 3
    assert metrics["complete_rows"] == 2
    assert metrics["complete_rows_percent"] == 50.0
    assert metrics["missing_cells_percent"] == 16.67
    assert metrics["usable_survival_rows"] == 3
    assert metrics["events"] == 2
    assert metrics["censored"] == 1
    assert metrics["event_rate"] == 66.67
    assert metrics["median_followup"] == 20.0
    assert metrics["max_followup"] == 30.0
    assert metrics["time_unit"] == "days"
    pd.testing.assert_frame_equal(df, original_df)
    pd.testing.assert_frame_equal(survival_ready_df, original_survival_df)


def test_compute_cohort_overview_metrics_without_survival_mapping_sets_survival_fields_to_none():
    df = pd.DataFrame({"age": [50, 60], "sex": ["F", "M"]})

    metrics = compute_cohort_overview_metrics(df)

    assert metrics["n_rows"] == 2
    assert metrics["usable_survival_rows"] is None
    assert metrics["events"] is None
    assert metrics["censored"] is None
    assert metrics["event_rate"] is None
    assert metrics["median_followup"] is None


def test_compute_cohort_overview_metrics_counts_distinct_patient_ids_and_median_age():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", None],
            "age": [40, 40, 60, 80],
        }
    )

    metrics = compute_cohort_overview_metrics(
        df,
        id_col="patient_id",
        age_col="age",
    )

    assert metrics["n_patients"] == 2
    assert metrics["patient_count_basis"] == "Distinct patient_id"
    assert metrics["missing_patient_ids"] == 1
    assert metrics["median_age"] == 50.0


def test_get_cohort_role_columns_uses_saved_semantic_annotations():
    from src.column_annotations import ColumnAnnotation

    df = pd.DataFrame(
        {
            "record": ["P1"],
            "years": [55],
            "gender_code": ["F"],
            "condition": ["NSCLC"],
            "arm_code": ["A"],
            "response_code": ["CR"],
        }
    )
    annotations = {
        "record": ColumnAnnotation("record", "Patient ID"),
        "years": ColumnAnnotation("years", "Age"),
        "gender_code": ColumnAnnotation("gender_code", "Sex / gender"),
        "condition": ColumnAnnotation("condition", "Diagnosis"),
        "arm_code": ColumnAnnotation("arm_code", "Treatment / exposure group"),
        "response_code": ColumnAnnotation("response_code", "Outcome other than survival"),
    }

    assert get_cohort_role_columns(df, annotations) == {
        "patient_id": ["record"],
        "age": ["years"],
        "sex": ["gender_code"],
        "diagnosis": ["condition"],
        "treatment": ["arm_code"],
        "outcome": ["response_code"],
    }


def test_classify_summary_variable_honors_survival_mapping_and_clinical_ordinal_hints():
    df = pd.DataFrame(
        {
            "inst": list(range(100, 112)),
            "patient_id": [f"P{index:03d}" for index in range(12)],
            "time": list(range(10, 22)),
            "status": [1, 0] * 6,
            "age": list(range(50, 62)),
            "sex": [1, 2] * 6,
            "ph.karno": [60, 70, 80, 90] * 3,
        }
    )
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        id_col="patient_id",
        group_col="sex",
    )
    profile = profile_dataframe(df).set_index("column_name").to_dict("index")

    assert classify_summary_variable("time", df["time"], profile["time"], config) == "excluded"
    assert classify_summary_variable("status", df["status"], profile["status"], config) == "excluded"
    assert classify_summary_variable("inst", df["inst"], profile["inst"], config) == "excluded"
    assert classify_summary_variable("patient_id", df["patient_id"], profile["patient_id"], config) == "id"
    assert classify_summary_variable("age", df["age"], profile["age"], config) == "continuous"
    assert classify_summary_variable("sex", df["sex"], profile["sex"], config) == "categorical"
    assert classify_summary_variable("ph.karno", df["ph.karno"], profile["ph.karno"], config) == "categorical"


def test_get_default_baseline_variables_excludes_ids_text_dates_and_survival_columns():
    df = pd.DataFrame(
        {
            "inst": list(range(100, 112)),
            "patient_id": [f"P{index:03d}" for index in range(12)],
            "time": list(range(10, 22)),
            "status": [1, 0] * 6,
            "age": list(range(50, 62)),
            "meal.cal": list(range(400, 412)),
            "wt.loss": list(range(0, 12)),
            "sex": [1, 2] * 6,
            "ph.ecog": [0, 1, 2, 3] * 3,
            "visit_date": ["2024-01-01"] * 12,
            "notes": [f"Long free text note for patient {index} with enough detail to be narrative." for index in range(12)],
        }
    )
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        id_col="patient_id",
        group_col="sex",
    )

    defaults = get_default_baseline_variables(df, profile_dataframe(df), config)

    assert defaults["continuous"] == ["age", "meal.cal", "wt.loss"]
    assert set(defaults["categorical"]) == {"sex", "ph.ecog"}
    assert {"inst", "patient_id", "time", "status", "visit_date", "notes"}.issubset(defaults["excluded"])


def test_summarize_continuous_variable_returns_descriptive_statistics():
    df = pd.DataFrame({"age": [10, 20, 30, None, "bad"]})

    summary = summarize_continuous_variable(df, "age")

    assert summary["n"] == 3
    assert summary["missing"] == 2
    assert summary["mean"] == 20.0
    assert summary["sd"] == 10.0
    assert summary["median"] == 20.0
    assert summary["q1"] == 15.0
    assert summary["q3"] == 25.0
    assert summary["summary"] == "20 +/- 10; median 20 [15, 25]"


def test_summarize_categorical_variable_collapses_rare_levels_and_adds_missing_row():
    df = pd.DataFrame({"stage": ["III", "I", "III", "II", "IV", None]})

    summary = summarize_categorical_variable(df, "stage", max_levels=2, include_missing=True)

    assert summary["level"].tolist() == ["III", "I", "Other (collapsed)", "Missing"]
    assert summary["count"].tolist() == [2, 1, 2, 1]
    assert summary["percent"].tolist() == [33.33, 16.67, 33.33, 16.67]
    assert summary["summary"].tolist() == ["2 (33.33%)", "1 (16.67%)", "2 (33.33%)", "1 (16.67%)"]


def test_grouped_summaries_exclude_missing_group_values():
    df = pd.DataFrame(
        {
            "arm": ["A", "A", "B", "B", None],
            "age": [50, 70, 60, None, 90],
            "stage": ["I", "II", "I", None, "III"],
        }
    )

    continuous = summarize_continuous_by_group(df, "age", "arm")
    categorical = summarize_categorical_by_group(df, "stage", "arm", include_missing=True)

    assert continuous["group"].tolist() == ["A", "B"]
    assert continuous.loc[continuous["group"] == "A", "summary"].iloc[0] == "60 +/- 14.14; median 60 [55, 65]"
    assert set(categorical["group"]) == {"A", "B"}
    assert categorical.loc[
        (categorical["group"] == "B") & (categorical["level"] == "Missing"),
        "summary",
    ].iloc[0] == "1 (50.00%)"


def test_build_baseline_table_creates_overall_and_group_columns_with_labels():
    df = pd.DataFrame(
        {
            "age": [50, 70, 60, 80],
            "sex": [1, 1, 2, 2],
            "stage": ["I", "II", "I", None],
            "time": [10, 20, 30, 40],
            "status": [1, 0, 1, 0],
        }
    )
    config = SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        group_col="sex",
    )
    survival_ready_df = create_survival_ready_dataframe(df, config)
    assert len(survival_ready_df) == 4

    table = build_baseline_table(
        df,
        continuous_vars=["age"],
        categorical_vars=["stage"],
        group_col="sex",
        include_missing=True,
        group_value_labels={"1": "Group 1", "2": "Group 2"},
    )

    assert table.columns.tolist() == ["Variable", "Overall", "Group 1", "Group 2"]
    assert table.loc[table["Variable"] == "age", "Overall"].iloc[0] == "65 +/- 12.91; median 65 [57.50, 72.50]"
    assert table.loc[table["Variable"] == "stage = I", "Group 1"].iloc[0] == "1 (50.00%)"
    assert table.loc[table["Variable"] == "stage = Missing", "Group 2"].iloc[0] == "1 (50.00%)"


def test_build_baseline_table_does_not_repeat_grouping_variable():
    df = pd.DataFrame(
        {
            "sex": ["F", "M", "F", "M"],
            "age": [50, 60, 70, 80],
        }
    )

    table = build_baseline_table(
        df,
        continuous_vars=["age"],
        categorical_vars=["sex"],
        group_col="sex",
    )

    assert table["Variable"].tolist() == [
        "n",
        "Events, n (%)",
        "Observed duration, median [IQR]",
        "age",
    ]
    assert table.columns.tolist() == ["Variable", "Overall", "F", "M"]


def test_build_baseline_table_starts_with_patient_event_and_followup_rows():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P3", None],
            "arm": ["A", "A", "A", "B", "B"],
        }
    )
    survival_df = pd.DataFrame(
        {
            "_time": [10, 20, 30, 40],
            "_event": [1, 0, 1, 0],
            "_group": ["A", "A", "B", "B"],
        }
    )

    table = build_baseline_table(
        df,
        continuous_vars=[],
        categorical_vars=[],
        group_col="arm",
        survival_ready_df=survival_df,
        id_col="patient_id",
    )

    assert table["Variable"].tolist() == [
        "n",
        "Events, n (%)",
        "Observed duration, median [IQR]",
    ]
    assert table.loc[table["Variable"] == "n"].iloc[0].to_dict() == {
        "Variable": "n",
        "Overall": "3",
        "A": "2",
        "B": "1",
    }
    assert table.loc[table["Variable"] == "Events, n (%)", "Overall"].iloc[0] == "2 (50.00%)"
    assert table.loc[table["Variable"] == "Events, n (%)", "A"].iloc[0] == "1 (50.00%)"
    assert table.loc[table["Variable"] == "Observed duration, median [IQR]", "B"].iloc[0] == "35 [32.50, 37.50]"


def test_real_other_level_is_not_overwritten_by_collapsed_levels():
    df = pd.DataFrame(
        {"stage": ["Other", "Other", "A", "B", "C", "D"]}
    )

    summary = summarize_categorical_variable(
        df,
        "stage",
        max_levels=2,
        include_missing=False,
    )

    assert "Other" in summary["level"].tolist()
    assert "Other (collapsed)" in summary["level"].tolist()
    assert int(summary["count"].sum()) == 6


def test_table1_uses_global_category_collapse_for_every_group():
    df = pd.DataFrame(
        {
            "arm": ["A"] * 6 + ["B"] * 6,
            "stage": ["I", "I", "I", "II", "III", "IV"]
            + ["I", "II", "II", "II", "V", "VI"],
        }
    )

    table = build_baseline_table(
        df,
        continuous_vars=[],
        categorical_vars=["stage"],
        group_col="arm",
        max_levels=2,
        include_missing=False,
    )

    assert set(table["Variable"]) >= {
        "stage = I",
        "stage = II",
        "stage = Other (collapsed)",
    }
    collapsed = table.loc[
        table["Variable"] == "stage = Other (collapsed)"
    ].iloc[0]
    assert collapsed["A"] == "2 (33.33%)"
    assert collapsed["B"] == "2 (33.33%)"


def test_table1_uses_one_record_per_patient_for_all_baseline_statistics():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P1", "P2"],
            "age": [50, 50, 50, 80],
            "arm": ["A", "A", "A", "B"],
        }
    )

    table = build_baseline_table(
        df,
        continuous_vars=["age"],
        categorical_vars=[],
        group_col="arm",
        id_col="patient_id",
    )

    assert table.loc[table["Variable"] == "n", "Overall"].iloc[0] == "2"
    assert table.loc[table["Variable"] == "age", "Overall"].iloc[0].startswith(
        "65 +/-"
    )


def test_mixed_group_values_and_duplicate_custom_labels_get_distinct_columns():
    df = pd.DataFrame(
        {
            "arm": [1, 1, "1", "1"],
            "age": [50, 60, 70, 80],
        }
    )

    table = build_baseline_table(
        df,
        continuous_vars=["age"],
        categorical_vars=[],
        group_col="arm",
        group_value_labels={"1": "Same"},
    )

    group_columns = [
        column for column in table.columns if column not in {"Variable", "Overall"}
    ]
    assert len(group_columns) == 2
    assert len(set(group_columns)) == 2
