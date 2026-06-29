import pandas as pd
import plotly.graph_objects as go
import numpy as np

from src.survival_analysis import (
    compute_group_survival_summary,
    compute_number_at_risk_table,
    detect_curve_crossing,
    format_group_label,
    format_p_value,
    format_survival_time,
    generate_survival_interpretation_warnings,
    fit_km_by_group,
    fit_km_overall,
    get_survival_summary,
    pivot_at_risk_table,
    pivot_survival_probability_table,
    run_logrank_test,
    run_pairwise_logrank_tests,
    suggest_timepoints,
    survival_probabilities_at_years,
    survival_probability_at_times,
    survival_probability_table_by_group,
    validate_survival_ready_dataframe,
)
from src.survival_plots import plot_km_curve


def test_validate_good_survival_dataframe():
    df = pd.DataFrame({"_time": [10, 20, 30, 40], "_event": [1, 0, 1, 0]})

    errors, warnings = validate_survival_ready_dataframe(df)

    assert errors == []
    assert any("Less than 10 usable rows" in warning for warning in warnings)


def test_validate_missing_columns():
    df = pd.DataFrame({"time": [10, 20], "event": [1, 0]})

    errors, warnings = validate_survival_ready_dataframe(df)

    assert any("_time column is missing" in error for error in errors)
    assert any("_event column is missing" in error for error in errors)
    assert warnings == []


def test_validate_negative_time_error():
    df = pd.DataFrame({"_time": [10, -5, 30], "_event": [1, 0, 1]})

    errors, _ = validate_survival_ready_dataframe(df)

    assert any("negative" in error for error in errors)


def test_get_survival_summary_statistics():
    df = pd.DataFrame({"_time": [10, 20, 30, 40], "_event": [1, 0, 1, 0]})

    summary = get_survival_summary(df, time_unit="days")

    assert summary["n"] == 4
    assert summary["events"] == 2
    assert summary["censored"] == 2
    assert summary["event_rate"] == 50.0
    assert summary["max_followup"] == 40
    assert summary["time_unit"] == "days"


def test_fit_km_overall_returns_expected_objects():
    df = pd.DataFrame({"_time": [10, 20, 30, 40], "_event": [1, 0, 1, 0]})

    result = fit_km_overall(df)

    assert result["kmf"] is not None
    assert set(["time", "survival", "ci_lower", "ci_upper", "group"]).issubset(
        result["curve"].columns
    )
    assert "median_survival" in result


def test_survival_probability_at_times():
    df = pd.DataFrame({"_time": [10, 20, 30, 40], "_event": [1, 0, 1, 0]})
    result = fit_km_overall(df)

    probabilities = survival_probability_at_times(result["kmf"], [10, 20, 30])

    assert len(probabilities) == 3
    assert probabilities["survival_probability"].between(0, 1).all()


def test_survival_probabilities_at_years_respects_units_and_followup():
    df = pd.DataFrame({"_time": [6, 12, 24, 36, 48], "_event": [1, 0, 1, 0, 1]})
    result = fit_km_overall(df)

    probabilities = survival_probabilities_at_years(result["kmf"], "months")

    assert probabilities[1] == round(float(result["kmf"].predict(12)), 4)
    assert probabilities[3] == round(float(result["kmf"].predict(36)), 4)
    assert probabilities[5] is None
    assert survival_probabilities_at_years(result["kmf"], "unknown") == {
        1: None,
        3: None,
        5: None,
    }


def test_fit_km_by_group_returns_two_groups():
    df = pd.DataFrame(
        {
            "_time": [10, 20, 30, 40, 50, 60],
            "_event": [1, 0, 1, 0, 1, 0],
            "_group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    group_results, warnings = fit_km_by_group(df)

    assert [result["label"] for result in group_results] == ["A", "B"]
    assert any("Group A has fewer than 5 rows" in warning for warning in warnings)
    assert any("Group B has fewer than 5 rows" in warning for warning in warnings)


def test_fit_km_by_group_too_many_groups_warning():
    df = pd.DataFrame(
        {
            "_time": list(range(1, 12)),
            "_event": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "_group": [f"G{index}" for index in range(11)],
        }
    )

    group_results, warnings = fit_km_by_group(df)

    assert group_results == []
    assert any("too many groups" in warning for warning in warnings)


def test_suggest_timepoints_uses_unit_defaults_and_quartiles():
    assert suggest_timepoints(400, "days") == [30.0, 90.0, 180.0, 365.0]
    assert suggest_timepoints(0.8, "years") == [0.2, 0.4, 0.6]
    assert suggest_timepoints(40, "unknown") == [10.0, 20.0, 30.0]


def test_plot_km_curve_returns_figure():
    curve_df = pd.DataFrame(
        {
            "time": [0, 10, 20],
            "survival": [1.0, 0.8, 0.6],
            "ci_lower": [1.0, 0.7, 0.5],
            "ci_upper": [1.0, 0.9, 0.8],
            "group": ["Overall", "Overall", "Overall"],
        }
    )

    figure = plot_km_curve(curve_df)

    assert isinstance(figure, go.Figure)


def test_plot_km_curve_censor_markers_can_be_toggled():
    curve_df = pd.DataFrame(
        {
            "time": [0, 10, 20],
            "survival": [1.0, 0.8, 0.8],
            "ci_lower": [1.0, 0.7, 0.7],
            "ci_upper": [1.0, 0.9, 0.9],
            "censored": [0, 0, 2],
            "group": ["Overall", "Overall", "Overall"],
        }
    )

    with_censors = plot_km_curve(curve_df)
    without_censors = plot_km_curve(curve_df, show_censors=False)

    marker_trace = next(trace for trace in with_censors.data if trace.mode == "markers")
    assert list(marker_trace.x) == [20]
    assert list(marker_trace.customdata) == [2]
    assert all(trace.mode != "markers" for trace in without_censors.data)


def test_format_p_value():
    assert format_p_value(0.0004) == "p < 0.001"
    assert format_p_value(0.03424) == "p = 0.0342"


def test_format_survival_time():
    assert format_survival_time(310, "days") == "310.00 days"
    assert format_survival_time(np.inf) == "Not reached"


def test_format_group_label():
    labels = {"sex": {"1": "Male", "2": "Female"}}

    assert format_group_label(1, "sex", labels) == "Male"
    assert format_group_label(1, "sex", None) == "sex = 1"


def test_compute_group_survival_summary():
    df = pd.DataFrame(
        {
            "_time": [10, 20, 30, 40, 50, 60],
            "_event": [1, 0, 1, 0, 1, 0],
            "_group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    summary = compute_group_survival_summary(df)

    assert len(summary) == 2
    assert set(summary["group"]) == {"A", "B"}
    assert int(summary.loc[summary["group"] == "A", "n"].iloc[0]) == 3
    assert int(summary.loc[summary["group"] == "B", "n"].iloc[0]) == 3
    assert int(summary.loc[summary["group"] == "A", "events"].iloc[0]) == 2
    assert int(summary.loc[summary["group"] == "B", "events"].iloc[0]) == 1


def test_run_logrank_test_available():
    df = pd.DataFrame(
        {
            "_time": [10, 20, 30, 40, 50, 60],
            "_event": [1, 0, 1, 0, 1, 0],
            "_group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    result = run_logrank_test(df)

    assert result["available"] is True
    assert "p_value" in result
    assert "test_statistic" in result
    assert result["n_groups"] == 2


def test_run_logrank_test_unavailable_with_one_group():
    df = pd.DataFrame({"_time": [10, 20, 30], "_event": [1, 0, 1], "_group": ["A", "A", "A"]})

    result = run_logrank_test(df)

    assert result["available"] is False
    assert "At least two groups" in result["reason"]


def test_run_pairwise_logrank_tests_for_three_groups():
    df = pd.DataFrame(
        {
            "_time": [10, 20, 30, 40, 50, 60, 70, 80, 90],
            "_event": [1, 0, 1, 0, 1, 0, 1, 0, 1],
            "_group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        }
    )

    result = run_pairwise_logrank_tests(df)

    assert len(result) == 3
    assert set(zip(result["group_1"], result["group_2"])) == {("A", "B"), ("A", "C"), ("B", "C")}


def test_run_pairwise_logrank_tests_empty_for_two_groups():
    df = pd.DataFrame({"_time": [10, 20, 30, 40], "_event": [1, 0, 1, 0], "_group": ["A", "A", "B", "B"]})

    result = run_pairwise_logrank_tests(df)

    assert result.empty


def test_compute_number_at_risk_table_overall():
    df = pd.DataFrame({"_time": [10, 20, 30, 40], "_event": [1, 0, 1, 0]})

    result = compute_number_at_risk_table(df, [0, 20, 40], group_col=None)

    assert int(result.loc[result["time"] == 0, "at_risk"].iloc[0]) == 4
    assert int(result.loc[result["time"] == 20, "at_risk"].iloc[0]) == 3
    assert int(result.loc[result["time"] == 40, "at_risk"].iloc[0]) == 1


def test_compute_number_at_risk_table_grouped():
    df = pd.DataFrame(
        {
            "_time": [10, 20, 30, 40],
            "_event": [1, 0, 1, 0],
            "_group": ["A", "A", "B", "B"],
        }
    )

    result = compute_number_at_risk_table(df, [0, 20])

    assert int(result[(result["group"] == "A") & (result["time"] == 0)]["at_risk"].iloc[0]) == 2
    assert int(result[(result["group"] == "A") & (result["time"] == 20)]["at_risk"].iloc[0]) == 1
    assert int(result[(result["group"] == "B") & (result["time"] == 0)]["at_risk"].iloc[0]) == 2
    assert int(result[(result["group"] == "B") & (result["time"] == 20)]["at_risk"].iloc[0]) == 2


def test_pivot_at_risk_table():
    at_risk_df = pd.DataFrame(
        {
            "group": ["Overall", "Overall", "A", "A"],
            "time": [0, 20, 0, 20],
            "at_risk": [4, 3, 2, 1],
            "events_up_to_time": [0, 1, 0, 1],
            "censored_up_to_time": [0, 1, 0, 1],
        }
    )

    result = pivot_at_risk_table(at_risk_df)

    assert "Group" in result.columns
    assert 0 in result.columns
    assert 20 in result.columns
    assert int(result.loc[result["Group"] == "Overall", 20].iloc[0]) == 3


def test_survival_probability_table_by_group():
    df = pd.DataFrame({"_time": [10, 20, 30, 40], "_event": [1, 0, 1, 0]})
    overall = fit_km_overall(df, label="Overall")
    group_a = fit_km_overall(df.iloc[:2], label="A")
    group_b = fit_km_overall(df.iloc[2:], label="B")

    result = survival_probability_table_by_group([overall, group_a, group_b], [10, 20, 30])

    assert set(result.columns) == {"group", "time", "survival_probability"}
    assert result["survival_probability"].between(0, 1).all()


def test_pivot_survival_probability_table():
    prob_df = pd.DataFrame(
        {
            "group": ["Overall", "Overall", "A", "A"],
            "time": [10, 20, 10, 20],
            "survival_probability": [0.9, 0.8, 0.85, 0.7],
        }
    )

    result = pivot_survival_probability_table(prob_df)

    assert "Group" in result.columns
    assert 10 in result.columns
    assert 20 in result.columns
    assert float(result.loc[result["Group"] == "A", 20].iloc[0]) == 0.7


def test_detect_curve_crossing():
    curve_df = pd.DataFrame(
        {
            "time": [0, 1, 2, 0, 1, 2],
            "survival": [1.0, 0.8, 0.4, 1.0, 0.7, 0.6],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    assert detect_curve_crossing(curve_df) is True


def test_survival_warning_for_small_group():
    df = pd.DataFrame(
        {
            "_time": [10, 20, 30, 40],
            "_event": [1, 0, 1, 0],
            "_group": ["A", "A", "A", "B"],
        }
    )

    warnings = generate_survival_interpretation_warnings(df, min_group_size=2, min_events_per_group=1)

    assert any("fewer than 2 rows" in warning for warning in warnings)


def test_survival_warning_for_few_events():
    df = pd.DataFrame(
        {
            "_time": [10, 20, 30, 40, 50, 60],
            "_event": [0, 0, 0, 0, 1, 0],
            "_group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    warnings = generate_survival_interpretation_warnings(df, min_group_size=1, min_events_per_group=2)

    assert any("fewer than 2 events" in warning for warning in warnings)
