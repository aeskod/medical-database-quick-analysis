import pandas as pd
import plotly.graph_objects as go

from src.survival_analysis import (
    fit_km_by_group,
    fit_km_overall,
    get_survival_summary,
    suggest_timepoints,
    survival_probability_at_times,
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
