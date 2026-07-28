import pandas as pd
import plotly.graph_objects as go
import pytest

from src.charts import (
    build_chart,
    build_chart_dataframe,
    get_chart_variable_type,
    plot_bar_chart,
    plot_box_plot,
    plot_correlation_heatmap,
    plot_histogram,
    plot_missingness_bar,
    plot_missingness_heatmap,
    plot_scatter,
    plot_time_series,
    plot_violin,
    prepare_categorical_series,
    recommend_chart_type,
    summarize_chart_variables,
)


def test_chart_variable_type_numeric():
    df = pd.DataFrame({"age": [50, 60, 70, 80, 90, 55, 65, 75, 85, 95, 100]})

    assert get_chart_variable_type("age", df) == "numeric"


def test_chart_variable_type_categorical():
    df = pd.DataFrame({"sex": ["M", "F", "M", "F"]})

    assert get_chart_variable_type("sex", df) == "categorical"


def test_numeric_coded_category():
    df = pd.DataFrame({"ph.ecog": [0, 1, 1, 2, 0, 1, 2]})

    assert get_chart_variable_type("ph.ecog", df) == "categorical"


def test_lung_like_variable_types():
    df = pd.DataFrame(
        {
            "inst": [1, 1, 2, 2, 3, 3],
            "time": [100, 200, 300, 400, 500, 600],
            "status": [0, 1, 1, 0, 1, 0],
            "age": [55, 63, 72, 70, 64, 58],
        }
    )

    assert get_chart_variable_type("inst", df) == "categorical"
    assert get_chart_variable_type("time", df) == "numeric"
    assert get_chart_variable_type("status", df) == "categorical"
    assert get_chart_variable_type("age", df) == "numeric"


def test_recommend_histogram():
    df = pd.DataFrame({"age": [50, 60, 70]})

    assert recommend_chart_type("age", None, df) == "histogram"


def test_recommend_bar():
    df = pd.DataFrame({"sex": ["M", "F", "M"]})

    assert recommend_chart_type("sex", None, df) == "bar"


def test_recommend_box_plot_in_both_column_orders():
    df = pd.DataFrame({"age": [50, 60, 70, 80], "sex": ["M", "F", "M", "F"]})

    assert recommend_chart_type("sex", "age", df) == "box"
    assert recommend_chart_type("age", "sex", df) == "box"


def test_recommend_scatter():
    df = pd.DataFrame({"age": [50, 60, 70], "meal_cal": [1000, 1100, 900]})

    assert recommend_chart_type("age", "meal_cal", df) == "scatter"


def test_recommend_stacked_bar():
    df = pd.DataFrame({"treatment": ["A", "B", "A"], "stage": ["I", "II", "I"]})

    assert recommend_chart_type("treatment", "stage", df) == "stacked_bar"


def test_prepare_categorical_series_collapses_rare_levels():
    df = pd.DataFrame({"diagnosis": ["A", "B", "C", "D", "E", "F"]})

    prepared = prepare_categorical_series(df, "diagnosis", max_levels=3)

    assert "Other" in set(prepared)
    assert prepared.nunique() <= 4


def test_prepare_categorical_series_preserves_real_other_level():
    df = pd.DataFrame(
        {"diagnosis": ["Other", "A", "B", "C", "Missing", None]}
    )

    prepared = prepare_categorical_series(df, "diagnosis", max_levels=2)

    assert "Other" in set(prepared)
    assert "Other (collapsed)" in set(prepared)
    assert "Missing" in set(prepared)


def test_build_chart_dataframe_does_not_mutate_input_and_drops_required_numeric_missing():
    df = pd.DataFrame(
        {
            "age": [50, None, 70],
            "sex": ["M", "F", None],
            "treatment": ["A", "B", "A"],
        }
    )
    original = df.copy(deep=True)

    chart_df = build_chart_dataframe(
        df,
        x_col="age",
        y_col="sex",
        color_col="treatment",
        include_missing=True,
    )

    assert chart_df["age"].tolist() == [50.0, 70.0]
    assert chart_df["sex"].tolist() == ["M", "Missing"]
    pd.testing.assert_frame_equal(df, original)


def test_histogram_includes_marginal_box_plot():
    df = pd.DataFrame({"age": [50, 60, 70, 80]})

    figure = plot_histogram(df, "age")

    assert isinstance(figure, go.Figure)
    assert {trace.type for trace in figure.data} == {"histogram", "box"}
    assert figure.layout.yaxis2.matches is None


def test_bar_chart_returns_figure():
    df = pd.DataFrame({"sex": ["M", "F", "M"]})

    assert isinstance(plot_bar_chart(df, "sex"), go.Figure)


def test_box_plot_returns_figure():
    df = pd.DataFrame({"age": [50, 60, 70, 80], "sex": ["M", "F", "M", "F"]})

    assert isinstance(plot_box_plot(df, numeric_col="age", category_col="sex"), go.Figure)


def test_violin_plot_supports_one_numeric_or_numeric_by_category():
    df = pd.DataFrame({"age": [50, 60, 70, 80], "sex": ["M", "F", "M", "F"]})

    numeric_figure = plot_violin(df, numeric_col="age")
    grouped_figure = plot_violin(df, numeric_col="age", category_col="sex")

    assert isinstance(numeric_figure, go.Figure)
    assert isinstance(grouped_figure, go.Figure)
    assert all(trace.type == "violin" for trace in numeric_figure.data)
    assert all(trace.type == "violin" for trace in grouped_figure.data)


def test_summarize_chart_variables_returns_numeric_and_categorical_statistics():
    df = pd.DataFrame(
        {
            "age": [50, 60, None, 80],
            "sex": ["F", "F", "M", None],
        }
    )

    summaries = summarize_chart_variables(df, ["age", "sex", "age"])

    assert list(summaries) == ["age", "sex"]
    assert summaries["age"] == {
        "Type": "Numeric",
        "Valid": "3",
        "Missing": "1",
        "Mean": "63.33",
        "SD": "15.28",
        "Median": "60",
        "IQR": "55–70",
        "Range": "50–80",
    }
    assert summaries["sex"] == {
        "Type": "Categorical",
        "Valid": "3",
        "Missing": "1",
        "Levels": "2",
        "Most common": "F — 2 (66.67%)",
    }


def test_scatter_returns_figure():
    df = pd.DataFrame({"age": [50, 60, 70, 80], "meal_cal": [1000, 1100, 900, 1200]})

    assert isinstance(plot_scatter(df, "age", "meal_cal"), go.Figure)


def test_scatter_color_groups_respect_max_category_levels():
    df = pd.DataFrame(
        {
            "time": list(range(13)),
            "meal.cal": [1000, 1100, 900, 1200, 950, 1050, 1150, 980, 1080, 990, 1010, 970, 1030],
            "ph.karno": [
                90.0,
                90.0,
                90.0,
                90.0,
                80.0,
                80.0,
                80.0,
                70.0,
                70.0,
                60.0,
                60.0,
                50.0,
                100.0,
            ],
        }
    )

    full_result = build_chart(
        df,
        chart_type="scatter",
        x_col="time",
        y_col="meal.cal",
        color_col="ph.karno",
        max_category_levels=10,
    )
    limited_result = build_chart(
        df,
        chart_type="scatter",
        x_col="time",
        y_col="meal.cal",
        color_col="ph.karno",
        max_category_levels=4,
    )

    full_legend = {trace.name for trace in full_result["fig"].data}
    limited_legend = {trace.name for trace in limited_result["fig"].data}

    assert "Other" not in full_legend
    assert "Other" in limited_legend
    assert len(limited_legend) < len(full_legend)
    assert any("ph.karno has many levels" in warning for warning in limited_result["warnings"])


def test_missingness_bar_returns_figure():
    df = pd.DataFrame({"age": [50, None, 70], "sex": ["M", "F", None]})

    assert isinstance(plot_missingness_bar(df), go.Figure)


def test_missingness_heatmap_marks_normalized_missing_values_and_caps_rows():
    df = pd.DataFrame(
        {
            "a": [1, None, 3],
            "b": ["NA", "x", None],
        }
    )

    figure = plot_missingness_heatmap(df, max_rows=2)

    assert isinstance(figure, go.Figure)
    assert figure.data[0].z.tolist() == [[0, 1], [1, 0]]
    assert list(figure.data[0].y) == ["a", "b"]
    assert list(figure.data[0].x) == ["0", "1"]
    assert any("Showing first 2 of 3 rows" in item.text for item in figure.layout.annotations)


def test_correlation_heatmap_requires_two_numeric_columns():
    df = pd.DataFrame({"age": [50, 60, 70], "sex": ["M", "F", "M"]})

    with pytest.raises(ValueError, match="requires at least two numeric variables"):
        plot_correlation_heatmap(df)

    result = build_chart(df, chart_type="correlation_heatmap")
    assert result["fig"] is None
    assert any("requires at least two numeric variables" in warning for warning in result["warnings"])


def test_correlation_heatmap_rejects_constant_numeric_columns():
    df = pd.DataFrame(
        {
            "age": [50, 60, 70],
            "constant": [1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="non-constant values"):
        plot_correlation_heatmap(df)


def test_build_chart_auto_returns_histogram_with_box_plot():
    df = pd.DataFrame({"age": [50, 60, 70], "sex": ["M", "F", "M"]})

    result = build_chart(df=df, chart_type="auto", x_col="age", y_col=None)

    assert result["chart_type"] == "histogram"
    assert isinstance(result["fig"], go.Figure)
    assert isinstance(result["warnings"], list)


def test_build_chart_returns_violin_for_valid_variables():
    df = pd.DataFrame({"age": [50, 60, 70, 80], "sex": ["M", "F", "M", "F"]})

    result = build_chart(
        df=df,
        chart_type="violin",
        x_col="age",
        y_col="sex",
    )

    assert result["chart_type"] == "violin"
    assert isinstance(result["fig"], go.Figure)
    assert all(trace.type == "violin" for trace in result["fig"].data)
    assert result["warnings"] == []


def test_datetime_and_numeric_variables_produce_time_series():
    df = pd.DataFrame(
        {
            "visit_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01"]
            ),
            "weight": [70.0, 69.5, 69.0],
        }
    )

    assert recommend_chart_type("visit_date", "weight", df) == "line"
    result = build_chart(
        df,
        chart_type="auto",
        x_col="visit_date",
        y_col="weight",
    )

    assert result["chart_type"] == "line"
    assert isinstance(result["fig"], go.Figure)
    assert isinstance(plot_time_series(df, "visit_date", "weight"), go.Figure)


def test_numeric_color_is_ignored_for_histogram_instead_of_creating_one_trace_per_value():
    df = pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "weight": [60 + value / 10 for value in range(20)],
        }
    )

    result = build_chart(
        df,
        chart_type="histogram",
        x_col="age",
        color_col="weight",
    )

    assert len(result["fig"].data) == 2  # histogram + marginal box
    assert any("Numeric color" in warning for warning in result["warnings"])
