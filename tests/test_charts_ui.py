import pandas as pd
from streamlit.testing.v1 import AppTest

from src.column_annotations import ColumnAnnotation, USE_CHARTS
from src.profiling import profile_dataframe


def charts_harness():
    from app import _render_charts_tab

    _render_charts_tab()


def test_charts_ui_shows_combined_auto_chart_violin_option_and_summary_statistics():
    df = pd.DataFrame(
        {
            "age": [50, 60, 70, 80],
            "sex": ["F", "M", "F", "M"],
        }
    )
    app_test = AppTest.from_function(charts_harness)
    app_test.session_state["uploaded_df"] = df
    app_test.session_state["profile_df"] = profile_dataframe(df)
    app_test.session_state["column_annotations"] = {
        column: ColumnAnnotation(column, meaning, frozenset({USE_CHARTS}))
        for column, meaning in {"age": "Age", "sex": "Sex / gender"}.items()
    }

    app_test.run(timeout=20)

    assert not app_test.exception
    chart_type = next(
        item for item in app_test.selectbox if item.label == "Chart type"
    )
    assert "Histogram + box plot" in chart_type.options
    assert "Violin plot" in chart_type.options
    assert any(
        "Suggested chart: Histogram + box plot" in item.value
        for item in app_test.info
    )
    assert "Summary statistics" in [item.value for item in app_test.subheader]
    summary_table = next(
        dataframe.value
        for dataframe in app_test.dataframe
        if dataframe.value.columns.tolist() == ["Statistic", "Value"]
    )
    assert summary_table.iloc[0].to_dict() == {
        "Statistic": "Type",
        "Value": "Numeric",
    }

    app_test = chart_type.set_value("Violin plot").run(timeout=20)
    y_variable = next(
        item for item in app_test.selectbox if item.label == "Y variable"
    )
    app_test = y_variable.set_value("sex").run(timeout=20)

    assert not app_test.exception
    assert next(
        item.value for item in app_test.selectbox if item.label == "Chart type"
    ) == "Violin plot"
    assert next(
        item.value for item in app_test.selectbox if item.label == "Y variable"
    ) == "sex"
    assert len(app_test.get("plotly_chart")) == 1
    image_format = next(
        item
        for item in app_test.radio
        if item.label == "Plot image download format"
    )
    assert image_format.options == ["PNG", "SVG"]

    app_test = image_format.set_value("SVG").run(timeout=20)

    assert not app_test.exception
    assert next(
        item.value
        for item in app_test.radio
        if item.label == "Plot image download format"
    ) == "SVG"
