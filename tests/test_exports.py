import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.column_annotations import ColumnAnnotation, USE_FILTER, USE_GROUP
from src.data_quality import build_data_quality_report
from src.exports import (
    build_html_report,
    build_pdf_report,
    deserialize_analysis_configuration,
    serialize_analysis_configuration,
)
from src.profiling import profile_dataframe
from src.survival_mapping import (
    SurvivalConfig,
    create_cleaned_mapped_dataframe,
    create_survival_ready_dataframe,
)


def _dataset():
    return pd.DataFrame(
        {
            "time": [10, 20, None, 40],
            "status": ["Dead", "Alive", "Unknown", "Dead"],
            "sex": ["F", "M", "F", "M"],
            "<script>": [1, 2, 3, 4],
        }
    )


def _config():
    return SurvivalConfig(
        time_col="time",
        event_col="status",
        event_values=["Dead"],
        censor_values=["Alive"],
        group_col="sex",
        time_unit="days",
    )


def _annotations():
    return {
        "time": ColumnAnnotation("time", "Follow-up time / survival time"),
        "status": ColumnAnnotation("status", "Event status"),
        "sex": ColumnAnnotation(
            "sex",
            "Sex / gender",
            frozenset({USE_FILTER, USE_GROUP}),
        ),
        "<script>": ColumnAnnotation("<script>", "Other numeric variable"),
    }


def export_harness(df, profile):
    from app import _render_export_section

    _render_export_section(df, profile)


def test_cleaned_mapped_export_keeps_source_columns_and_standardized_values():
    result = create_cleaned_mapped_dataframe(_dataset(), _config())

    assert list(result.columns) == [
        "time",
        "status",
        "sex",
        "<script>",
        "_time",
        "_event",
        "_group",
    ]
    assert result["status"].tolist() == ["Dead", "Alive", "Dead"]
    assert result["_time"].tolist() == [10.0, 20.0, 40.0]
    assert result["_event"].tolist() == [1, 0, 1]
    assert result["_group"].tolist() == ["F", "M", "M"]


def test_configuration_round_trip_preserves_mapping_and_annotations():
    encoded = serialize_analysis_configuration(_config(), _annotations())

    config, annotations = deserialize_analysis_configuration(encoded)

    assert config == _config()
    assert annotations == _annotations()


def test_configuration_loader_rejects_unknown_annotation_uses():
    payload = json.loads(
        serialize_analysis_configuration(_config(), _annotations()).decode()
    )
    payload["annotations"]["sex"]["uses"] = ["execute"]

    with pytest.raises(ValueError, match="Annotation uses"):
        deserialize_analysis_configuration(json.dumps(payload))


def test_combined_reports_include_escaped_sections_and_valid_pdf_bytes():
    df = _dataset()
    config = _config()
    annotations = _annotations()
    ready = create_survival_ready_dataframe(df, config)
    quality = build_data_quality_report(
        df,
        profile_dataframe(df),
        config,
        ready,
        annotations,
    )

    html_report = build_html_report(df, config, annotations, quality, ready)
    pdf_report = build_pdf_report(df, config, annotations, quality, ready)

    html_text = html_report.decode()
    assert "Dataset overview" in html_text
    assert "Data-quality findings" in html_text
    assert "Overall survival summary" in html_text
    assert "&lt;script&gt;" in html_text
    assert "<td><script>" not in html_text
    assert pdf_report.startswith(b"%PDF-")
    assert pdf_report.rstrip().endswith(b"%%EOF")
    assert len(pdf_report) > 2_000


def test_export_component_prepares_all_downloads_on_demand():
    df = _dataset()
    config = _config()
    app_test = AppTest.from_function(
        export_harness,
        args=(df, profile_dataframe(df)),
    )
    app_test.session_state["survival_config"] = config
    app_test.session_state["survival_ready_df"] = create_survival_ready_dataframe(
        df,
        config,
    )
    app_test.session_state["column_annotations"] = _annotations()

    app_test.run(timeout=20)

    assert not app_test.exception
    assert {button.label for button in app_test.get("download_button")} == {
        "Download cleaned mapped data as CSV",
        "Save mapping and annotations as JSON",
    }
    prepare = next(
        button
        for button in app_test.button
        if button.label == "Prepare combined HTML and PDF reports"
    )

    app_test = prepare.click().run(timeout=20)

    assert not app_test.exception
    assert {
        "Download combined report as HTML",
        "Download combined report as PDF",
    }.issubset({button.label for button in app_test.get("download_button")})
    assert app_test.session_state["combined_report_pdf"].startswith(b"%PDF-")


def test_export_component_loads_configuration_into_current_dataset():
    df = _dataset()
    profile = profile_dataframe(df)
    saved = serialize_analysis_configuration(_config(), _annotations())
    app_test = AppTest.from_function(export_harness, args=(df, profile)).run(
        timeout=20
    )
    uploader = next(
        item
        for item in app_test.file_uploader
        if item.label == "Load mapping and annotation configuration"
    )
    app_test = uploader.upload(
        "analysis_configuration.json",
        saved,
        "application/json",
    ).run(timeout=20)
    apply_button = next(
        button
        for button in app_test.button
        if button.label == "Apply uploaded configuration"
    )

    app_test = apply_button.click().run(timeout=20)

    assert not app_test.exception
    assert app_test.session_state["survival_config"] == _config()
    assert app_test.session_state["column_annotations"] == _annotations()
    assert any(
        "Mapping and annotations loaded" in message.value
        for message in app_test.success
    )
