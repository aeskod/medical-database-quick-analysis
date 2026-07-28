from dataclasses import asdict, fields
from datetime import date, datetime, timezone
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.column_annotations import ColumnAnnotation, USE_KEYS
from src.survival_analysis import compute_overall_survival_summary_table
from src.survival_mapping import SurvivalConfig


CONFIG_VERSION = 1
MAX_CONFIG_BYTES = 1_000_000
ALLOWED_TIME_UNITS = {"unknown", "days", "weeks", "months", "years"}
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")
PDF_CELL_LIMIT = 600


def serialize_analysis_configuration(
    config: SurvivalConfig,
    annotations: Mapping[str, Any],
) -> bytes:
    payload = {
        "version": CONFIG_VERSION,
        "survival_mapping": _json_safe(asdict(config)),
        "annotations": {
            str(column): {
                "meaning": str(getattr(annotation, "meaning", "Ignore column")),
                "uses": sorted(getattr(annotation, "uses", [])),
                "custom_meaning": str(getattr(annotation, "custom_meaning", "")),
            }
            for column, annotation in annotations.items()
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def dataframe_to_safe_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a table while neutralizing spreadsheet formula execution."""
    safe = df.copy()
    safe.columns = [_safe_spreadsheet_text(str(column)) for column in safe.columns]
    object_columns = safe.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        safe[column] = safe[column].map(
            lambda value: (
                _safe_spreadsheet_text(value) if isinstance(value, str) else value
            )
        )
    return safe.to_csv(index=False).encode("utf-8")


def deserialize_analysis_configuration(
    data: bytes | str,
) -> tuple[SurvivalConfig, dict[str, ColumnAnnotation]]:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("Configuration file is too large.")
    try:
        payload = json.loads(
            raw.decode("utf-8-sig"),
            parse_constant=lambda value: (_raise_invalid_json_number(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Configuration must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict) or payload.get("version") != CONFIG_VERSION:
        raise ValueError(f"Configuration version must be {CONFIG_VERSION}.")

    mapping = payload.get("survival_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Configuration is missing survival_mapping.")
    field_names = {field.name for field in fields(SurvivalConfig)}
    unknown_fields = set(mapping) - field_names
    if unknown_fields:
        raise ValueError("Unsupported mapping fields: " + ", ".join(sorted(unknown_fields)))
    for required in ("time_col", "event_col", "event_values", "censor_values"):
        if required not in mapping:
            raise ValueError(f"Configuration is missing mapping field '{required}'.")
    if not isinstance(mapping["event_values"], list) or not isinstance(
        mapping["censor_values"], list
    ):
        raise ValueError("Event and censor values must be JSON arrays.")
    optional_columns = {
        "time_col",
        "event_col",
        "id_col",
        "group_col",
        "start_date_col",
        "event_date_col",
        "last_followup_date_col",
    }
    for field_name in optional_columns:
        value = mapping.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Mapping field '{field_name}' must be text or null.")
    for field_name in ("event_values", "censor_values"):
        if any(isinstance(value, (dict, list)) for value in mapping[field_name]):
            raise ValueError(f"Mapping field '{field_name}' must contain scalar values.")
    if not isinstance(mapping.get("time_unit", "unknown"), str):
        raise ValueError("Mapping field 'time_unit' must be text.")
    if mapping.get("time_unit", "unknown") not in ALLOWED_TIME_UNITS:
        raise ValueError("Unsupported time unit in configuration.")
    if not isinstance(mapping.get("time_source", "duration"), str):
        raise ValueError("Mapping field 'time_source' must be text.")
    if not isinstance(mapping.get("missing_event_handling", "exclude"), str):
        raise ValueError("Mapping field 'missing_event_handling' must be text.")
    if not isinstance(mapping.get("unmapped_event_handling", "exclude"), str):
        raise ValueError("Mapping field 'unmapped_event_handling' must be text.")
    config = SurvivalConfig(**mapping)

    raw_annotations = payload.get("annotations")
    if not isinstance(raw_annotations, dict):
        raise ValueError("Configuration is missing annotations.")
    annotations = {}
    for column, value in raw_annotations.items():
        if not isinstance(column, str) or not isinstance(value, dict):
            raise ValueError("Each annotation must be an object keyed by column name.")
        meaning = value.get("meaning")
        uses = value.get("uses", [])
        custom_meaning = value.get("custom_meaning", "")
        if not isinstance(meaning, str) or not isinstance(custom_meaning, str):
            raise ValueError(f"Annotation text for '{column}' is invalid.")
        if not isinstance(uses, list) or any(use not in USE_KEYS for use in uses):
            raise ValueError(f"Annotation uses for '{column}' are invalid.")
        annotations[column] = ColumnAnnotation(
            column,
            meaning,
            frozenset(uses),
            custom_meaning,
        )
    return config, annotations


def build_html_report(
    df: pd.DataFrame,
    config: SurvivalConfig,
    annotations: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    survival_ready_df: pd.DataFrame,
) -> bytes:
    tables = _report_tables(df, config, annotations, quality_report, survival_ready_df)
    sections = "".join(
        f"<section><h2>{escape(title)}</h2>{table.to_html(index=False, escape=True)}</section>"
        for title, table in tables
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Medical Dataset Report</title>"
        "<style>body{font:14px Arial,sans-serif;max-width:1100px;margin:40px auto;color:#17202a}"
        "h1{margin-bottom:4px}h2{margin-top:32px;border-bottom:2px solid #2874a6;padding-bottom:6px}"
        "table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #ccd1d1;"
        "padding:7px;text-align:left;vertical-align:top}th{background:#eaf2f8}"
        ".meta{color:#566573}</style></head><body>"
        f"<h1>Medical Dataset Report</h1><p class='meta'>Generated {generated}</p>{sections}"
        "</body></html>"
    ).encode("utf-8")


def build_pdf_report(
    df: pd.DataFrame,
    config: SurvivalConfig,
    annotations: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    survival_ready_df: pd.DataFrame,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        CondPageBreak,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_name, bold_font_name = _register_unicode_pdf_fonts(
        pdfmetrics,
        TTFont,
    )

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="Medical Dataset Report",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=bold_font_name,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1B4F72"),
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName=bold_font_name,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1B4F72"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8,
        leading=10,
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story = [
        Paragraph("Medical Dataset Report", title_style),
        Paragraph(f"Generated {generated}", body_style),
        Spacer(1, 8),
    ]
    tables = _report_tables(df, config, annotations, quality_report, survival_ready_df)
    for index, (title, frame) in enumerate(tables):
        if index == 3:
            story.append(PageBreak())
        elif index:
            story.append(CondPageBreak(35 * mm))
        data = [
            [Paragraph(_pdf_text(column), body_style) for column in frame.columns],
            *[
                [Paragraph(_pdf_text(value), body_style) for value in row]
                for row in frame.itertuples(index=False, name=None)
            ],
        ]
        table = Table(
            data,
            colWidths=[document.width / len(frame.columns)] * len(frame.columns),
            repeatRows=1,
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6EAF8")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#154360")),
                    ("FONTNAME", (0, 0), (-1, 0), bold_font_name),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAB7B8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.extend([Paragraph(title, heading_style), table, Spacer(1, 8)])

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.drawRightString(landscape(A4)[0] - 15 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return buffer.getvalue()


def _report_tables(
    df: pd.DataFrame,
    config: SurvivalConfig,
    annotations: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    survival_ready_df: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    overview = quality_report.get("overview", {})
    overview_df = pd.DataFrame(
        [
            ("Rows", overview.get("n_rows", len(df))),
            ("Columns", overview.get("n_columns", len(df.columns))),
            ("Missing cells", overview.get("missing_cells", int(df.isna().sum().sum()))),
            ("Complete rows", overview.get("complete_rows", int(df.notna().all(axis=1).sum()))),
        ],
        columns=["Metric", "Value"],
    )
    mapping_df = pd.DataFrame(
        [
            (field, _display_value(value))
            for field, value in asdict(config).items()
        ],
        columns=["Mapping field", "Value"],
    )
    issue_rows = [
        (
            str(getattr(issue, "severity", "")),
            str(getattr(issue, "category", "")),
            str(getattr(issue, "message", "")),
            getattr(issue, "affected_rows_count", None),
        )
        for issue in quality_report.get("issues", [])
    ]
    issues_df = pd.DataFrame(
        issue_rows or [("none", "", "No data-quality issues detected.", None)],
        columns=["Severity", "Category", "Message", "Affected rows"],
    )
    annotation_df = pd.DataFrame(
        [
            (
                str(column),
                str(getattr(annotation, "resolved_meaning", getattr(annotation, "meaning", ""))),
                ", ".join(sorted(getattr(annotation, "uses", []))),
            )
            for column, annotation in annotations.items()
        ],
        columns=["Column", "Meaning", "Analysis uses"],
    )
    survival_df = compute_overall_survival_summary_table(
        survival_ready_df,
        config.time_unit,
    )
    return [
        ("Dataset overview", overview_df),
        ("Survival mapping", mapping_df),
        ("Data-quality findings", issues_df),
        ("Column annotations", annotation_df),
        ("Overall survival summary", survival_df),
    ]


def _display_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_display_value(item) for item in value) or "None"
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return "None"
    return str(value)


def _pdf_text(value: Any) -> str:
    text = _display_value(value)
    if len(text) > PDF_CELL_LIMIT:
        text = text[: PDF_CELL_LIMIT - 1] + "…"
    escaped = escape(text).replace("\n", "<br/>")
    return "&#8203;".join(
        escaped[index : index + 40]
        for index in range(0, len(escaped), 40)
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or (not isinstance(value, (dict, list, tuple, set)) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _safe_spreadsheet_text(value: str) -> str:
    if value.lstrip(" \ufeff").startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def _raise_invalid_json_number(value: str) -> None:
    raise ValueError(f"Configuration contains invalid JSON number '{value}'.")


def _register_unicode_pdf_fonts(pdfmetrics: Any, tt_font: Any) -> tuple[str, str]:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        ),
        (
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
    ]
    for regular_path, bold_path in candidates:
        if not regular_path.exists() or not bold_path.exists():
            continue
        try:
            pdfmetrics.registerFont(tt_font("ReportFont", str(regular_path)))
            pdfmetrics.registerFont(tt_font("ReportFontBold", str(bold_path)))
            return "ReportFont", "ReportFontBold"
        except Exception:
            continue
    try:
        pdfmetrics.registerFont(tt_font("ReportFont", "Vera.ttf"))
        pdfmetrics.registerFont(tt_font("ReportFontBold", "VeraBd.ttf"))
        return "ReportFont", "ReportFontBold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"
