from streamlit.testing.v1 import AppTest


def _txt_dataset(row_count: int = 35) -> bytes:
    rows = ["time;status;note"]
    rows.extend(
        f"{index};{index % 2};{'NA' if index == 3 else f'note {index}'}"
        for index in range(1, row_count + 1)
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


def test_txt_upload_reports_metadata_types_and_missing_highlighting():
    app_test = AppTest.from_file("app.py").run(timeout=30)

    assert ".txt" in app_test.file_uploader[0].allowed_type
    app_test.file_uploader[0].upload(
        "clinical.txt",
        _txt_dataset(),
        "text/plain",
    ).run(timeout=30)

    assert not app_test.exception
    summary_metrics = {
        metric.label: metric.value
        for metric in app_test.metric[:4]
    }
    assert summary_metrics == {
        "File": "clinical.txt",
        "Size": "450 B",
        "Rows": "35",
        "Columns": "3",
    }
    assert any(
        "**Detected format:** TXT" in block.value
        and "**Detected delimiter:** Semicolon (;)" in block.value
        and "**Detected encoding:** utf-8" in block.value
        for block in app_test.markdown
    )

    preview = app_test.dataframe[0]
    assert preview.value.shape == (20, 3)
    assert preview.value.columns.tolist() == [
        "time · integer",
        "status · binary",
        "note · text",
    ]
    assert any(
        caption.value == "Missing values are highlighted in red."
        for caption in app_test.caption
    )
    show_more = next(
        checkbox
        for checkbox in app_test.checkbox
        if checkbox.label == "Show more rows"
    )
    assert show_more.value is False
    assert show_more.disabled is False


def test_show_more_rows_expands_preview_and_resets_for_replacement_dataset():
    app_test = AppTest.from_file("app.py").run(timeout=30)
    app_test.file_uploader[0].upload(
        "clinical.txt",
        _txt_dataset(),
        "text/plain",
    ).run(timeout=30)
    show_more = next(
        checkbox
        for checkbox in app_test.checkbox
        if checkbox.label == "Show more rows"
    )

    show_more.set_value(True).run(timeout=30)

    assert not app_test.exception
    assert app_test.dataframe[0].value.shape == (35, 3)
    assert any(
        caption.value == "Showing 35 of 35 rows. Column types appear in the headers."
        for caption in app_test.caption
    )

    replacement = b"time,status\n90,0\n80,1\n"
    app_test.file_uploader[0].clear().upload(
        "clinical.csv",
        replacement,
        "text/csv",
    ).run(timeout=30)

    reset_control = next(
        checkbox
        for checkbox in app_test.checkbox
        if checkbox.label == "Show more rows"
    )
    assert reset_control.value is False
    assert reset_control.disabled is True
    assert app_test.dataframe[0].value.shape == (2, 2)
