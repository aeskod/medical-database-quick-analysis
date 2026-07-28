from io import BytesIO

import pandas as pd
import pytest

from src.data_loading import (
    DatasetLoadResult,
    read_dataset,
    read_dataset_with_metadata,
)


class NamedBytesIO(BytesIO):
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


def test_csv_loader_reports_metadata_and_detects_semicolon_delimiter():
    uploaded = NamedBytesIO(
        b"time;status;group\n10;1;A\n20;0;B\n",
        "clinical.csv",
    )

    result = read_dataset_with_metadata(uploaded)

    assert isinstance(result, DatasetLoadResult)
    assert result.metadata.file_name == "clinical.csv"
    assert result.metadata.file_size_bytes == len(uploaded.getvalue())
    assert result.metadata.format_name == "CSV"
    assert result.metadata.delimiter == ";"
    assert result.metadata.encoding == "utf-8"
    pd.testing.assert_frame_equal(
        result.dataframe,
        pd.DataFrame(
            {
                "time": [10, 20],
                "status": [1, 0],
                "group": ["A", "B"],
            }
        ),
    )


def test_txt_loader_detects_delimiter_and_windows_1252_encoding():
    content = "time|status|city\n10|1|Málaga\n20|0|Zürich\n".encode("cp1252")
    uploaded = NamedBytesIO(content, "clinical.txt")

    result = read_dataset_with_metadata(uploaded)

    assert result.metadata.format_name == "TXT"
    assert result.metadata.delimiter == "|"
    assert result.metadata.encoding == "cp1252"
    assert result.dataframe["city"].tolist() == ["Málaga", "Zürich"]


def test_tsv_loader_reports_tab_and_utf8_bom():
    content = "time\tstatus\n10\t1\n20\t0\n".encode("utf-8-sig")

    result = read_dataset_with_metadata(NamedBytesIO(content, "clinical.tsv"))

    assert result.metadata.format_name == "TSV"
    assert result.metadata.delimiter == "\t"
    assert result.metadata.encoding == "utf-8-sig"
    assert result.dataframe.columns.tolist() == ["time", "status"]


def test_txt_loader_supports_a_single_column_when_no_delimiter_is_detected():
    result = read_dataset_with_metadata(
        NamedBytesIO(b"patient_id\nP-001\nP-002\n", "ids.txt")
    )

    assert result.metadata.delimiter is None
    assert result.dataframe["patient_id"].tolist() == ["P-001", "P-002"]


def test_excel_loader_reports_binary_format_metadata():
    content = BytesIO()
    expected = pd.DataFrame({"time": [10, 20], "status": [1, 0]})
    expected.to_excel(content, index=False)
    uploaded = NamedBytesIO(content.getvalue(), "clinical.xlsx")

    result = read_dataset_with_metadata(uploaded)

    assert result.metadata.format_name == "Excel"
    assert result.metadata.delimiter is None
    assert result.metadata.encoding is None
    pd.testing.assert_frame_equal(result.dataframe, expected)


def test_existing_dataframe_only_loader_remains_compatible():
    uploaded = NamedBytesIO(b"time,status\n10,1\n20,0\n", "clinical.csv")

    dataframe = read_dataset(uploaded)

    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape == (2, 2)


def test_unsupported_extension_error_mentions_txt_support():
    with pytest.raises(ValueError, match="CSV, TSV, TXT, or XLSX"):
        read_dataset_with_metadata(NamedBytesIO(b"a,b\n1,2\n", "clinical.json"))


def test_ragged_csv_is_rejected_instead_of_silently_using_first_field_as_index():
    uploaded = NamedBytesIO(b"a,b\n1,2,3\n4,5,6\n", "ragged.csv")

    with pytest.raises(ValueError, match=r"Row 2 has 3 fields; expected 2"):
        read_dataset_with_metadata(uploaded)


def test_identifier_columns_keep_leading_zeroes_and_large_integer_precision():
    uploaded = NamedBytesIO(
        (
            "patient_id,time,status\n"
            "00123,10,1\n"
            "9007199254740993,20,0\n"
        ).encode(),
        "ids.csv",
    )

    result = read_dataset_with_metadata(uploaded)

    assert result.dataframe["patient_id"].tolist() == [
        "00123",
        "9007199254740993",
    ]
    assert result.dataframe["time"].tolist() == [10, 20]


def test_utf32_bom_is_detected_before_utf16_bom():
    content = "time,status\n10,1\n".encode("utf-32")

    result = read_dataset_with_metadata(NamedBytesIO(content, "utf32.csv"))

    assert result.metadata.encoding == "utf-32"
    assert result.dataframe.to_dict("records") == [{"time": 10, "status": 1}]


def test_manual_encoding_override_supports_gb18030():
    content = "time,city\n10,北京\n".encode("gb18030")

    result = read_dataset_with_metadata(
        NamedBytesIO(content, "clinical.csv"),
        encoding_override="gb18030",
    )

    assert result.metadata.encoding == "gb18030"
    assert result.dataframe["city"].tolist() == ["北京"]


def test_duplicate_and_blank_headers_are_rejected():
    with pytest.raises(ValueError, match="Duplicate column headers"):
        read_dataset_with_metadata(NamedBytesIO(b"a,a\n1,2\n", "duplicate.csv"))

    with pytest.raises(ValueError, match="non-empty header"):
        read_dataset_with_metadata(NamedBytesIO(b"a,\n1,2\n", "blank.csv"))


def test_excel_sheet_can_be_selected_and_is_reported():
    content = BytesIO()
    with pd.ExcelWriter(content) as writer:
        pd.DataFrame({"value": [1]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"value": [2]}).to_excel(writer, sheet_name="Second", index=False)

    result = read_dataset_with_metadata(
        NamedBytesIO(content.getvalue(), "multi.xlsx"),
        sheet_name="Second",
    )

    assert result.metadata.sheet_name == "Second"
    assert result.metadata.available_sheets == ("First", "Second")
    assert result.dataframe["value"].tolist() == [2]


def test_binary_payload_renamed_to_csv_is_rejected():
    uploaded = NamedBytesIO(b"a,b\n1,\x00\x01\x02\n", "not-really.csv")

    with pytest.raises(ValueError, match="binary"):
        read_dataset_with_metadata(uploaded)


def test_excel_formula_text_is_preserved_when_no_cached_value_exists():
    from openpyxl import Workbook

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Data"
    worksheet.append(["time", "derived"])
    worksheet.append([10, "=A2*2"])
    content = BytesIO()
    workbook.save(content)

    result = read_dataset_with_metadata(
        NamedBytesIO(content.getvalue(), "formula.xlsx")
    )

    assert result.dataframe.loc[0, "derived"] == "=A2*2"


def test_excel_duplicate_headers_are_rejected_before_pandas_can_mangle_them():
    from openpyxl import Workbook

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["time", "time"])
    worksheet.append([1, 2])
    content = BytesIO()
    workbook.save(content)

    with pytest.raises(ValueError, match="Duplicate column headers"):
        read_dataset_with_metadata(
            NamedBytesIO(content.getvalue(), "duplicate.xlsx")
        )
