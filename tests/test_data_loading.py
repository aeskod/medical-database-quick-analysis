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
