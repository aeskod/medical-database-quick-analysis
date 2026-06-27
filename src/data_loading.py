import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import PurePath
from typing import Any

import pandas as pd


UNSUPPORTED_FILE_ERROR = "Unsupported file type. Please upload a CSV, TSV, TXT, or XLSX file."
PARSE_FILE_ERROR = (
    "Could not read the uploaded file. Please check that it is a valid "
    "CSV, TSV, TXT, or Excel file."
)
INVALID_TABLE_ERROR = "Uploaded file does not contain a valid table."
TEXT_EXTENSIONS = {".csv", ".tsv", ".txt"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".xlsx"}
DELIMITER_CANDIDATES = ",\t;|"
SNIFF_SAMPLE_SIZE = 64 * 1024


@dataclass(frozen=True)
class DatasetMetadata:
    file_name: str
    file_size_bytes: int
    format_name: str
    delimiter: str | None
    encoding: str | None


@dataclass(frozen=True)
class DatasetLoadResult:
    dataframe: pd.DataFrame
    metadata: DatasetMetadata


def read_dataset(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded supported file into a DataFrame."""
    return read_dataset_with_metadata(uploaded_file).dataframe


def read_dataset_with_metadata(uploaded_file: Any) -> DatasetLoadResult:
    """Read a dataset and report the detected file characteristics."""
    file_name = str(getattr(uploaded_file, "name", ""))
    extension = PurePath(file_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(UNSUPPORTED_FILE_ERROR)

    content = _uploaded_file_bytes(uploaded_file)

    try:
        if extension == ".xlsx":
            dataframe = pd.read_excel(BytesIO(content))
            delimiter = None
            encoding = None
        else:
            encoding = _detect_text_encoding(content)
            text = content.decode(encoding)
            delimiter = _detect_delimiter(text, extension)
            parse_delimiter = delimiter if delimiter is not None else ","
            dataframe = pd.read_csv(StringIO(text), sep=parse_delimiter)
    except Exception as exc:
        raise ValueError(PARSE_FILE_ERROR) from exc

    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError(INVALID_TABLE_ERROR)

    return DatasetLoadResult(
        dataframe=dataframe,
        metadata=DatasetMetadata(
            file_name=file_name,
            file_size_bytes=len(content),
            format_name=_format_name(extension),
            delimiter=delimiter,
            encoding=encoding,
        ),
    )


def _uploaded_file_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        content = uploaded_file.getvalue()
    else:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        content = uploaded_file.read()

    if isinstance(content, str):
        return content.encode("utf-8")
    if isinstance(content, memoryview):
        return content.tobytes()
    if isinstance(content, bytes):
        return content
    return bytes(content)


def _detect_text_encoding(content: bytes) -> str:
    if content.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"

    if _looks_like_utf16(content, little_endian=True):
        return "utf-16-le"
    if _looks_like_utf16(content, little_endian=False):
        return "utf-16-be"

    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content.decode("cp1252")
        except UnicodeDecodeError:
            return "latin-1"
        return "cp1252"
    return "utf-8"


def _looks_like_utf16(content: bytes, *, little_endian: bool) -> bool:
    sample = content[:4096]
    if len(sample) < 4:
        return False

    null_positions = sample[1::2] if little_endian else sample[0::2]
    other_positions = sample[0::2] if little_endian else sample[1::2]
    null_ratio = null_positions.count(0) / len(null_positions)
    other_null_ratio = other_positions.count(0) / len(other_positions)
    return null_ratio >= 0.3 and other_null_ratio < 0.1


def _detect_delimiter(text: str, extension: str) -> str | None:
    if extension == ".tsv":
        return "\t"

    sample = text[:SNIFF_SAMPLE_SIZE]
    try:
        return csv.Sniffer().sniff(
            sample,
            delimiters=DELIMITER_CANDIDATES,
        ).delimiter
    except csv.Error:
        return _default_delimiter(extension)


def _default_delimiter(extension: str) -> str | None:
    if extension == ".csv":
        return ","
    if extension == ".tsv":
        return "\t"
    return None


def _format_name(extension: str) -> str:
    return {
        ".csv": "CSV",
        ".tsv": "TSV",
        ".txt": "TXT",
        ".xlsx": "Excel",
    }[extension]
