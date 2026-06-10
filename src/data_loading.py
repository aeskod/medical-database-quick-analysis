from pathlib import PurePath
from typing import Any

import pandas as pd


UNSUPPORTED_FILE_ERROR = "Unsupported file type. Please upload a CSV, TSV, or XLSX file."
PARSE_FILE_ERROR = (
    "Could not read the uploaded file. Please check that it is a valid CSV, TSV, or Excel file."
)
INVALID_TABLE_ERROR = "Uploaded file does not contain a valid table."


def read_dataset(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded CSV, TSV, or XLSX file into a DataFrame."""
    file_name = getattr(uploaded_file, "name", "")
    extension = PurePath(file_name).suffix.lower()

    if extension not in {".csv", ".tsv", ".xlsx"}:
        raise ValueError(UNSUPPORTED_FILE_ERROR)

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    try:
        if extension == ".csv":
            df = pd.read_csv(uploaded_file)
        elif extension == ".tsv":
            df = pd.read_csv(uploaded_file, sep="\t")
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as exc:
        raise ValueError(PARSE_FILE_ERROR) from exc

    if df.empty or len(df.columns) == 0:
        raise ValueError(INVALID_TABLE_ERROR)

    return df
