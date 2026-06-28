import hashlib
import json
import pickle
from pathlib import PurePath
from typing import Any

import pandas as pd


def dataset_content_signature(
    file_name: str,
    df: pd.DataFrame,
    *,
    content_digest: str | None = None,
) -> str:
    """Return a stable identity for the dataset as parsed by the application.

    Uploaded bytes are the primary identity. The normalized extension is also
    included because it selects the parser and identical bytes can produce
    different tables when, for example, they are uploaded as CSV versus TSV.
    Dataframe-only callers use a content digest of the parsed table.
    """
    if content_digest is not None:
        parser_extension = PurePath(str(file_name)).suffix.casefold() or "<none>"
        return f"sha256:{content_digest};parser-extension:{parser_extension}"

    return f"dataframe-sha256:{dataframe_content_digest(df)}"


def uploaded_file_content_digest(uploaded_file: Any) -> str:
    """Return a SHA-256 digest of an uploaded file without changing its position."""
    if hasattr(uploaded_file, "getvalue"):
        content = uploaded_file.getvalue()
    else:
        original_position = uploaded_file.tell() if hasattr(uploaded_file, "tell") else None
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        content = uploaded_file.read()
        if original_position is not None and hasattr(uploaded_file, "seek"):
            uploaded_file.seek(original_position)

    if isinstance(content, str):
        content = content.encode("utf-8")
    elif isinstance(content, memoryview):
        content = content.tobytes()
    elif not isinstance(content, bytes):
        content = bytes(content)

    return hashlib.sha256(content).hexdigest()


def dataframe_content_digest(df: pd.DataFrame) -> str:
    """Return a content digest for callers that do not have the original file."""
    digest = hashlib.sha256()
    schema = [
        {
            "name_type": type(column).__qualname__,
            "name": repr(column),
            "dtype": str(dtype),
        }
        for column, dtype in zip(df.columns, df.dtypes, strict=True)
    ]
    digest.update(
        json.dumps(
            {"shape": df.shape, "schema": schema},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    try:
        row_hashes = pd.util.hash_pandas_object(
            df,
            index=True,
            categorize=False,
        )
        digest.update(row_hashes.to_numpy().tobytes())
    except (TypeError, ValueError):
        # Object columns can contain unhashable Python values. Pickling an
        # in-memory dataframe is safe here because this code never unpickles it.
        digest.update(pickle.dumps(df, protocol=5))

    return digest.hexdigest()
