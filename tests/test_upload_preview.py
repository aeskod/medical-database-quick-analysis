import pandas as pd

from app import (
    _format_delimiter,
    _format_file_size,
    _prepare_dataset_preview,
)
from src.profiling import profile_dataframe


def test_preview_adds_detected_type_labels_and_missing_mask_without_mutating_data():
    df = pd.DataFrame(
        {
            "age": [55, "NA", 72],
            "status": ["Dead", "Alive", None],
        }
    )
    original = df.copy(deep=True)
    profile = profile_dataframe(df)

    preview, missing_mask = _prepare_dataset_preview(df, profile, row_limit=2)

    assert preview.columns.tolist() == ["age · integer", "status · binary"]
    assert preview.iloc[1, 0] == "NA"
    assert missing_mask.iloc[1, 0]
    assert not missing_mask.iloc[0, 0]
    pd.testing.assert_frame_equal(df, original)


def test_preview_row_limit_and_metadata_formatters():
    df = pd.DataFrame({"value": range(50)})
    profile = profile_dataframe(df)

    preview, missing_mask = _prepare_dataset_preview(df, profile, row_limit=35)

    assert len(preview) == 35
    assert missing_mask.shape == preview.shape
    assert _format_file_size(999) == "999 B"
    assert _format_file_size(1536) == "1.5 KB"
    assert _format_file_size(2 * 1024 * 1024) == "2.0 MB"
    assert _format_delimiter(",") == "Comma (,)"
    assert _format_delimiter("\t") == "Tab"
    assert _format_delimiter(None) == "Not detected (single column)"
