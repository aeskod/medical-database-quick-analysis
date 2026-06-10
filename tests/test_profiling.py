import pandas as pd

from src.profiling import infer_column_type, normalize_missing_values, profile_dataframe


def test_normalize_missing_values_returns_copy_and_handles_tokens():
    original = pd.DataFrame(
        {
            "a": ["", " ", "NA", "N/A", "na", "n/a"],
            "b": ["NULL", "null", "None", "none", "missing", "Missing"],
            "c": ["value", " NA ", 10, 0, False, "kept"],
        }
    )

    normalized = normalize_missing_values(original)

    assert normalized is not original
    assert original.loc[0, "a"] == ""
    assert normalized["a"].isna().all()
    assert normalized["b"].isna().all()
    assert normalized.loc[0, "c"] == "value"
    assert pd.isna(normalized.loc[1, "c"])
    assert normalized.loc[2, "c"] == 10


def test_infer_column_type_empty():
    assert infer_column_type(pd.Series(["", "NA", None], name="notes")) == "empty"


def test_infer_column_type_boolean():
    assert infer_column_type(pd.Series(["yes", "No", "Y", "n"], name="event")) == "boolean"
    assert infer_column_type(pd.Series([0, 1, 1, 0], name="flag")) == "boolean"


def test_infer_column_type_integer_and_float():
    assert infer_column_type(pd.Series(["1", "2", "3"], name="age")) == "integer"
    assert infer_column_type(pd.Series(["1", "2.5", "3"], name="score")) == "float"


def test_infer_column_type_date():
    series = pd.Series(
        ["2024-01-01", "2024-02-15", "03/20/2024", "not a date", "2024-05-01"],
        name="diagnosis_date",
    )

    assert infer_column_type(series) == "date"


def test_infer_column_type_id_like_with_name_hint():
    series = pd.Series([f"P-{index:03d}" for index in range(20)], name="patient_id")

    assert infer_column_type(series) == "id_like"


def test_infer_column_type_categorical_text_and_mixed():
    assert infer_column_type(pd.Series(["Male", "Female", "Female"], name="sex")) == "categorical"

    long_text = pd.Series(
        [
            "Patient reports sustained fatigue and shortness of breath after activity.",
            "Follow-up note documents improved tolerance after medication adjustment.",
            "Discharge summary includes care plan and monitoring recommendations.",
        ],
        name="clinical_note",
    )
    assert infer_column_type(long_text) == "text"

    mixed = pd.Series([10, 12, "unknown", "high", "low"], name="measurement")
    assert infer_column_type(mixed) == "mixed"


def test_profile_dataframe_returns_required_columns_and_values():
    df = pd.DataFrame(
        {
            "age": [55, 63, "NA", 80],
            "status": ["Dead", "Alive", "Dead", ""],
        }
    )

    profile = profile_dataframe(df)

    assert list(profile.columns) == [
        "column_name",
        "detected_type",
        "missing_count",
        "missing_percent",
        "non_missing_count",
        "unique_count",
        "example_values",
    ]

    age_profile = profile.loc[profile["column_name"] == "age"].iloc[0]
    status_profile = profile.loc[profile["column_name"] == "status"].iloc[0]

    assert age_profile["detected_type"] == "integer"
    assert age_profile["missing_count"] == 1
    assert age_profile["missing_percent"] == 25.0
    assert age_profile["non_missing_count"] == 3
    assert age_profile["unique_count"] == 3
    assert age_profile["example_values"] == "55, 63, 80"

    assert status_profile["detected_type"] == "categorical"
    assert status_profile["missing_count"] == 1
    assert status_profile["example_values"] == "Dead, Alive"
