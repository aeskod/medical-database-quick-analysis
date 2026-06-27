import pandas as pd
import pytest

import app
from src.column_annotations import (
    ColumnAnnotation,
    USE_BASELINE,
    USE_CHARTS,
    USE_COX,
    USE_FILTER,
    USE_GROUP,
    USE_IGNORE,
    annotations_from_dataframe,
    annotations_to_dataframe,
    build_default_annotations,
    get_columns_for_use,
    sync_annotations,
)
from src.profiling import profile_dataframe
from src.survival_mapping import SurvivalConfig, create_survival_ready_dataframe


@pytest.fixture
def clinical_df():
    return pd.DataFrame(
        {
            "patient_id": [f"P{index:02d}" for index in range(12)],
            "time_days": list(range(10, 130, 10)),
            "status": [1, 0] * 6,
            "age": list(range(50, 62)),
            "sex": ["F", "M"] * 6,
            "treatment": ["A", "B", "C"] * 4,
            "notes": [f"Long clinical note {index}" for index in range(12)],
        }
    )


def test_default_annotations_infer_meanings_and_independent_uses(clinical_df):
    annotations = build_default_annotations(clinical_df, profile_dataframe(clinical_df))

    assert annotations["patient_id"].meaning == "Patient ID"
    assert annotations["patient_id"].uses == frozenset({USE_IGNORE})
    assert annotations["age"].meaning == "Age"
    assert annotations["age"].uses == frozenset(
        {USE_FILTER, USE_BASELINE, USE_COX, USE_CHARTS}
    )
    assert annotations["sex"].meaning == "Sex / gender"
    assert USE_GROUP in annotations["sex"].uses
    assert annotations["notes"].meaning == "Notes / free text"


def test_survival_roles_override_required_meanings_and_seed_group_use(clinical_df):
    config = SurvivalConfig(
        time_col="time_days",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        id_col="patient_id",
        group_col="treatment",
    )

    annotations = build_default_annotations(
        clinical_df,
        profile_dataframe(clinical_df),
        config,
    )

    assert annotations["time_days"].meaning == "Follow-up time / survival time"
    assert annotations["status"].meaning == "Event status"
    assert annotations["patient_id"].meaning == "Patient ID"
    assert annotations["patient_id"].uses == frozenset({USE_IGNORE})
    assert USE_GROUP in annotations["treatment"].uses


def test_annotation_editor_round_trip_and_ignore_precedence(clinical_df):
    profile = profile_dataframe(clinical_df)
    annotations = build_default_annotations(clinical_df, profile)
    editor_df = annotations_to_dataframe(annotations, profile)
    age_row = editor_df["Column"] == "age"
    editor_df.loc[age_row, "Meaning"] = "Custom..."
    editor_df.loc[age_row, "Custom meaning"] = "Age at enrollment"
    editor_df.loc[age_row, "Ignore in analysis"] = True

    parsed = annotations_from_dataframe(editor_df, clinical_df.columns)

    assert parsed["age"].resolved_meaning == "Age at enrollment"
    assert parsed["age"].uses == frozenset({USE_IGNORE})


def test_annotation_editor_rejects_blank_custom_meaning(clinical_df):
    profile = profile_dataframe(clinical_df)
    editor_df = annotations_to_dataframe(
        build_default_annotations(clinical_df, profile),
        profile,
    )
    editor_df.loc[editor_df["Column"] == "age", "Meaning"] = "Custom..."

    with pytest.raises(ValueError, match="Enter a custom meaning"):
        annotations_from_dataframe(editor_df, clinical_df.columns)


def test_sync_preserves_user_choices_but_drops_stale_columns(clinical_df):
    profile = profile_dataframe(clinical_df)
    existing = build_default_annotations(clinical_df, profile)
    existing["age"] = ColumnAnnotation(
        "age",
        "Lab value / numeric measurement",
        frozenset({USE_CHARTS}),
    )
    existing["removed"] = ColumnAnnotation("removed", "Age", frozenset({USE_BASELINE}))

    reduced_df = clinical_df.drop(columns=["notes"])
    synced = sync_annotations(existing, reduced_df, profile_dataframe(reduced_df))

    assert synced["age"] == existing["age"]
    assert "removed" not in synced
    assert "notes" not in synced


def test_sync_protects_survival_meanings_without_overwriting_user_analysis_uses(
    clinical_df,
):
    config = SurvivalConfig(
        time_col="time_days",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        group_col="sex",
    )
    existing = build_default_annotations(
        clinical_df,
        profile_dataframe(clinical_df),
        config,
    )
    existing["time_days"] = ColumnAnnotation(
        "time_days",
        "Age",
        frozenset({USE_IGNORE}),
    )
    existing["sex"] = ColumnAnnotation(
        "sex",
        "Sex / gender",
        frozenset({USE_CHARTS}),
    )

    synced = sync_annotations(
        existing,
        clinical_df,
        profile_dataframe(clinical_df),
        config,
    )

    assert synced["time_days"].meaning == "Follow-up time / survival time"
    assert synced["time_days"].uses == frozenset({USE_IGNORE})
    assert synced["sex"].uses == frozenset({USE_CHARTS})


def test_get_columns_for_use_excludes_ignored_and_unknown_columns(clinical_df):
    annotations = {
        "age": ColumnAnnotation("age", "Age", frozenset({USE_BASELINE})),
        "sex": ColumnAnnotation("sex", "Sex / gender", frozenset({USE_GROUP, USE_IGNORE})),
        "stale": ColumnAnnotation("stale", "Age", frozenset({USE_BASELINE})),
    }

    assert get_columns_for_use(
        annotations,
        USE_BASELINE,
        clinical_df.columns,
    ) == ["age"]
    assert get_columns_for_use(annotations, USE_GROUP, clinical_df.columns) == []
    assert get_columns_for_use(annotations, USE_IGNORE, clinical_df.columns) == ["sex"]


def test_uploaded_dataset_state_persists_annotations_until_dataset_changes(
    clinical_df,
    monkeypatch,
):
    state = {}
    monkeypatch.setattr(app.st, "session_state", state)
    app._sync_uploaded_dataset_state("clinical.csv", clinical_df)
    state["column_annotations"]["age"] = ColumnAnnotation(
        "age",
        "Age",
        frozenset({USE_CHARTS}),
    )

    app._sync_uploaded_dataset_state("clinical.csv", clinical_df.copy(deep=True))
    assert state["column_annotations"]["age"].uses == frozenset({USE_CHARTS})

    changed_df = clinical_df.rename(columns={"age": "age_years"})
    app._sync_uploaded_dataset_state("changed.csv", changed_df)
    assert "age" not in state["column_annotations"]
    assert "age_years" in state["column_annotations"]


def test_uploaded_dataset_state_detects_changed_values_with_same_name_and_shape(
    clinical_df,
    monkeypatch,
):
    state = {}
    monkeypatch.setattr(app.st, "session_state", state)
    app._sync_uploaded_dataset_state("clinical.csv", clinical_df)
    state["survival_config"] = object()
    state["survival_ready_df"] = object()

    changed_df = clinical_df.copy(deep=True)
    changed_df.loc[0, "time_days"] = changed_df.loc[0, "time_days"] + 1000
    app._sync_uploaded_dataset_state("clinical.csv", changed_df)

    assert "survival_config" not in state
    assert "survival_ready_df" not in state


def test_baseline_defaults_are_driven_by_annotations(clinical_df):
    annotations = {
        "age": ColumnAnnotation("age", "Age", frozenset({USE_BASELINE})),
        "sex": ColumnAnnotation("sex", "Sex / gender", frozenset({USE_CHARTS})),
        "treatment": ColumnAnnotation(
            "treatment",
            "Treatment / exposure group",
            frozenset({USE_BASELINE}),
        ),
    }
    inferred = {
        "continuous": ["age"],
        "categorical": ["sex", "treatment"],
        "excluded": [],
    }

    defaults = app._annotated_baseline_variables(clinical_df, inferred, annotations)

    assert defaults["continuous"] == ["age"]
    assert defaults["categorical"] == ["treatment"]
    assert "sex" in defaults["excluded"]


def test_survival_group_can_switch_between_annotated_columns_without_mutating_mapping(
    clinical_df,
):
    config = SurvivalConfig(
        time_col="time_days",
        event_col="status",
        event_values=[1],
        censor_values=[0],
        group_col="sex",
    )
    stored = create_survival_ready_dataframe(clinical_df, config)

    treatment_grouped = app._survival_dataframe_for_group(
        clinical_df,
        config,
        stored,
        "treatment",
    )
    ungrouped = app._survival_dataframe_for_group(
        clinical_df,
        config,
        stored,
        None,
    )

    assert treatment_grouped["_group"].tolist() == clinical_df["treatment"].tolist()
    assert "_group" not in ungrouped.columns
    assert config.group_col == "sex"
    assert stored["_group"].tolist() == clinical_df["sex"].tolist()
