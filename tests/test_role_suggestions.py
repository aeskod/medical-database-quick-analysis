import pandas as pd

from src.profiling import profile_dataframe
from src.role_suggestions import (
    confidence_label,
    score_event_candidate,
    score_group_candidate,
    score_id_candidate,
    score_time_candidate,
    suggest_survival_roles,
)


def test_confidence_label():
    assert confidence_label(80) == "High"
    assert confidence_label(50) == "Medium"
    assert confidence_label(25) == "Low"
    assert confidence_label(24) == "Unlikely"


def test_candidate_scoring_prefers_expected_survival_columns():
    df = pd.DataFrame(
        {
            "follow_up_days": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            "status": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "patient_id": [f"P-{index:03d}" for index in range(12)],
            "sex": ["M", "F"] * 6,
            "notes": [f"Free text note {index}" for index in range(12)],
        }
    )
    profile = profile_dataframe(df).set_index("column_name")

    time_score = score_time_candidate("follow_up_days", profile.loc["follow_up_days"].to_dict())
    event_score = score_event_candidate("status", profile.loc["status"].to_dict())
    id_score = score_id_candidate("patient_id", profile.loc["patient_id"].to_dict())
    group_score = score_group_candidate("sex", profile.loc["sex"].to_dict())

    assert time_score["score"] >= 80
    assert "Column is numeric" in time_score["reasons"]
    assert event_score["score"] >= 80
    assert "Column is binary-like" in event_score["reasons"]
    assert id_score["score"] >= 80
    assert "Column values are mostly unique" in id_score["reasons"]
    assert group_score["score"] >= 80
    assert "Column has 2 unique values" in group_score["reasons"]


def test_suggest_survival_roles_returns_ranked_candidates():
    df = pd.DataFrame(
        {
            "follow_up_days": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            "status": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "patient_id": [f"P-{index:03d}" for index in range(12)],
            "sex": ["M", "F"] * 6,
        }
    )

    suggestions = suggest_survival_roles(profile_dataframe(df), top_n=3)

    assert suggestions["time_candidates"][0]["column_name"] == "follow_up_days"
    assert suggestions["event_candidates"][0]["column_name"] == "status"
    assert suggestions["id_candidates"][0]["column_name"] == "patient_id"
    assert suggestions["group_candidates"][0]["column_name"] == "sex"
    assert all(candidate["score"] >= 25 for candidates in suggestions.values() for candidate in candidates)
