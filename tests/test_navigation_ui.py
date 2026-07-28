from streamlit.testing.v1 import AppTest


SIMPLE_CSV = b"patient_id,time,status,age\nA,10,1,50\nB,20,0,60\n"


def _navigation(app_test: AppTest):
    return next(item for item in app_test.radio if item.label == "Go to")


def test_initial_screen_shows_one_clear_dataset_task():
    app_test = AppTest.from_file("app.py").run(timeout=30)

    assert not app_test.exception
    assert [header.value for header in app_test.header] == ["Dataset"]
    assert [item.label for item in app_test.file_uploader] == ["Upload a dataset"]
    assert _navigation(app_test).options == [
        "Dataset",
        "Setup",
        "Data Quality",
        "Cohort Overview",
        "Charts",
        "Survival Analysis",
        "Export",
    ]
    assert not any(item.value == "Data Quality" for item in app_test.header)


def test_exploration_goal_skips_survival_setup_but_keeps_annotations_available():
    app_test = AppTest.from_file("app.py").run(timeout=30)
    app_test = app_test.file_uploader[0].upload(
        "clinical.csv",
        SIMPLE_CSV,
        "text/csv",
    ).run(timeout=30)

    goal = next(
        item for item in app_test.radio if item.label == "What would you like to do?"
    )
    assert goal.value == "Explore and review the dataset"
    assert any(
        button.label == "Continue to Data Quality"
        for button in app_test.button
    )
    assert not any(
        button.label == "Confirm survival mapping"
        for button in app_test.button
    )

    app_test = _navigation(app_test).set_value("Setup").run(timeout=30)

    assert not app_test.exception
    assert any(
        "Survival setup is skipped" in message.value
        for message in app_test.info
    )
    assert any(
        button.label == "Save annotations"
        for button in app_test.button
    )
    assert "survival_config" not in app_test.session_state


def test_bundled_example_loads_a_survival_ready_walkthrough():
    app_test = AppTest.from_file("app.py").run(timeout=30)
    load_example = next(
        button for button in app_test.button if button.label == "Load example dataset"
    )

    app_test = load_example.click().run(timeout=30)

    assert not app_test.exception
    assert app_test.session_state["dataset_metadata"].file_name == "lung.csv"
    assert len(app_test.session_state["uploaded_df"]) == 228
    assert app_test.session_state["analysis_goal"] == "Run survival analysis"
    assert app_test.session_state["time_unit_time"] == "days"
    assert any(
        button.label == "Continue to Survival Setup"
        for button in app_test.button
    )

    continue_button = next(
        button
        for button in app_test.button
        if button.label == "Continue to Survival Setup"
    )
    app_test = continue_button.click().run(timeout=30)
    time_unit = next(
        item for item in app_test.selectbox if item.label == "Time unit"
    )
    assert time_unit.value == "days"
