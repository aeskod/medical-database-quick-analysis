# Project Handoff: Medical Dataset Explorer

Last updated: 2026-06-15

This file is intended as context for a new Codex chat working in the same project folder:

```text
/Users/artemskorobogatov/Documents/MyCode/meddb_dashboard
```

## Current Repo State

This is a Streamlit app for exploratory medical dataset review and basic survival analysis.

The project is already a Git repo. The last committed checkpoint is:

```text
6c189c4 step 05 implementation
```

After that commit, Step 6 work is currently uncommitted:

```text
M  app.py
?? md/06.md
?? md/07.md
?? src/charts.py
?? tests/test_charts.py
```

Important: before making future edits, run:

```bash
git status --short
```

Do not discard or reset these uncommitted changes unless the user explicitly asks.

## How To Run

Use the virtual environment that exists in the project directory:

```bash
cd /Users/artemskorobogatov/Documents/MyCode/meddb_dashboard
source venv/bin/activate
streamlit run app.py
```

The app usually opens at:

```text
http://localhost:8501
```

Do not run `python app.py` directly for normal use. Streamlit should run the app.

## How To Test

Run:

```bash
venv/bin/python -m pytest
```

Current full test suite result after Step 6:

```text
65 passed
```

Also useful:

```bash
venv/bin/python -m compileall app.py src tests
git diff --check
```

## Current Implemented Features

### Upload Tab

Implemented:

- Upload CSV, TSV, and XLSX files.
- Preview first 20 rows.
- Show row and column counts.
- Profile columns with:
  - detected type
  - missing count / percent
  - non-missing count
  - unique count / ratio
  - example values
  - numeric/date parse rates
  - min/max for numeric-like columns
  - binary, low-cardinality, non-negative, and ID-like flags
- Suggest survival roles:
  - time column
  - event/status column
  - patient ID column
  - grouping column
- Let user confirm survival mapping:
  - time column
  - event column
  - event values
  - censor values
  - patient ID
  - group column
  - time unit
  - missing event handling
- Stores original uploaded dataframe and profile in Streamlit session state.
- Stores confirmed `SurvivalConfig` and `survival_ready_df`.

### Data Quality Tab

Implemented:

- Works on original uploaded dataframe.
- Handles no-upload state gracefully.
- Dataset overview and status cards.
- Missingness by column and by row.
- Duplicate row checks.
- Duplicate patient ID checks when an ID column is mapped.
- Survival-specific quality checks when survival mapping exists:
  - usable / excluded rows
  - event and censor counts
  - missing time/event values
  - negative and zero time values
  - unmapped event values
- Group quality checks when grouping exists.
- Sensitive / identifier-like column candidate detection.
- Stores `data_quality_report` in session state.

### Cohort Overview Tab

Implemented:

- Works on original uploaded dataframe, not only `survival_ready_df`.
- Handles no-upload state gracefully.
- Summary cards:
  - rows
  - columns
  - complete rows
  - missing cells %
  - usable survival rows
  - events
  - censored
  - event rate
  - median follow-up
- Shows compact survival mapping summary if mapping exists.
- Shows message if survival mapping is not confirmed.
- Classifies variables for baseline summaries.
- Builds Table 1-style baseline table:
  - continuous variables summarized as mean +/- SD and median [Q1, Q3]
  - categorical variables summarized as count (%)
  - optional grouped columns
  - rare categorical levels collapsed into `Other`
  - optional missing rows
- Supports group value labels via optional session state:

```python
st.session_state["group_value_labels"] = {
    "sex": {
        "1": "Male",
        "2": "Female",
    }
}
```

- CSV download for baseline table.
- Changing the Cohort Overview group selector does not mutate `survival_config`.

### Charts Tab

Implemented in the uncommitted Step 6 changes:

- New module: `src/charts.py`.
- New tests: `tests/test_charts.py`.
- Chart type selector:
  - Auto
  - Histogram
  - Bar chart
  - Box plot
  - Scatter plot
  - Stacked bar chart
  - Correlation heatmap
  - Missingness bar chart
- Variable type detection for charts:
  - numeric
  - categorical
  - datetime
  - text
  - id
  - unknown
- Auto chart recommendations:
  - one numeric variable -> histogram
  - one categorical variable -> bar chart
  - numeric + categorical -> box plot
  - numeric + numeric -> scatter plot
  - categorical + categorical -> stacked bar chart
- Plotly chart builders:
  - histogram
  - categorical bar chart
  - box plot
  - scatter plot
  - stacked bar chart
  - correlation heatmap
  - missingness bar chart
- Chart HTML download.
- Optional color/group variable where applicable.
- Max categorical level slider and missing-category checkbox.
- Stacked bar percent normalization checkbox.

Important UI behavior from user feedback:

- For `Missingness bar chart` and `Correlation heatmap`, X/Y/color selectors do not apply.
- These selectors are now disabled and display `N/A` instead of stale selected variable names.
- This avoids making irrelevant controls look locked to a real variable.

### Survival Analysis Tab

Implemented:

- Requires confirmed survival mapping.
- Shows current mapping summary.
- Validates `survival_ready_df`.
- Overall Kaplan-Meier fit.
- Optional grouped Kaplan-Meier fit when `_group` exists.
- Confidence interval toggle.
- Toggle to use grouping column if available.
- Plotly Kaplan-Meier curve.
- Survival summary cards:
  - usable rows
  - events
  - censored
  - event rate
  - median follow-up
  - max follow-up
  - median survival
- Suggested timepoints.
- Overall survival probability table at selected timepoints.

## Current Module Map

```text
app.py
    Streamlit UI and session state orchestration.

src/data_loading.py
    CSV, TSV, XLSX loading.

src/profiling.py
    Missing normalization and column profiling.

src/role_suggestions.py
    Survival role candidate detection.

src/survival_mapping.py
    SurvivalConfig, mapping validation, survival-ready dataframe creation.

src/survival_analysis.py
    Survival dataframe validation, KM fitting, survival summaries, survival probabilities.

src/survival_plots.py
    Plotly KM curve.

src/data_quality.py
    Data quality checks and report builder.

src/cohort_overview.py
    Cohort metrics, baseline-variable classification, Table 1 summaries.

src/charts.py
    Step 6 chart typing, recommendation, prep, and Plotly exploratory charts.
```

## Remaining Plan Based On `tech_spec.md`

### Step 7: Survival Analysis Improvements

`md/07.md` already exists and describes the next planned step. It is not implemented yet.

Expected additions:

- Group-wise survival summary table.
- Overall log-rank test.
- Pairwise log-rank tests for 3+ groups.
- Number-at-risk table.
- Group-specific survival probability table.
- Better survival interpretation warnings:
  - too few rows
  - too few events
  - too many groups
  - groups with no censored observations
  - curve crossing / overlap warning
- Optional group value label editor.
- CSV downloads for survival tables.
- Do not add Cox regression in Step 7.

### Not Yet Implemented From The Technical Spec

Upload / mapping:

- Optional full column annotation table.
- Separate semantic meaning vs analysis role management.
- Start/stop time format for advanced survival data.
- Date-derived survival time.
- Encoding / delimiter detection beyond current file handling.
- Sticky headers and missing-value highlighting in preview.
- `Fix` buttons that scroll to a specific issue.

Data quality:

- Missingness heatmap.
- Date consistency checks.
- Age range checks.
- Event date before diagnosis date checks.
- Last follow-up before start date checks.
- More complete PHI/privacy audit.

Cohort overview:

- P-values and statistical comparison tests.
- Advanced filters.
- Publication-ready table formatting.
- PDF/HTML report export.

Charts:

- Summary statistics beside charts.
- Faceting / multi-panel charts.
- PNG/SVG export.
- Chart annotation editing.
- Advanced statistical overlays.

Survival analysis:

- Step 7 enhancements listed above.
- Cohort filters.
- Censor marks toggle on KM curves.
- Number-at-risk table.
- Log-rank p-values.
- Group-wise median survival.
- Group-specific survival probabilities.
- Cox regression / hazard ratios.
- Forest plots.
- Start-stop / time-varying survival models.

Exports / app infrastructure:

- Full HTML/PDF reports.
- Saved projects.
- Authentication.
- Deployment setup.
- Docker.
- Streamlit Cloud / Render deployment configuration.

## Important Do's

- Use the existing virtual environment in `meddb_dashboard`.
- Prefer the repo's current patterns and helper modules.
- Keep original uploaded data unchanged. Use mapping/config/session state layers.
- Use `uploaded_df` for Cohort Overview and Charts because it contains all clinical variables.
- Use `survival_ready_df` only for survival-specific metrics and survival analysis.
- Keep survival mapping in `survival_config`; do not mutate it from Cohort Overview or Charts.
- Show clear no-upload messages in every tab.
- Keep controls honest:
  - disable controls that do not apply
  - show `N/A` or equivalent instead of stale values when a setting is irrelevant
- Use plain-language user messages.
- Use Plotly for charts.
- Add tests for new backend logic.
- Run `venv/bin/python -m pytest` after changes.
- For UI changes, do at least a browser or Streamlit smoke check when possible.
- Preserve group label flexibility with `st.session_state["group_value_labels"]`.
- Use raw group values if the user has not provided labels.

## Important Don'ts

- Do not hardcode dataset-specific labels such as `1 = Male` or `2 = Female`.
- Do not assume missing event values are censored unless the user explicitly configures that.
- Do not force users to rename columns or fit a fixed template.
- Do not require every column to be annotated before the app works.
- Do not use `survival_ready_df` for general charts or cohort baseline variables.
- Do not make inactive controls look like locked real selections.
- Do not add heavy dependencies like seaborn or scikit-survival for the MVP.
- Do not add Cox regression before the planned Kaplan-Meier/log-rank improvements.
- Do not rewrite or reset Git history without explicit user approval.
- Do not revert uncommitted user/Codex changes unless explicitly asked.

## User Feedback Captured So Far

- The user wants the app to remain practical and understandable, not overly abstract.
- The user asked whether Streamlit supports little info buttons/popups. Streamlit supports help tooltips on many widgets and popover/expander-style UI.
- The user noticed stale selected variables in disabled chart controls were confusing.
  - Result: for Missingness and Correlation Heatmap, X/Y/color now show `N/A` while disabled.
- The user changed GitHub username to `aeskod`, but local Git config may still show old author name unless updated.
  - This does not affect the app.
  - Existing commits do not need rewriting unless the user specifically asks.

## Useful Commands

Run app:

```bash
cd /Users/artemskorobogatov/Documents/MyCode/meddb_dashboard
source venv/bin/activate
streamlit run app.py
```

Run tests:

```bash
venv/bin/python -m pytest
```

Check current Git state:

```bash
git status --short
git log --oneline -5
```

Commit when the user asks:

```bash
git add -A
git commit -m "message"
```

If Git staging/commit fails with `.git/index.lock: Operation not permitted`, the Codex sandbox likely needs escalated permission for Git commands that write to `.git`.

## Recommended Next Chat Startup Checklist

1. Run `git status --short`.
2. Read this file.
3. Read `md/tech_spec.md` only if broader roadmap context is needed.
4. Read `md/07.md` before implementing Step 7.
5. Confirm whether the user wants to commit the current uncommitted Step 6 work before starting larger changes.
6. Use the virtual environment for all tests and Streamlit runs.
