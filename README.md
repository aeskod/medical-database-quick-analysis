# Medical Dataset Explorer

Streamlit MVP for uploading a tabular medical dataset, profiling its columns, and confirming the required survival-analysis mapping.

## Features

- Upload CSV, TSV, or XLSX files.
- Show dataset row and column counts.
- Preview the first 20 rows.
- Generate a column profile with inferred types, missingness, parse rates, uniqueness, and example values.
- Suggest candidate survival time, event/status, patient ID, and grouping columns.
- Let the user confirm event/censor value mapping and time units.
- Validate the survival mapping and store a survival-ready dataframe in Streamlit session state.
- Display Kaplan-Meier survival analysis after mapping is confirmed.
- Show overall and optionally grouped KM curves with Plotly.
- Show survival summary metrics and survival probabilities at selected time points.
- Run data-quality checks for missingness, duplicates, survival validity, group quality, and possible identifier columns.

This version intentionally does not include Cox models, log-rank tests, number-at-risk tables, cohort filtering, automatic cleaning, advanced modeling, storage, authentication, or export.

## Setup

Use the project virtual environment:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest
```
