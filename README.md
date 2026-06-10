# Medical Dataset Explorer

Streamlit MVP for uploading a tabular medical dataset and profiling its columns.

## Features

- Upload CSV, TSV, or XLSX files.
- Show dataset row and column counts.
- Preview the first 20 rows.
- Generate a column profile with inferred types, missingness, unique counts, and example values.

This first step intentionally does not include survival analysis, charts, cohort filtering, role mapping, modeling, storage, authentication, or export.

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
