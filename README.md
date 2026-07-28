# Medical Dataset Explorer

Streamlit dashboard for reviewing patient-level tabular datasets, checking data quality,
describing a cohort, creating exploratory charts, and running basic survival analysis.

The app works best when one row represents one patient, subject, admission, or case. It does
not automatically clean data or replace a statistical analysis plan.

## Quick start

```bash
source venv/bin/activate
streamlit run app.py
```

Open the local URL printed by Streamlit, then follow the navigation in the left sidebar.

For a first walkthrough, open **Dataset**, expand **Try an example dataset**, and load the
recommended lung cohort. Its default survival mapping is:

- follow-up time: `time`
- event status: `status`
- event occurred: `1`
- censored: `0`
- time unit: days
- optional grouping variable: `sex`

Always verify event coding against the source data dictionary; other datasets may use the
opposite convention.

## Workflow

1. **Dataset** — upload CSV, TSV, TXT, or XLSX, verify the preview, and choose whether the goal
   is general exploration or survival analysis.
2. **Setup** — optionally confirm survival time/event roles and edit advanced column meanings.
3. **Data Quality** — review blocking issues and warnings before interpreting results.
4. **Cohort Overview** — review cohort cards and optionally build a Table 1-style baseline table.
5. **Charts** — select variables and let Auto mode recommend a compatible chart.
6. **Survival Analysis** — apply selected filters, view Kaplan–Meier curves, group comparisons,
   log-rank tests, number-at-risk tables, and survival probabilities.
7. **Export** — download the current or cleaned dataset, reusable JSON configuration, and
   combined HTML/PDF reports.

Survival mapping is optional for Data Quality, Cohort Overview, Charts, and downloading the
current dataset.

## Main features

- CSV, TSV, TXT, and XLSX loading with delimiter and encoding detection.
- Dataset preview with detected types and missing-value highlighting.
- Column profiling, semantic annotations, and suggested analysis uses.
- Data-quality checks for missingness, duplicates, identifiers, clinical values, dates,
  grouping quality, and survival validity.
- Cohort summary cards, key characteristics, and grouped baseline tables.
- Histograms, bar charts, box plots, scatter plots, stacked bars, correlation heatmaps, and
  missingness charts.
- Kaplan–Meier curves, confidence intervals, censor marks, cohort filters, overall/group
  summaries, log-rank and pairwise log-rank tests, number at risk, and selected-time survival
  probabilities.
- CSV, JSON, HTML, PDF, PNG, and SVG-oriented export paths.

This version does not include Cox model fitting, automatic data cleaning, authentication,
persistent project storage, or causal/adjusted inference. The annotation model retains a Cox
use flag internally for forward compatibility, but the current UI intentionally hides it.

## Test

```bash
venv/bin/python -m pytest
```
