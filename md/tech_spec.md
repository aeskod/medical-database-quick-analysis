# Technical Development Checklist: Exploratory Medical Dataset + Survival Analysis Dashboard

## 1. Project Goal

Build a web-based dashboard that allows users to upload a tabular patient/cohort dataset and quickly inspect:

* dataset structure
* data quality
* cohort composition
* variable distributions
* basic survival analysis, when survival-compatible columns are available

The dashboard should **not assume a fixed dataset template**. Instead, it should create a flexible **column-mapping / annotation layer** on top of whatever dataset the user uploads.

The intended dataset shape is:

```text
One row = one patient / subject / admission / case
Columns = patient features, outcomes, dates, treatment groups, follow-up information, etc.
```

The dashboard should work best for:

```text
tabular patient-level cohort datasets
CSV / Excel / TSV files
datasets with follow-up time and event status
datasets with demographic, clinical, treatment, diagnostic, lab, or biomarker variables
```

The dashboard is **not primarily designed for**:

```text
ClinicalTrials.gov metadata exports
imaging-only datasets
free-text clinical notes only
genomic sequence reads only
aggregated tables from papers
trial-level summary records
```

---

## 2. Recommended Tech Stack

### MVP Stack

```text
Framework:        Streamlit
Language:         Python
Data handling:    pandas
Charts:           Plotly
Survival:         lifelines
Validation:       Pandera or custom validation functions
File support:     CSV, XLSX, TSV
Export:           CSV, PNG/SVG, HTML report later
```

### Later Additions

```text
Large data:       Polars, DuckDB, PyArrow/Parquet
Advanced models:  scikit-survival
Reports:          Quarto or custom HTML/PDF export
Deployment:       Docker, Streamlit Cloud, Render, internal server
Production app:   FastAPI + React/Next.js only if needed later
```

### Reasoning

The first version should stay **Python-first** because most complexity is in:

```text
data cleaning
type detection
survival calculation
summary statistics
missingness handling
validation
```

A full React/FastAPI stack is not necessary for the MVP unless the app needs accounts, saved projects, multi-user access, or a highly polished production UI.

---

## 3. Main App Tabs

The app should have 5 main tabs:

```text
1. Upload
2. Data Quality
3. Cohort Overview
4. Charts
5. Survival Analysis
```

Each tab should answer a specific question.

| Tab               | Main question                                                 |
| ----------------- | ------------------------------------------------------------- |
| Upload            | What dataset am I working with, and what do the columns mean? |
| Data Quality      | Can I trust this dataset for analysis?                        |
| Cohort Overview   | Who/what is in the cohort?                                    |
| Charts            | What do the variables look like?                              |
| Survival Analysis | What happens over time, and do groups differ?                 |

---

# 4. Upload Page

## 4.1 Purpose

The upload page is the most important setup page. It should:

```text
1. Let the user upload a dataset
2. Preview the data
3. Detect column types
4. Let the user map required survival fields
5. Let the user optionally annotate all columns
6. Validate whether survival analysis is possible
7. Show clear errors and warnings
```

## 4.2 Recommended Upload Page Layout

```text
Upload Page
├── File upload area
├── Dataset summary cards
├── Dataset preview table
├── Required survival mapping form
├── Optional column annotation table
├── Validation summary
└── Continue button
```

---

## 4.3 File Upload Section

Supported files:

```text
.csv
.xlsx
.tsv
.txt with delimiter detection
```

Show:

```text
file name
file size
number of rows
number of columns
detected delimiter
detected encoding if relevant
```

Example:

```text
Dataset loaded: heart_failure.csv
Rows: 299
Columns: 13
Detected format: CSV
```

---

## 4.4 Dataset Preview

Show the first 10–30 rows of the dataset.

Requirements:

```text
horizontal scrolling
sticky column headers if possible
detected type icons or labels
missing value highlighting
optional “show more rows”
```

The preview is for the user to confirm they uploaded the correct dataset.

---

## 4.5 Detected Column Summary

Below or beside the preview, show a compact data dictionary.

Example:

| Column      | Detected type | Missing | Unique/example values |
| ----------- | ------------- | ------: | --------------------- |
| `age`       | numeric       |      5% | 42, 55, 61            |
| `sex`       | categorical   |      0% | M, F                  |
| `surv_days` | numeric       |      3% | 120, 450, 800         |
| `status`    | categorical   |      0% | Dead, Alive           |
| `treatment` | categorical   |      2% | A, B                  |

Detected types should include:

```text
numeric
integer
float
categorical
binary
date/datetime
text/free text
ID-like
boolean
mixed/unknown
```

---

# 5. Column Mapping Design

## 5.1 Important Design Principle

Do **not** force users to make their dataset match a fixed template.

Instead, use a metadata/config layer:

```text
Original dataset column: "surv_days"
Assigned internal role:  "survival_time"
```

The original file remains unchanged. The app only stores a mapping.

Example internal config:

```python
config = {
    "id_col": "patient_id",
    "time_col": "surv_days",
    "event_col": "status",
    "event_values": ["Dead"],
    "censor_values": ["Alive"],
    "time_unit": "days",
    "group_col": "treatment",
    "baseline_vars": ["age", "sex", "stage", "treatment"],
    "filter_vars": ["age", "sex", "stage", "treatment"],
    "cox_covariates": ["age", "stage", "treatment"]
}
```

The analysis code should only use the config, not hardcoded column names.

---

## 5.2 Split Mapping into Two Parts

Do not rely only on dropdowns inside each table column header.

Use two separate mapping sections:

```text
A. Required Survival Mapping
B. Optional Column Annotation
```

This is clearer and more scalable.

---

# 6. Required Survival Mapping

This should be a small form above the dataset preview or beside it.

## 6.1 Required/Recommended Fields

```text
Patient ID:          [select column]
Follow-up time:      [select column]
Event status:        [select column]
Event value(s):      [select value(s)]
Censored value(s):   [select value(s), or auto = all non-event values]
Time unit:           [days / months / years / unknown]
```

## 6.2 Which Fields Are Actually Required?

For survival analysis:

| Field                          |            Required? | Notes                                             |
| ------------------------------ | -------------------: | ------------------------------------------------- |
| Follow-up time / time-to-event |                  Yes | Must be numeric or derivable from dates           |
| Event status                   |                  Yes | Must be convertible into event/censored           |
| Event value                    |                  Yes | User must specify what means “event occurred”     |
| Censor value                   |          Usually yes | Can sometimes be inferred as all non-event values |
| Patient ID                     | Strongly recommended | Needed for duplicate checks and repeated rows     |
| Time unit                      |          Recommended | Needed for correct labels and interpretation      |

## 6.3 What “Time-to-Event / Follow-up Time” Means

This is the time from the start of observation until either:

```text
the event occurs
OR
the patient is last observed without the event
```

Examples:

| Patient | Status | Follow-up time |
| ------- | ------ | -------------: |
| A       | Dead   |       500 days |
| B       | Alive  |       300 days |
| C       | Dead   |       800 days |

For survival analysis:

```text
Patient A: event at 500 days
Patient B: censored at 300 days
Patient C: event at 800 days
```

The time column may be named:

```text
time
survival_time
follow_up_days
followup_months
OS_months
days_to_death
recurrence_free_survival
duration
```

---

## 6.4 What “Event Status” Means

The event status column tells the model whether the event actually happened.

Examples:

```text
1 / 0
Dead / Alive
Relapsed / No relapse
Progressed / Stable
Event / Censored
Yes / No
TRUE / FALSE
```

Survival analysis internally needs something like:

```text
1 = event occurred
0 = censored
```

So the app must support converting text or categorical values into binary event/censor coding.

---

## 6.5 Event/Censor Value Mapping

If the selected event column contains text values, show a clarification box.

Example:

```text
We detected text values in your event column.

Values found:
Dead, Alive

Which value(s) should be treated as the event?

[ ] Dead
[ ] Alive
```

If user selects `Dead`, the app converts:

```text
Dead  -> 1/event
Alive -> 0/censored
```

For multi-value status columns:

```text
Dead
Relapsed
Progressed
Alive
Lost to follow-up
Unknown
```

Allow multiple event values:

```text
Event values:
[x] Dead
[x] Relapsed
[x] Progressed

Censored values:
[x] Alive
[x] Lost to follow-up
[ ] Unknown
```

Do not automatically assume missing values are censored unless the user confirms.

---

# 7. Optional Start Time / End Time

## 7.1 Basic Survival Format

Most datasets use a simple format:

```text
time_col = follow-up duration from time zero
event_col = event/censor status
```

In this case:

```text
start time = implicitly 0
end time = time_col
```

Example:

| patient_id | follow_up_days | status |
| ---------- | -------------: | ------ |
| P001       |            300 | Alive  |
| P002       |            850 | Dead   |

This is enough for Kaplan–Meier and basic Cox models.

## 7.2 Start–Stop Format

Start time and end time are useful for advanced datasets where each patient may have multiple time intervals.

Example:

| patient_id | start | stop | event | treatment_status |
| ---------- | ----: | ---: | ----: | ---------------- |
| A          |     0 |   25 |     0 | before treatment |
| A          |    25 |   50 |     1 | after treatment  |
| B          |     0 |   40 |     0 | before treatment |
| B          |    40 |   60 |     0 | after treatment  |

This supports time-varying covariates and interval-style data.

## 7.3 MVP Decision

For the MVP:

```text
Support simple one-time-column survival analysis first.
Keep start/stop support optional or hidden under “Advanced survival format.”
```

---

# 8. Optional Column Annotation

After the required survival setup, provide a full column annotation table.

## 8.1 Purpose

The optional annotation table tells the app what other columns mean and how they should be used.

Example:

| Column      | Type        | Missing | Example values | Meaning       | Use as                         |
| ----------- | ----------- | ------: | -------------- | ------------- | ------------------------------ |
| `age`       | numeric     |      5% | 55, 63, 72     | Age           | filter, Table 1, Cox covariate |
| `sex`       | categorical |      0% | M, F           | Sex/gender    | filter, group, Table 1         |
| `stage`     | categorical |      3% | I, II, III     | Disease stage | group, Table 1                 |
| `treatment` | categorical |      2% | A, B           | Treatment     | group, filter                  |
| `notes`     | text        |     10% | free text      | Notes         | ignore                         |

## 8.2 Recommended Annotation Dropdown Items

Use these as the column “meaning” dropdown:

```text
Ignore column
Patient ID
Follow-up time / survival time
Event status
Start time
End time
Age
Sex / gender
Diagnosis
Treatment / exposure group
Disease stage
Risk category
Comorbidity / precondition
Medication
Lab value / numeric measurement
Biomarker
Genetic marker
Procedure / surgery
Outcome other than survival
Date
Site / hospital / center
Race / ethnicity
Smoking / lifestyle factor
Other categorical variable
Other numeric variable
Notes / free text
Custom...
```

## 8.3 “Use As” Options

Separately from meaning, allow the user to choose how the column should be used:

```text
Use as filter
Use as group/stratification variable
Use in baseline table
Use as Cox covariate
Use only in charts
Ignore in analysis
```

One column can have multiple uses.

Example:

```text
age:
Meaning = Age
Uses = filter, Table 1, Cox covariate, charts
```

---

# 9. Data Type vs Semantic Meaning vs Analysis Role

The app should internally separate three ideas.

## 9.1 Data Type

What the computer detects:

```text
numeric
categorical
binary
date
text
ID-like
mixed
```

## 9.2 Semantic Meaning

What the column means medically:

```text
age
diagnosis
treatment
stage
lab value
mutation
event status
follow-up time
```

## 9.3 Analysis Role

How the app uses the column:

```text
survival time
survival event
filter
grouping variable
baseline table variable
Cox covariate
ignore
```

Example:

```text
Column: age_at_dx
Type: numeric
Meaning: Age
Role: filter + baseline table + Cox covariate
```

Example:

```text
Column: vital_status
Type: categorical
Meaning: Event status
Role: survival event
Mapping: Dead = 1, Alive = 0
```

---

# 10. Validation Logic

## 10.1 Blocking Errors

Blocking errors prevent survival analysis from running.

Examples:

```text
No follow-up/time column selected.
No event status column selected.
Selected time column is not numeric and no date derivation is configured.
Event column has values that have not been mapped to event/censored.
Time column contains negative values.
All rows are missing survival time.
No usable rows remain after filtering.
```

## 10.2 Non-Blocking Warnings

Warnings do not block analysis but should be shown clearly.

Examples:

```text
Event column has 10% missing values.
Time column has 3% missing values.
One group has fewer than 10 patients.
One group has fewer than 5 events.
Median survival was not reached.
Survival curves cross; log-rank test may be less reliable.
Dataset has duplicate patient IDs.
Column marked as diagnosis is numeric; confirm this is intentional.
Column has high missingness.
```

## 10.3 Error Message Style

Avoid technical error text like:

```text
interpretation assigned to col3 expects 0 or 1 but received text
```

Use user-facing text:

```text
The selected event status column contains text values: Dead, Alive.
Please choose which value means the event occurred.
[Fix]
```

For numeric-coded categories:

```text
This column is numeric, but it was marked as Diagnosis.
If these numbers represent categories, you can keep this mapping.
[Change mapping] [Keep as diagnosis]
```

## 10.4 Error Summary Design

Keep an error/warning panel at the bottom or side:

```text
Errors
- Event status needs value mapping. [Fix]
- Follow-up time contains negative values. [Fix]

Warnings
- Event column has 10% missing values.
- Treatment group B has only 8 patients.
```

Clicking `Fix` should scroll to or open the relevant mapping UI.

---

# 11. Clarification Dialogs

Use clarification dialogs instead of technical errors whenever possible.

## 11.1 Text Event Column

```text
We detected text values in your event column.

Which value(s) should be treated as the event?

[ ] Dead
[ ] Alive
```

## 11.2 Missing Event Date

If the dataset has an event date but no event status column:

```text
You selected death_date as the event date column.

How should missing dates be interpreted?

[ ] Missing date means censored / event did not occur before last follow-up
[ ] Missing date means unknown / exclude from survival analysis
```

## 11.3 Multi-Category Status

```text
Values found:
Alive, Dead, Relapsed, Lost to follow-up, Unknown

Select event values:
[ ] Dead
[ ] Relapsed

Select censored values:
[ ] Alive
[ ] Lost to follow-up

Handle unknown:
[ ] Exclude
[ ] Treat as censored
```

Default should be conservative:

```text
Unknown/missing should not automatically be treated as censored unless user confirms.
```

---

# 12. Data Quality Tab

## 12.1 Purpose

Answer:

```text
Can this dataset be trusted for analysis?
```

## 12.2 Contents

```text
quality summary cards
missing data table
missingness heatmap
duplicate ID check
invalid value checks
date consistency checks
event coding check
follow-up time check
privacy/PHI warning
analysis-readiness status
```

## 12.3 Quality Cards

Example:

```text
Missing cells:          4.8%
Duplicate IDs:          0
Invalid ages:           3
Invalid event values:   0
Zero follow-up rows:    12
Analysis-ready rows:    1,205 / 1,240
```

## 12.4 Checks

```text
negative survival time
zero follow-up time
missing event status
missing follow-up time
age < 0 or age > 120
duplicate patient IDs
impossible dates
event date before diagnosis date
last follow-up before start date
columns with very high missingness
groups with very small sample size
```

## 12.5 Privacy/PHI Warning

Flag suspicious columns:

```text
name
email
phone
address
passport
national_id
medical_record_number
MRN
date_of_birth
exact address
free-text notes
```

Show warning:

```text
This dataset may contain direct identifiers. Make sure the data is de-identified before uploading or sharing.
```

---

# 13. Cohort Overview Tab

## 13.1 Purpose

Answer:

```text
Who is in the dataset?
```

## 13.2 Contents

```text
total patients
events
censored
event rate
median follow-up
age summary
sex/gender distribution
diagnosis distribution
treatment distribution
baseline characteristics table
outcome summary
download Table 1
```

## 13.3 Summary Cards

```text
Total patients:       1,240
Events:               312
Censored:             928
Event rate:           25.2%
Median follow-up:     36 months
Median age:           62 years
```

## 13.4 Baseline Characteristics Table

Example:

| Variable                |     Overall |     Group A |     Group B |
| ----------------------- | ----------: | ----------: | ----------: |
| n                       |       1,240 |         610 |         630 |
| Age, mean ± SD          | 61.2 ± 12.4 | 60.8 ± 12.1 | 61.6 ± 12.7 |
| Female, n (%)           | 590 (47.6%) | 284 (46.6%) | 306 (48.6%) |
| Stage III/IV, n (%)     | 410 (33.1%) | 190 (31.1%) | 220 (34.9%) |
| Events, n (%)           | 312 (25.2%) | 140 (23.0%) | 172 (27.3%) |
| Follow-up, median [IQR] |  36 [18–60] |  38 [20–62] |  34 [17–58] |

---

# 14. Charts Tab

## 14.1 Purpose

Answer:

```text
What do the variables look like?
```

This should be a general visual explorer, not survival-specific.

## 14.2 Auto-Chart Rules

| Selected variables        | Suggested chart       |
| ------------------------- | --------------------- |
| One numeric variable      | Histogram + boxplot   |
| One categorical variable  | Bar chart             |
| Numeric + categorical     | Boxplot / violin plot |
| Two numeric variables     | Scatter plot          |
| Two categorical variables | Stacked bar chart     |
| Many numeric variables    | Correlation heatmap   |

## 14.3 Contents

```text
variable selector
auto-generated chart
histogram
boxplot
bar chart
scatter plot
correlation heatmap
missingness chart
group split option
summary statistics beside chart
```

Example UI:

```text
Variable 1: [age]
Variable 2: [event_status]

Suggested chart:
Boxplot of age by event status
```

---

# 15. Survival Analysis Tab

## 15.1 Purpose

Answer:

```text
How does survival/event-free probability change over time?
```

## 15.2 Contents

```text
cohort filters
survival variable setup summary
group-by selector
Kaplan–Meier curve
confidence interval toggle
censor marks toggle
number-at-risk table
survival summary cards
log-rank test
optional Cox model
export plot/results
warnings
```

## 15.3 Filters

Filters should be based on columns marked as usable filters.

Examples:

```text
age range
sex/gender
diagnosis
treatment
stage
mutation
comorbidity
site
lab value range
```

Always show current filtered cohort size:

```text
Current cohort: 284
Events: 71
Censored: 213
```

## 15.4 Survival Plot

Kaplan–Meier plot should include:

```text
survival probability vs time
confidence interval band
censoring marks
group curves if group-by selected
hover labels
download image button
```

## 15.5 Number-at-Risk Table

Below plot:

```text
time points: 0, 6, 12, 24, 36, 60 months
rows: groups
values: number still at risk
```

## 15.6 Survival Summary Cards

```text
Median survival
1-year survival
3-year survival
5-year survival
Events
Censored
Log-rank p-value if groups compared
```

## 15.7 Warnings

Examples:

```text
Group B has only 8 patients. Interpret cautiously.
Group B has only 2 events. Cox/log-rank results may be unstable.
Curves cross; log-rank test may be misleading.
Median survival was not reached.
```

---

# 16. Dataset Suitability Logic

## 16.1 Good Fit

A dataset fits the app if it has:

```text
one row per patient/case/admission
structured columns
follow-up time or event dates
event/censor status or derivable event status
clinical/demographic/treatment variables
```

## 16.2 Partial Fit

The app can still show quality/cohort/charts if the dataset lacks survival columns.

Example:

```text
No survival time column found.
Survival Analysis tab disabled.
You can still use Data Quality, Cohort Overview, and Charts.
```

## 16.3 Poor Fit

Not suitable for this dashboard:

```text
ClinicalTrials.gov metadata export
imaging-only dataset
free-text notes only
aggregate summary table
genomic raw sequence files
one row = one trial rather than one patient
```

---

# 17. Dataset Sources for Testing

Use practice and real public/controlled datasets to test the dashboard.

## 17.1 Easy Testing Datasets

```text
UCI Heart Failure Clinical Records
scikit-survival WHAS500
scikit-survival Veterans Lung Cancer
lifelines built-in datasets
Kaggle survival-style datasets
```

## 17.2 More Realistic Datasets

```text
MIMIC-IV demo / full MIMIC-IV
SEER cancer registry
TCGA / cBioPortal clinical data
NHANES + linked mortality
```

## 17.3 Important Clarification

ClinicalTrials.gov usually provides:

```text
trial metadata
study status
enrollment count
summary outcomes
aggregate results
```

It usually does **not** provide:

```text
patient-level time-to-event data
individual survival times
individual event indicators
```

So ClinicalTrials.gov exports are generally not suitable for this dashboard’s survival analysis.

---

# 18. Internal State Model

The app should maintain a session-level state object.

Example:

```python
state = {
    "raw_df": df,
    "clean_df": cleaned_df,
    "column_profile": {
        "age": {
            "detected_type": "numeric",
            "missing_pct": 0.05,
            "unique_count": None,
            "examples": [55, 63, 72]
        }
    },
    "mapping": {
        "id_col": "patient_id",
        "time_col": "surv_days",
        "event_col": "status",
        "event_values": ["Dead"],
        "censor_values": ["Alive"],
        "time_unit": "days"
    },
    "annotations": {
        "age": {
            "meaning": "Age",
            "uses": ["filter", "baseline_table", "cox_covariate"]
        },
        "sex": {
            "meaning": "Sex / gender",
            "uses": ["filter", "group", "baseline_table"]
        }
    },
    "validation": {
        "errors": [],
        "warnings": []
    },
    "filters": {
        "age": [40, 80],
        "sex": ["M", "F"]
    }
}
```

---

# 19. Core Backend Functions

Recommended module structure:

```text
src/
├── load_data.py
├── profile_data.py
├── infer_types.py
├── validate_mapping.py
├── clean_data.py
├── summarize.py
├── survival.py
├── plots.py
├── filters.py
└── export.py
```

## 19.1 `load_data.py`

Responsibilities:

```text
read CSV/XLSX/TSV
detect delimiter
handle encoding
standardize missing values
return dataframe
```

## 19.2 `profile_data.py`

Responsibilities:

```text
row count
column count
missing count
missing %
unique count
example values
min/max for numeric
date range for date
detected type
```

## 19.3 `validate_mapping.py`

Responsibilities:

```text
check time column exists
check event column exists
check time is numeric or derivable
check event values are mapped
check negative times
check empty rows
check duplicated IDs
generate errors and warnings
```

## 19.4 `survival.py`

Responsibilities:

```text
prepare survival dataframe
convert event status to binary
fit Kaplan–Meier model
calculate median survival
calculate survival at fixed time points
run log-rank test
fit optional Cox model
```

## 19.5 `plots.py`

Responsibilities:

```text
KM curves
histograms
bar charts
boxplots
scatter plots
correlation heatmap
missingness heatmap
forest plot later
```

---

# 20. Event Status Conversion Logic

## 20.1 Text to Binary

Example:

```python
def convert_event_status(series, event_values, censor_values=None):
    event_mask = series.isin(event_values)

    if censor_values is not None:
        censor_mask = series.isin(censor_values)
        unknown_mask = ~(event_mask | censor_mask)
    else:
        censor_mask = ~event_mask & series.notna()
        unknown_mask = series.isna()

    return event_mask.astype(int), unknown_mask
```

## 20.2 Missing Values

Do not automatically treat missing as censored without confirmation.

Options:

```text
exclude missing event status
treat missing as censored
ask user to define missing behavior
```

Recommended default:

```text
Missing/unknown event status = exclude from survival analysis unless user confirms otherwise.
```

---

# 21. UX Decisions

## 21.1 Good UX Choices

```text
Use plain-language errors.
Show real values from the dataset.
Ask the user to assign meaning, not technical data types.
Keep required survival setup separate from optional annotations.
Disable survival analysis until required mapping is valid.
Show warnings without blocking the whole app.
Use “Fix” buttons to navigate to the specific issue.
```

## 21.2 Avoid

```text
Raw technical messages like “expects binary but received text”
Forcing users to rename dataset columns
Requiring all columns to be annotated before dashboard opens
Assuming missing values mean censored
Assuming numeric categories are invalid
Using only dropdowns in column headers for all mapping
```

---

# 22. MVP Development Plan

## Phase 1: Basic Upload + Profiling

```text
Upload CSV/XLSX
Preview table
Detect column types
Show missing/unique summary
```

## Phase 2: Required Survival Mapping

```text
Select time column
Select event column
Map event/censor values
Validate mapping
Convert text status to binary
```

## Phase 3: Cohort Overview

```text
Total rows/patients
Event/censor count
Basic demographic summaries
Baseline characteristics table
```

## Phase 4: Charts

```text
Numeric histograms
Categorical bar charts
Boxplots
Scatter plots
Correlation heatmap
```

## Phase 5: Survival Analysis

```text
Kaplan–Meier curve
Group-by selector
Censor marks
Confidence intervals
Number-at-risk table
Survival summary cards
Log-rank test
```

## Phase 6: Data Quality

```text
Missingness table
Missingness heatmap
Duplicate ID check
Invalid value checks
PHI warning
Analysis-readiness summary
```

## Phase 7: Export

```text
Download cleaned/mapped dataset
Download Table 1
Download plots
Export HTML/PDF report
Save mapping config
```

## Phase 8: Advanced Features

```text
Cox proportional hazards model
Hazard ratio forest plot
Start-stop survival format
Time-varying covariates
Large dataset support with Polars/DuckDB
Multi-user deployment
```

---

# 23. Final Product Definition

The final product should be described as:

```text
A web-based exploratory dashboard for tabular patient-level cohort datasets, with automatic data profiling, flexible column annotation, cohort summaries, visual exploration, and optional survival analysis when time-to-event and event-status fields are available.
```

It should **not** be described as:

```text
A tool for any medical dataset
A tool that automatically understands all clinical data
A ClinicalTrials.gov patient-data downloader
A replacement for full statistical software
```

The strongest version of the project is a focused, practical, upload-based tool for helping users quickly understand a structured clinical cohort dataset.
