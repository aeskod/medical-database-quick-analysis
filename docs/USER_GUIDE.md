# Medical Dataset Explorer

## User and Administrator Guide

Medical Dataset Explorer is a local Streamlit application for patient-level tables.
It checks data quality, describes cohorts, creates charts, and runs basic survival analyses.

The application does not change source data automatically.
It does not replace clinical review or a statistical analysis plan.

> [!WARNING]
> Use the results only for exploratory research.
> Do not use the application as a diagnostic or treatment system.

## Contents

1. [Intended use](#1-intended-use)
2. [First analysis](#2-first-analysis)
3. [Installation and startup](#3-installation-and-startup)
4. [Input data](#4-input-data)
5. [Main concepts](#5-main-concepts)
6. [Navigation and session state](#6-navigation-and-session-state)
7. [Dataset page](#7-dataset-page)
8. [Setup page](#8-setup-page)
9. [Data Quality page](#9-data-quality-page)
10. [Cohort Overview page](#10-cohort-overview-page)
11. [Charts page](#11-charts-page)
12. [Survival Analysis page](#12-survival-analysis-page)
13. [Export page](#13-export-page)
14. [Interpretation guidance](#14-interpretation-guidance)
15. [Privacy and security](#15-privacy-and-security)
16. [Troubleshooting](#16-troubleshooting)
17. [Administrator reference](#17-administrator-reference)
18. [Analysis checklist](#18-analysis-checklist)

## 1. Intended use

### 1.1 Suitable data

The application works best with one row for each independent analysis unit.

An analysis unit can be:

- A patient.
- A study participant.
- An admission.
- A clinical case.
- One final observation for a survival analysis.

The table can contain:

- Demographic variables.
- Diagnoses.
- Treatments.
- Disease stages.
- Laboratory values.
- Biomarkers.
- Dates.
- Follow-up time.
- Event status.
- A de-identified patient ID.

### 1.2 Data that needs preparation

Convert the data before upload when one row represents one visit.
The survival workflow expects one final analysis row for each patient.

Prepare these data types before use:

- Repeated visits.
- Longitudinal measurements.
- Start-stop intervals.
- Recurrent events.
- Competing risks.
- Aggregated publication tables.
- Free text.
- Medical images.
- Genomic sequences.

The repository contains a Stanford heart transplant dataset with start-stop intervals.
The application does not offer this file as an interactive example.

### 1.3 Features outside the current scope

The current version does not include:

- Automatic data cleaning.
- Missing-value imputation.
- Cox regression.
- Confounder adjustment.
- Causal inference.
- Competing-risk analysis.
- Recurrent-event analysis.
- Persistent project storage.
- User accounts.
- Access control.
- Multi-user collaboration.
- An audit log.

These limits define the current scope.
They do not prevent general data review or Kaplan–Meier analysis.

## 2. First analysis

This tutorial uses the included lung cancer cohort.
The complete workflow usually takes 10 to 15 minutes.

### 2.1 Start the application

Activate the project environment:

~~~bash
source venv/bin/activate
streamlit run app.py
~~~

Open the local URL that Streamlit prints.
The default URL is usually http://localhost:8501.

### 2.2 Load the example dataset

1. Open **Dataset**.
2. Expand **Try an example dataset**.
3. Select **Lung cancer cohort (recommended)**.
4. Select **Load example dataset**.
5. Check the table size.

The table must contain 228 rows and 10 columns.
The application selects **Run survival analysis** as the goal.

### 2.3 Confirm the survival mapping

1. Select **Continue to Survival Setup**.
2. Keep **Use a follow-up duration column**.
3. Select **time** as the follow-up column.
4. Select **status** as the event column.
5. Select **1** as the event value.
6. Select **0** as the censor value.
7. Select **days** as the time unit.
8. Keep the patient ID empty.
9. Select **sex** as the optional group.
10. Select **Confirm survival mapping**.

The confirmed mapping must produce:

- 228 usable rows.
- 165 events.
- 63 censored observations.

The suggested mapping does not confirm the clinical meaning of a column.
Always compare the mapping with the source data dictionary.

### 2.4 Review data quality

1. Open **Data Quality**.
2. Read the **Issues** section.
3. Expand **Detailed diagnostics**.
4. Review missing values.
5. Review duplicate checks.
6. Review excluded rows.
7. Note the warning about the missing patient ID.

A warning does not always block an analysis.
It identifies a condition that needs review.

### 2.5 Review the cohort

1. Open **Cohort Overview**.
2. Check the row, event, and censor counts.
3. Expand **Build a baseline characteristics table**.
4. Select **sex** in **Group by**.
5. Review the selected variables.
6. Download the table when required.

### 2.6 Create a chart

1. Open **Charts**.
2. Keep **Auto** as the chart type.
3. Select **age** as the X variable.
4. Keep the Y variable empty.
5. Review the histogram and box plot.
6. Use the Plotly camera button to save an image.

### 2.7 Run the survival analysis

1. Open **Survival Analysis**.
2. Keep confidence intervals visible.
3. Keep censor marks visible.
4. Select **sex** as the group variable.
5. Review the warning messages.
6. Review the Kaplan–Meier curves.
7. Review the log-rank result.
8. Review the number-at-risk table.
9. Review survival estimates at selected times.

Do not assign clinical labels to coded values without a data dictionary.
Display labels do not change the source data.

### 2.8 Save the results

1. Open **Export**.
2. Download the current dataset when required.
3. Download the mapped analysis table.
4. Save the JSON configuration.
5. Select **Prepare combined HTML and PDF reports**.
6. Download the required report.

Save the JSON configuration before you close the session.
The configuration can restore the mapping for a compatible table.

## 3. Installation and startup

### 3.1 Requirements

Install these tools:

- Python 3.10 or later.
- pip.
- A modern web browser.

The project depends on:

- Streamlit.
- pandas.
- NumPy.
- openpyxl.
- lifelines.
- Plotly.
- ReportLab.
- pytest.

### 3.2 macOS or Linux

Run these commands from the project directory:

~~~bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
~~~

### 3.3 Windows PowerShell

Run these commands from the project directory:

~~~powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
~~~

If PowerShell blocks activation, start Streamlit directly:

~~~powershell
.\venv\Scripts\streamlit.exe run app.py
~~~

### 3.4 Stop the application

Return to the terminal.
Press Ctrl+C.

### 3.5 Verify the installation

Run the complete test suite:

~~~bash
venv/bin/python -m pytest
~~~

Use this command when the environment is active:

~~~bash
python -m pytest
~~~

Run the additional checks:

~~~bash
python -m compileall app.py src tests
git diff --check
~~~

Do not use python app.py for normal operation.
Streamlit must start the application.

## 4. Input data

### 4.1 Supported files

| Format | Extension | Notes |
|---|---|---|
| CSV | .csv | Comma, tab, semicolon, or pipe |
| TSV | .tsv | Tab-separated |
| Text table | .txt | Automatic delimiter detection |
| Excel | .xlsx | Worksheet selection |

The application does not support .xls, .xlsm, Parquet, or dataset JSON files.

### 4.2 Upload limits

| Limit | Value |
|---|---:|
| File size | 50 MB |
| Rows | 1,000,000 |
| Columns | 1,000 |
| Column-name length | 512 characters |
| Uncompressed XLSX size | 200 MB |
| XLSX archive entries | 10,000 |

Each column must have a unique and non-empty name.
Each text row must have the expected number of fields.

### 4.3 Encodings

Automatic detection supports:

- UTF-8.
- UTF-8 with a byte-order mark.
- UTF-16.
- UTF-32.
- Windows-1252.
- Latin-1.

The interface also offers GB18030 and Shift-JIS.
Select the encoding manually when text appears damaged.

An encoding change reloads the table.
The reload clears dependent analysis settings.

### 4.4 Missing values

The application treats empty cells as missing.
It also recognizes these text markers:

~~~text
NA, N/A, na, n/a, NULL, null, None, none, missing, Missing
~~~

Spaces around a marker do not change the result.
Use one clear missing-value marker in the source table.

### 4.5 Duration-based survival data

Use a table such as:

~~~csv
patient_id,followup_days,status,treatment,age
P001,365,death,A,67
P002,420,alive,B,59
P003,180,death,A,71
~~~

The follow-up column must contain finite, nonnegative numbers.
The event column must have documented values.

### 4.6 Date-based survival data

Use a table such as:

~~~csv
patient_id,index_date,event_date,last_followup_date,treatment
P001,2024-01-10,2024-08-04,2024-08-04,A
P002,2024-02-01,,2025-02-01,B
P003,2024-03-15,2024-09-20,2024-09-20,A
~~~

Use ISO dates in YYYY-MM-DD format.
The application rejects ambiguous dates such as 01/02/2025.

For an observed event, the application calculates:

~~~text
time = event date - start date
event = 1
~~~

For a censored observation, the application calculates:

~~~text
time = last follow-up date - start date
event = 0
~~~

### 4.7 Time and event rules

The follow-up column must:

- Contain numbers.
- Use one time unit.
- Exclude negative values.
- Exclude infinite values.

Zero follow-up is allowed.
The application reports it as a warning.

A missing follow-up value excludes the row from survival analysis.
It does not remove the row from the source table.

The user must define event and censor values.
Do not assume that 1 always means an event.

### 4.8 Patient IDs

Use a stable, de-identified patient ID.
The ID helps the application find repeated patients.

When no ID is selected, each row represents one case.
Repeated IDs produce a warning.

Do not delete repeated IDs without review.
They can indicate visits, intervals, or data errors.

### 4.9 Group variables

A useful group variable usually contains two to eight levels.

Examples include:

- Treatment group.
- Sex.
- Disease stage.
- Risk category.
- Study center.

Do not use a unique patient ID as a group.
Many small groups produce unstable estimates.

### 4.10 Included example data

The **Dataset** page offers two examples:

- The lung cancer cohort.
- The Rossi recidivism cohort.

The lung dataset comes from a North Central Cancer Treatment Group study.
Loprinzi and colleagues published the cited study in 1994.

The Rossi dataset describes 432 people released from Maryland prisons.
Rossi, Berk, and Lenihan published the original study in 1980.

The Stanford file contains time-varying heart transplant records.
Crowley and Hu published the source analysis in 1977.

The project uses these files for software demonstration.
Review the original sources before scientific reuse.

## 5. Main concepts

### 5.1 Source table

The source table is the current uploaded or example table.
General summaries and charts use this table.

The application keeps the source values unchanged.
Exports can create separate derived tables.

### 5.2 Column profile

The application calculates a profile for each column.
The profile can contain:

- The detected type.
- The missing count and percentage.
- The unique-value count.
- Example values.
- The numeric parse rate.
- The date parse rate.
- Minimum and maximum values.
- Binary, categorical, and ID indicators.

The profile provides suggestions.
It does not replace a data dictionary.

### 5.3 Survival mapping

The survival mapping connects source columns to analysis roles.
It defines:

- Follow-up time.
- Event status.
- Event and censor values.
- Patient ID.
- Group.
- Time unit.
- Unknown-status rules.

The mapping does not rename source columns.

### 5.4 Survival analysis table

The confirmed mapping creates these internal columns:

| Column | Meaning |
|---|---|
| _time | Time to an event or censoring |
| _event | 1 for an event and 0 for censoring |
| _id | Selected patient ID |
| _group | Selected analysis group |

Rows without valid time and event values do not enter this table.
They remain in the source table.

### 5.5 Column annotation

Each annotation has a meaning and analysis-use flags.
One column can support more than one use.

Available uses include:

- Filter.
- Group.
- Baseline table.
- Chart.
- Ignore.

The ignore flag overrides the other use flags.

### 5.6 Censoring

A censored observation has no observed event during the available follow-up.
Censoring does not mean that an event can never occur.

## 6. Navigation and session state

The sidebar contains seven pages:

| Page | Purpose |
|---|---|
| **Dataset** | Load and preview data |
| **Setup** | Confirm roles and annotations |
| **Data Quality** | Review errors and warnings |
| **Cohort Overview** | Describe the cohort |
| **Charts** | Explore variables |
| **Survival Analysis** | Fit and compare survival curves |
| **Export** | Save data, settings, and reports |

The sidebar also shows:

- The current file name and size.
- The survival-mapping status.
- The data-quality status.
- The active survival-filter count.

Settings exist only in the current Streamlit session.
Save required files before you close the application.

A new table clears:

- The survival mapping.
- Column annotations.
- Survival filters.
- Chart selections.
- Prepared reports.

This reset prevents old settings from affecting new data.

## 7. Dataset page

### 7.1 Upload a file

1. Select **Browse files**.
2. Select a supported file.
3. Check the text encoding when applicable.
4. Select an Excel worksheet when applicable.
5. Wait for **Dataset loaded successfully**.

An invalid new file does not replace the valid current table.

### 7.2 Load an example

The application includes:

- **Lung cancer cohort (recommended)**.
- **Rossi recidivism cohort**.

The example can temporarily replace an uploaded table.
Select **Use selected upload** to restore the uploaded file.

A typical Rossi mapping uses:

| Role | Value |
|---|---|
| Follow-up time | week |
| Event status | arrest |
| Event occurred | 1 |
| Censored | 0 |
| Time unit | weeks |

Verify this mapping against the source description.

### 7.3 Review file metadata

The page shows:

- File name.
- File size.
- Row count.
- Column count.
- File format.
- Delimiter.
- Encoding.
- Selected Excel worksheet.

### 7.4 Select an analysis goal

Select one goal:

- **Explore and review the dataset**.
- **Run survival analysis**.

The first goal does not require a survival mapping.
Data quality, cohort summaries, charts, and current-data export remain available.

The second goal directs the workflow to **Setup**.

### 7.5 Review the preview

The default preview shows 20 rows.
Select **Show more rows** to show up to 100 rows.

The column header shows the detected type.
A red cell background identifies a missing value.

The preview does not change the source table.
Removing the uploaded file clears the active uploaded table.

### 7.6 Review the column profile

Expand **Column profile and type detection**.
Review these fields when a suggestion appears incorrect:

- detected_type.
- missing_percent.
- unique_count.
- numeric_parse_rate.
- date_parse_rate.
- is_binary_like.
- is_id_like.

## 8. Setup page

### 8.1 Select a time source

Select one mode:

- **Use a follow-up duration column**.
- **Derive from date columns**.

Select the mode that matches the source table.

### 8.2 Configure a duration column

1. Select **Follow-up / survival time column**.
2. Select **Event/status column**.
3. Select the event values.
4. Select the censor values.
5. Review unknown values.
6. Select the time unit.
7. Select the optional patient ID.
8. Select the optional group.
9. Review **Validation summary**.
10. Select **Confirm survival mapping**.

The page shows suggested columns first.
Select **Search all columns** to open the full list.

Expand **Why these columns were recommended** for suggestion details.
A high score does not confirm clinical meaning.

### 8.3 Define event values

**Which value(s) mean the event occurred?** creates _event = 1.
**Which value(s) explicitly mean censored?** creates _event = 0.

One source value cannot belong to both groups.
Review all unique event values.

The application suggests 1 and 0 only for a clear binary pair.
It can also recognize clear words such as dead and alive.

Unassigned values appear under **Currently unmapped**.

### 8.4 Handle unknown and missing statuses

Expand **Advanced event handling**.

For an unknown non-empty value, select one action:

- Exclude the row.
- Treat the value as censored.
- Treat the value as an event.

For a missing value, select one action:

- Exclude the row.
- Treat the value as censored.

Exclusion is the conservative default.
Change it only when the data dictionary supports another rule.

### 8.5 Select the time unit

Available units are:

- days.
- weeks.
- months.
- years.
- unknown.

The unit controls labels and year-based estimates.
It does not convert source values.

Selecting years for a month column does not divide values by 12.

### 8.6 Derive follow-up from dates

1. Select **Derive from date columns**.
2. Select **Start date column**.
3. Select **Event date column**.
4. Select **Last follow-up date column**.
5. Select the missing-event-date rule.
6. Select the optional ID.
7. Select the optional group.
8. Confirm the mapping.

Date-derived follow-up always uses days.

### 8.7 Resolve validation messages

The confirmation button remains disabled for blocking errors.

Blocking conditions include:

- A missing required column.
- Nonnumeric follow-up values.
- Negative or infinite follow-up.
- No event value.
- Overlapping event and censor values.
- One column assigned to incompatible roles.
- Invalid or ambiguous dates.
- A negative date-derived duration.
- An event after the last contact.
- No usable rows.

Warnings include:

- Missing values.
- Zero follow-up.
- Repeated IDs.
- Small groups.
- Few events.
- Few censored observations.

Select a **Fix** link to return to the related control.

### 8.8 Edit column annotations

Expand **Advanced: column meanings and analysis uses**.

You can edit:

- **Meaning**.
- **Custom meaning**.
- **Use as filter**.
- **Use as group**.
- **Use in baseline table**.
- **Use in charts**.
- **Ignore in analysis**.

The current interface does not show the Cox-model flag.
The current version does not fit Cox models.

Select **Save annotations** after an edit.
Select **Reset to suggested annotations** to remove manual changes.

## 9. Data Quality page

The page builds a report from the current table.
It does not change source values.

### 9.1 Read the overall status

| Status | Meaning |
|---|---|
| Red error | A blocking problem exists |
| Yellow warning | Manual review is required |
| Green status | No major problem was found |

A green status does not confirm clinical validity.

### 9.2 Review the summary cards

The page can show:

- Missing-cell percentage.
- Repeated patient IDs.
- Invalid ages.
- Unknown event statuses.
- Zero-time rows.
- Usable survival rows.

Survival-specific values show **Not mapped** without a confirmed mapping.

### 9.3 Review issues

The issue list shows errors before warnings and information.
Each item includes a category and affected data.

Correct a source-data error in the source file.
Correct a mapping error on **Setup**.

### 9.4 Review missing values

A column gets a warning at 20 percent missing values.
A column gets a high-missingness warning at 50 percent.

Detailed diagnostics include:

- A summary for each column.
- A missingness heatmap.
- The first 50 affected rows.

The heatmap shows no more than 250 rows.
The column summary uses all rows.

### 9.5 Review duplicates

The application checks:

- Fully duplicated rows.
- Repeated selected patient IDs.

A repeated ID can indicate visits or observation intervals.
Review the study structure before you remove rows.

### 9.6 Review ages

The application finds age columns from names and annotations.
The accepted age range is 0 through 120.

It also counts nonnumeric values.
Correct the annotation when the selected column is not an age.

### 9.7 Review dates

The application checks:

- Date parsing.
- An event before the start date.
- Last contact before the start date.
- An event after the last contact.

A chronology error blocks date-derived survival analysis.

### 9.8 Review survival quality

After mapping, the page can show:

- Source rows.
- Usable rows.
- Excluded rows.
- Events.
- Censored observations.
- Event percentage.
- Missing follow-up.
- Nonnumeric follow-up.
- Negative, infinite, and zero follow-up.
- Unknown event values.
- Status-handling rules.
- Exclusion reasons.

One row can have more than one exclusion reason.
Reason counts can exceed the excluded-row count.

### 9.9 Review group quality

The group table shows:

- Group size.
- Event count.
- Censor count.
- Median follow-up.

The application warns when:

- A group has fewer than five rows.
- The table has more than eight groups.
- A group has no events.
- A group has no censored observations.

### 9.10 Review possible sensitive columns

The application searches for:

- Names.
- Contact fields.
- Medical record numbers.
- Address-like fields.
- Email addresses.
- Phone-like values.
- Long free text.
- ID-like columns.

This check uses heuristics.
It does not replace a privacy audit.

## 10. Cohort Overview page

### 10.1 Review cohort metrics

The page can show:

- Patient count.
- Row count.
- Events.
- Censored observations.
- Event percentage.
- Median follow-up.
- Median age.
- Complete rows.
- Missing-cell percentage.

With a selected ID, patient count uses unique non-empty IDs.
Without an ID, patient count equals row count.

Median follow-up uses the reverse Kaplan–Meier method.
This estimates the potential follow-up period.

### 10.2 Review key characteristics

The section uses column meanings from saved annotations.
It can show age, sex, diagnosis, treatment, and outcome fields.

Numeric age data gets a summary and histogram.
Categorical data gets counts and a bar chart.

### 10.3 Build a baseline table

1. Expand **Build a baseline characteristics table**.
2. Select **Group by** when required.
3. Review continuous variables.
4. Review categorical variables.
5. Select the maximum category count.
6. Select the missing-category option.
7. Review the result.
8. Download the CSV file.

The table starts with:

- n.
- Event count and percentage.
- Median observed duration and interquartile range.

A continuous variable uses this format:

~~~text
mean +/- SD; median [Q1, Q3]
~~~

A category level uses this format:

~~~text
n (percent)
~~~

The table combines rare levels as **Other**.
The category limit can range from 3 through 30.

With a selected ID, the table uses the first row for each patient.
Make sure that the first row contains valid baseline values.

Rows without a group remain in **Overall**.
They do not create a separate group column.

## 11. Charts page

The page uses columns with **Use in charts**.

### 11.1 Use Auto mode

**Auto** selects a chart from the selected variable types.

| X variable | Y variable | Result |
|---|---|---|
| Numeric | Empty | Histogram and box plot |
| Categorical | Empty | Bar chart |
| Numeric | Categorical | Box plot |
| Categorical | Numeric | Box plot |
| Numeric | Numeric | Scatter plot |
| Date | Numeric | Time series |
| Categorical | Categorical | Stacked bar chart |

The interface explains the selection.

### 11.2 Select a chart type

| Chart | Requirements |
|---|---|
| **Histogram + box plot** | Numeric X |
| **Bar chart** | Categorical X |
| **Box plot** | One numeric and one categorical variable |
| **Violin plot** | One numeric variable and an optional category |
| **Scatter plot** | Numeric X and Y |
| **Time series** | Date X and numeric Y |
| **Stacked bar chart** | Categorical X and Y |
| **Correlation heatmap** | At least two nonconstant numeric columns |
| **Missingness bar chart** | The current table |

The correlation heatmap uses Pearson correlation.
It uses no more than 20 numeric columns.

The time-series chart sorts rows by date.
It does not aggregate repeated dates.

### 11.3 Add color or groups

Use **Color/group variable** to divide plotted values.
A scatter plot can use a numeric color variable.

Other chart types need a categorical color variable.
They ignore an incompatible numeric color variable.

### 11.4 Change advanced options

Expand **Advanced chart options** to set:

- The maximum category count.
- Missing-category display.
- Percentage normalization for a stacked bar chart.

The chart combines rare levels as **Other**.
It shows selected missing categories as **Missing**.

### 11.5 Save a chart

Select PNG or SVG in the chart toolbar.
Use the Plotly camera button to save the image.

Select **Download chart as HTML** for an interactive file.
The HTML file contains the required Plotly code.

## 12. Survival Analysis page

This page requires a confirmed survival mapping.

### 12.1 Review analysis readiness

Resolve all blocking mapping errors before analysis.
Review warnings before you interpret a curve.

The page uses only rows in the survival analysis table.
It does not silently add excluded rows.

### 12.2 Apply cohort filters

Filters use columns with **Use as filter**.
Available controls depend on the column type.

The page can provide:

- Numeric ranges.
- Date ranges.
- Category selections.

The sidebar shows the active filter count.
Clear filters to return to the complete mapped cohort.

Check the remaining sample size after each filter.
Small filtered cohorts produce unstable results.

### 12.3 Select a group

Use **Group / stratification variable** for separate curves.
The list uses columns with **Use as group**.

Use two through eight clear groups.
Avoid identifiers and high-cardinality variables.

Display labels affect charts and tables only.
They do not change source values.

### 12.4 Read the Kaplan–Meier curve

The plot can show:

- The overall survival curve.
- A curve for each group.
- A 95 percent confidence interval.
- Censor marks.

The curve decreases at observed event times.
A censor mark does not cause a decrease.

Wide confidence intervals indicate low precision.
Few remaining patients usually cause wider intervals.

### 12.5 Read summary values

The page can show:

- Usable observations.
- Events.
- Censored observations.
- Median survival.
- Median follow-up.
- Maximum follow-up.

Median survival is unavailable when the curve never reaches 0.5.
This result is not an application error.

### 12.6 Read the log-rank test

The log-rank test compares complete survival curves.
It tests whether groups have the same survival experience.

The result does not adjust for:

- Age.
- Disease stage.
- Treatment selection.
- Center effects.
- Other confounders.

A small p-value does not prove causation.
Crossing curves can reduce the usefulness of the test.

For three through eight groups, the application also runs pairwise tests.
It adjusts pairwise p-values with the Holm method.

### 12.7 Read the number-at-risk table

The table shows observations still at risk at selected times.
Use it to judge the reliable part of each curve.

Treat estimates with few remaining observations cautiously.
The far right side of a curve often has low precision.

### 12.8 Read survival estimates

The page reports survival probabilities at selected times.
It can also show one-year, three-year, and five-year estimates.

Year-based estimates require a known time unit.
They also require sufficient follow-up.

The application does not extrapolate beyond observed follow-up.

## 13. Export page

### 13.1 Export the current dataset

**current_dataset.csv** contains the current parsed source table.
It does not require a survival mapping.

### 13.2 Export mapped survival data

**cleaned_mapped_data.csv** contains source columns and internal roles.
It includes only rows accepted by the survival mapping.

Review excluded-row counts before you use this file.

### 13.3 Export the configuration

**analysis_configuration.json** can contain:

- The survival mapping.
- Event-handling rules.
- Time-source settings.
- Column annotations.

Load the configuration only with a compatible table.
The application validates column names and field types.

### 13.4 Export reports

Select **Prepare combined HTML and PDF reports**.
The application creates:

- **medical_dataset_report.html**.
- **medical_dataset_report.pdf**.

The reports contain table-based summaries.
They do not replace a reviewed statistical report.

### 13.5 Spreadsheet safety

CSV export neutralizes text that can execute as a spreadsheet formula.
This protects users who open an export in spreadsheet software.

Review exported content before distribution.
The application cannot determine every disclosure risk.

## 14. Interpretation guidance

### 14.1 Use the correct analysis unit

Kaplan–Meier analysis assumes independent observations.
Repeated rows for one patient can violate this assumption.

Convert repeated records to a valid patient-level structure first.

### 14.2 Confirm time origin

All patients need a comparable start point.
Examples include diagnosis, enrollment, or treatment start.

Do not mix different start definitions in one analysis.

### 14.3 Confirm event coding

Check event and censor values against the data dictionary.
A reversed mapping reverses the meaning of the result.

### 14.4 Review exclusions

Report the number of excluded rows and the reasons.
Large or selective exclusions can bias results.

### 14.5 Review censoring

Kaplan–Meier analysis assumes non-informative censoring.
This means censoring should not depend on unobserved event risk.

The application cannot verify this assumption.

### 14.6 Review group comparisons

The log-rank test is unadjusted.
It does not control for baseline differences.

Use an adjusted model when the study question requires adjustment.
The current application does not fit that model.

### 14.7 Avoid causal claims

An association between group and survival does not prove a treatment effect.
Study design and confounding determine valid causal interpretation.

## 15. Privacy and security

### 15.1 Remove direct identifiers

Remove these fields before upload:

- Names.
- Addresses.
- Phone numbers.
- Email addresses.
- Medical record numbers.
- Government identifiers.
- Direct patient identifiers.
- Free text that can contain identifying details.

Use a de-identified study ID when an ID is necessary.

### 15.2 Understand local processing

The standard setup runs on the local computer.
The application does not include its own external API integration.

Browser and operating-system behavior still affects local security.
Use a trusted computer and an approved storage location.

### 15.3 Do not expose the local server

The application has no authentication or access control.
Do not expose it to a public network.

Add external protection before any shared deployment.

### 15.4 Review sensitive-column warnings

The sensitive-column check uses names and value patterns.
It can miss sensitive data.
It can also produce false warnings.

Complete a separate privacy review before data sharing.

### 15.5 Protect exports

Exports can contain the complete current table.
Store them only in approved locations.

Delete temporary files according to the applicable retention policy.

## 16. Troubleshooting

### 16.1 Streamlit command not found

Activate the environment:

~~~bash
source venv/bin/activate
~~~

Install the requirements when the command remains unavailable:

~~~bash
python -m pip install -r requirements.txt
~~~

### 16.2 The application does not open

1. Read the terminal output.
2. Open the printed local URL manually.
3. Check whether another process uses port 8501.
4. Start the application on another port when required.

~~~bash
streamlit run app.py --server.port 8502
~~~

### 16.3 The file does not load

Check:

- The extension.
- The file-size limit.
- The delimiter.
- The encoding.
- Unique and non-empty headers.
- Consistent field counts.
- The selected Excel worksheet.

### 16.4 Text appears damaged

Return to **Dataset**.
Select another encoding.

The reload clears dependent settings.

### 16.5 An expected column is not suggested

Expand **Column profile and type detection**.
Check its parse rates and detected type.

On **Setup**, select **Search all columns**.
Select the correct column manually.

### 16.6 The mapping cannot be confirmed

Read **Validation summary**.
Resolve each blocking error.

Typical causes include:

- Overlapping event and censor values.
- Nonnumeric follow-up.
- Negative follow-up.
- Invalid dates.
- One column assigned to two incompatible roles.
- No usable rows.

### 16.7 Median survival is unavailable

The curve did not reach 50 percent.
Report the available time-specific estimates instead.

### 16.8 A year-based estimate is unavailable

Check the selected time unit.
Check the maximum follow-up.

The application does not extrapolate beyond observed data.

### 16.9 A PDF export fails

Check that ReportLab is installed:

~~~bash
python -m pip install reportlab
~~~

Review the application error message.
Use the HTML report when PDF generation remains unavailable.

### 16.10 Settings disappeared

A table change resets dependent state.
A closed browser session also loses session-only settings.

Load a saved JSON configuration when the table is compatible.

## 17. Administrator reference

### 17.1 Project structure

~~~text
meddb_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── datasets/
│   ├── lung.csv
│   ├── rossi.csv
│   └── stanford_heart_transplants.csv
├── docs/
│   └── USER_GUIDE.md
├── src/
│   ├── charts.py
│   ├── cohort_overview.py
│   ├── column_annotations.py
│   ├── data_loading.py
│   ├── data_quality.py
│   ├── exports.py
│   ├── profiling.py
│   ├── role_suggestions.py
│   ├── survival_analysis.py
│   ├── survival_mapping.py
│   ├── survival_plots.py
│   └── upload_state.py
└── tests/
~~~

### 17.2 Module responsibilities

| Module | Responsibility |
|---|---|
| app.py | Streamlit interface and session state |
| src/data_loading.py | File validation and parsing |
| src/profiling.py | Missing-value normalization and column profiles |
| src/column_annotations.py | Column meanings and analysis uses |
| src/data_quality.py | Data-quality checks |
| src/cohort_overview.py | Cohort metrics and baseline tables |
| src/charts.py | Chart data and Plotly figures |
| src/survival_mapping.py | Mapping validation and internal roles |
| src/survival_analysis.py | Kaplan–Meier estimates and log-rank tests |
| src/survival_plots.py | Survival figures |
| src/exports.py | CSV, JSON, HTML, and PDF exports |
| src/upload_state.py | Uploaded-table identity and state resets |

### 17.3 Main session objects

| Key | Contents |
|---|---|
| uploaded_df | Current source table |
| profile_df | Column profile |
| survival_config | Confirmed survival mapping |
| survival_ready_df | Standardized survival table |
| column_annotations | Saved annotations |
| data_quality_report | Latest quality report |
| active_survival_filters | Active filters |
| group_value_labels | Display labels |

A table-content change clears dependent objects.

### 17.4 Survival configuration

The SurvivalConfig object stores:

~~~text
time_col
event_col
event_values
censor_values
id_col
group_col
time_unit
missing_event_handling
unmapped_event_handling
time_source
start_date_col
event_date_col
last_followup_date_col
~~~

The time_source value is duration or dates.

### 17.5 Analysis methods

The application uses lifelines.KaplanMeierFitter.
It uses multivariate_logrank_test for grouped comparisons.

For three through eight groups, it runs pairwise log-rank tests.
It applies the Holm correction to pairwise p-values.

It estimates median follow-up with reverse Kaplan–Meier.
It calculates correlation matrices with Pearson correlation.

### 17.6 Configuration import

The current JSON configuration format version is 1.
The maximum configuration size is 1 MB.

The importer rejects:

- An unknown format version.
- Unknown mapping fields.
- Invalid field types.
- Unknown annotation flags.
- Nonnumeric JSON constants.

### 17.7 Upload configuration

The .streamlit/config.toml file contains:

~~~toml
[server]
maxUploadSize = 50
~~~

This value matches the validation limit in src/data_loading.py.

### 17.8 Test coverage

Run:

~~~bash
venv/bin/python -m pytest
~~~

The tests cover:

- File loading and encodings.
- Uploaded-table changes.
- Profiles and annotations.
- Data-quality reports.
- Cohort summaries.
- Charts.
- Survival mapping.
- Filters.
- Kaplan–Meier estimates.
- Log-rank tests.
- Exports.
- Main interface workflows.

### 17.9 Deployment limits

The repository does not include a Dockerfile or cloud configuration.
The standard setup is local.

A shared deployment needs:

- TLS.
- Authentication.
- Authorization.
- Network restrictions.
- Safe logging.
- A retention policy.
- Monitoring.
- Backups.
- A review of medical-data requirements.

## 18. Analysis checklist

### Before upload

- [ ] One row represents one analysis unit.
- [ ] The table contains no direct identifiers.
- [ ] Headers are unique and non-empty.
- [ ] The follow-up unit is known.
- [ ] The event coding has documentation.
- [ ] The time origin is consistent.

### After upload

- [ ] The format and encoding are correct.
- [ ] The row and column counts are correct.
- [ ] Missing values appear as expected.
- [ ] Detected column types are reasonable.
- [ ] The correct analysis goal is selected.

### After setup

- [ ] The follow-up column is correct.
- [ ] Event and censor values are not reversed.
- [ ] Unknown statuses have a documented rule.
- [ ] The time unit matches the source data.
- [ ] Repeated IDs have an explanation.
- [ ] All blocking errors are resolved.
- [ ] Column annotations are saved.

### Before interpretation

- [ ] Missing values and exclusions were reviewed.
- [ ] Duplicates were reviewed.
- [ ] Date chronology was reviewed.
- [ ] Group sizes are sufficient.
- [ ] The number-at-risk table was reviewed.
- [ ] Confidence intervals were reviewed.
- [ ] Curve crossings were reviewed.
- [ ] The log-rank result is not treated as causal.

### Before distribution

- [ ] The JSON configuration was saved.
- [ ] Required CSV files were saved.
- [ ] Required charts were saved.
- [ ] The report was reviewed manually.
- [ ] Exports contain no prohibited data.
- [ ] Files use an approved storage location.
