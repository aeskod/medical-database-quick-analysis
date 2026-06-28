import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.profiling import normalize_missing_values, profile_dataframe


CHART_TYPE_OPTIONS = {
    "Auto": "auto",
    "Histogram": "histogram",
    "Bar chart": "bar",
    "Box plot": "box",
    "Scatter plot": "scatter",
    "Stacked bar chart": "stacked_bar",
    "Correlation heatmap": "correlation_heatmap",
    "Missingness bar chart": "missingness_bar",
}

CHART_TYPE_LABELS = {value: key for key, value in CHART_TYPE_OPTIONS.items()}
SPECIAL_MISSINGNESS_COLUMN = "__missingness__"
SPECIAL_CORRELATION_COLUMN = "__correlation__"
MISSING_LABEL = "Missing"
OTHER_LABEL = "Other"
TEXT_LENGTH_THRESHOLD = 80

ID_NAME_HINTS = {
    "id",
    "patient_id",
    "subject_id",
    "record_id",
    "case_id",
    "participant_id",
    "person_id",
    "sample_id",
    "mrn",
}
CATEGORICAL_NAME_HINTS = {
    "sex",
    "gender",
    "stage",
    "ecog",
    "karno",
    "status",
    "treatment",
    "arm",
    "group",
    "inst",
}
CONTINUOUS_NAME_HINTS = {
    "age",
    "weight",
    "height",
    "bmi",
    "cal",
    "meal",
    "loss",
    "time",
    "duration",
    "followup",
    "follow_up",
    "survival",
    "lab",
    "level",
    "score",
    "dose",
    "count",
}


def get_chart_variable_type(
    column_name: str,
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None = None,
) -> str:
    if column_name not in df.columns:
        return "unknown"

    normalized = normalize_missing_values(df[[column_name]])[column_name]
    profile_row = _profile_row_for_column(column_name, df, profile_df)
    unique_count = int(profile_row.get("unique_count") or 0)
    non_missing_count = int(profile_row.get("non_missing_count") or normalized.dropna().shape[0])
    unique_ratio = float(profile_row.get("unique_ratio") or 0)
    numeric_parse_rate = float(profile_row.get("numeric_parse_rate") or 0)
    date_parse_rate = float(profile_row.get("date_parse_rate") or 0)
    detected_type = str(profile_row.get("detected_type") or "")
    is_binary_like = bool(profile_row.get("is_binary_like"))
    is_id_like = bool(profile_row.get("is_id_like"))

    if _looks_like_id_column(column_name, unique_ratio, non_missing_count, is_id_like, numeric_parse_rate):
        return "id"

    if date_parse_rate >= 0.8 and numeric_parse_rate < 0.95:
        return "datetime"

    if detected_type == "text" and _looks_like_free_text(normalized):
        return "text"

    if _has_categorical_name_hint(column_name) and unique_count <= 50:
        return "categorical"

    if numeric_parse_rate >= 0.95 and not is_binary_like:
        if unique_count > 10 or _has_continuous_name_hint(column_name):
            return "numeric"

    if is_binary_like or detected_type in {"boolean", "categorical"} or 2 <= unique_count <= 30:
        return "categorical"

    if detected_type == "text" or _looks_like_free_text(normalized):
        return "text"

    return "unknown"


def recommend_chart_type(
    x_col: str | None,
    y_col: str | None,
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None = None,
) -> str:
    if x_col == SPECIAL_MISSINGNESS_COLUMN:
        return "missingness_bar"

    if x_col == SPECIAL_CORRELATION_COLUMN:
        return "correlation_heatmap"

    if x_col is None and y_col is None:
        return "none"

    if x_col is None:
        return "none"

    x_type = get_chart_variable_type(x_col, df, profile_df)
    y_type = get_chart_variable_type(y_col, df, profile_df) if y_col is not None else None

    if x_type in {"text", "id", "unknown"} or y_type in {"text", "id", "unknown"}:
        return "none"

    if y_col is None:
        if x_type == "numeric":
            return "histogram"
        if x_type == "categorical":
            return "bar"
        return "none"

    if {x_type, y_type} == {"numeric", "categorical"}:
        return "box"

    if x_type == "numeric" and y_type == "numeric":
        return "scatter"

    if x_type == "categorical" and y_type == "categorical":
        return "stacked_bar"

    return "none"


def explain_chart_recommendation(
    resolved_chart_type: str,
    x_col: str | None,
    y_col: str | None,
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None = None,
) -> str:
    if resolved_chart_type == "missingness_bar":
        return "Missingness by column was selected, so a missingness bar chart is recommended."

    if resolved_chart_type == "correlation_heatmap":
        return "Correlation heatmap was selected, so numeric variables will be compared."

    if x_col is None:
        return "Select a chartable X variable to receive a recommendation."

    x_type = get_chart_variable_type(x_col, df, profile_df)
    y_type = get_chart_variable_type(y_col, df, profile_df) if y_col is not None else None

    if resolved_chart_type == "histogram":
        return f"{x_col} is numeric, so a histogram is recommended."
    if resolved_chart_type == "bar":
        return f"{x_col} is categorical, so a bar chart is recommended."
    if resolved_chart_type == "box" and y_col is not None:
        return f"{x_col} is {x_type} and {y_col} is {y_type}, so a box plot is recommended."
    if resolved_chart_type == "scatter" and y_col is not None:
        return f"{x_col} and {y_col} are both numeric, so a scatter plot is recommended."
    if resolved_chart_type == "stacked_bar" and y_col is not None:
        return f"{x_col} and {y_col} are both categorical, so a stacked bar chart is recommended."

    return "The selected variables are not suitable for the supported chart types."


def prepare_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    normalized = normalize_missing_values(df[[column]])[column]
    return pd.to_numeric(normalized, errors="coerce").dropna()


def prepare_categorical_series(
    df: pd.DataFrame,
    column: str,
    max_levels: int = 20,
    include_missing: bool = True,
) -> pd.Series:
    normalized = normalize_missing_values(df[[column]])[column]
    if include_missing:
        prepared = normalized.fillna(MISSING_LABEL).astype(str)
    else:
        prepared = normalized.dropna().astype(str)

    max_levels = max(1, int(max_levels))
    value_counts = prepared.value_counts(sort=False)
    value_counts = value_counts.sort_index(kind="stable").sort_values(ascending=False, kind="stable")

    if len(value_counts) <= max_levels:
        return prepared

    top_levels = set(value_counts.iloc[:max_levels].index.astype(str))
    return prepared.where(prepared.isin(top_levels), OTHER_LABEL)


def build_chart_dataframe(
    df: pd.DataFrame,
    x_col: str | None = None,
    y_col: str | None = None,
    color_col: str | None = None,
    max_category_levels: int = 20,
    include_missing: bool = True,
) -> pd.DataFrame:
    selected_columns = [
        column
        for column in [x_col, y_col, color_col]
        if column is not None and column in df.columns
    ]
    selected_columns = list(dict.fromkeys(selected_columns))
    chart_df = normalize_missing_values(df[selected_columns]) if selected_columns else pd.DataFrame(index=df.index)
    required_numeric_columns = []

    for column in [x_col, y_col]:
        if column is None or column not in chart_df.columns:
            continue

        variable_type = get_chart_variable_type(column, df)
        if variable_type == "numeric":
            chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
            required_numeric_columns.append(column)
        elif variable_type == "categorical":
            chart_df[column] = prepare_categorical_series(
                df,
                column,
                max_levels=max_category_levels,
                include_missing=include_missing,
            )
        elif variable_type == "datetime":
            chart_df[column] = pd.to_datetime(chart_df[column], errors="coerce", format="mixed")

    if color_col is not None and color_col in chart_df.columns:
        color_type = get_chart_variable_type(color_col, df)
        if color_type == "numeric":
            chart_df[color_col] = pd.to_numeric(chart_df[color_col], errors="coerce")
        else:
            chart_df[color_col] = prepare_categorical_series(
                df,
                color_col,
                max_levels=max_category_levels,
                include_missing=include_missing,
            )

    if required_numeric_columns:
        chart_df = chart_df.dropna(subset=required_numeric_columns)

    return chart_df.reset_index(drop=True)


def plot_histogram(
    df: pd.DataFrame,
    x_col: str,
    color_col: str | None = None,
    nbins: int | None = None,
    title: str | None = None,
    max_levels: int = 20,
    include_missing: bool = True,
) -> go.Figure:
    plot_df = _prepare_numeric_plot_dataframe(
        df,
        [x_col],
        color_col,
        max_levels=max_levels,
        include_missing=include_missing,
    )
    fig = px.histogram(
        plot_df,
        x=x_col,
        color=color_col if color_col in plot_df.columns else None,
        nbins=nbins,
        title=title or f"Distribution of {x_col}",
    )
    _style_figure(fig)
    return fig


def plot_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    color_col: str | None = None,
    max_levels: int = 20,
    title: str | None = None,
    include_missing: bool = True,
) -> go.Figure:
    categorical_columns = [x_col] + ([color_col] if color_col else [])
    plot_df = _prepare_categorical_plot_dataframe(
        df,
        categorical_columns,
        max_levels=max_levels,
        include_missing=include_missing,
    )

    if color_col:
        count_df = plot_df.groupby([x_col, color_col], dropna=False).size().reset_index(name="count")
        category_order = (
            count_df.groupby(x_col)["count"]
            .sum()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        fig = px.bar(
            count_df,
            x=x_col,
            y="count",
            color=color_col,
            barmode="group",
            category_orders={x_col: category_order},
            title=title or f"Counts of {x_col}",
        )
    else:
        count_df = plot_df[x_col].value_counts().rename_axis(x_col).reset_index(name="count")
        fig = px.bar(
            count_df,
            x=x_col,
            y="count",
            title=title or f"Counts of {x_col}",
        )

    fig.update_yaxes(title_text="Count")
    _style_figure(fig)
    return fig


def plot_box_plot(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: str,
    color_col: str | None = None,
    max_levels: int = 20,
    title: str | None = None,
    include_missing: bool = True,
) -> go.Figure:
    selected_columns = list(dict.fromkeys([numeric_col, category_col] + ([color_col] if color_col else [])))
    working = normalize_missing_values(df[selected_columns])
    working[numeric_col] = pd.to_numeric(working[numeric_col], errors="coerce")
    drop_columns = [numeric_col]
    if not include_missing:
        drop_columns.extend([category_col] + ([color_col] if color_col else []))
    working = working.dropna(subset=drop_columns)
    plot_df = pd.DataFrame(
        {
            numeric_col: working[numeric_col],
            category_col: prepare_categorical_series(
                working,
                category_col,
                max_levels=max_levels,
                include_missing=include_missing,
            ),
        },
        index=working.index,
    )

    if color_col and color_col not in {numeric_col, category_col}:
        plot_df[color_col] = prepare_categorical_series(
            working,
            color_col,
            max_levels=max_levels,
            include_missing=include_missing,
        )

    fig = px.box(
        plot_df,
        x=category_col,
        y=numeric_col,
        color=color_col if color_col in plot_df.columns else None,
        title=title or f"{numeric_col} by {category_col}",
        points="outliers",
    )
    _style_figure(fig)
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str | None = None,
    max_levels: int = 20,
    include_missing: bool = True,
) -> go.Figure:
    plot_df = _prepare_numeric_plot_dataframe(
        df,
        [x_col, y_col],
        color_col,
        max_levels=max_levels,
        include_missing=include_missing,
    )
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in plot_df.columns else None,
        title=title or f"{y_col} vs {x_col}",
    )
    _style_figure(fig)
    return fig


def plot_stacked_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_levels: int = 20,
    normalize: bool = False,
    title: str | None = None,
    include_missing: bool = True,
) -> go.Figure:
    plot_df = _prepare_categorical_plot_dataframe(
        df,
        [x_col, y_col],
        max_levels=max_levels,
        include_missing=include_missing,
    )
    count_df = plot_df.groupby([x_col, y_col], dropna=False).size().reset_index(name="count")

    if normalize:
        totals = count_df.groupby(x_col)["count"].transform("sum")
        count_df["percent"] = (count_df["count"] / totals * 100).round(2)
        value_col = "percent"
        y_axis_title = "Percent"
    else:
        value_col = "count"
        y_axis_title = "Count"

    category_order = (
        count_df.groupby(x_col)["count"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    fig = px.bar(
        count_df,
        x=x_col,
        y=value_col,
        color=y_col,
        barmode="stack",
        category_orders={x_col: category_order},
        title=title or f"{y_col} by {x_col}",
    )
    fig.update_yaxes(title_text=y_axis_title)
    _style_figure(fig)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    method: str = "pearson",
    max_columns: int = 20,
    title: str = "Correlation heatmap",
) -> go.Figure:
    if method not in {"pearson", "spearman"}:
        raise ValueError("Correlation method must be either 'pearson' or 'spearman'.")

    selected_columns = numeric_cols if numeric_cols is not None else _numeric_chart_columns(df)
    selected_columns = [column for column in selected_columns if column in df.columns]
    selected_columns = selected_columns[:max_columns]

    numeric_df = pd.DataFrame(
        {
            column: pd.to_numeric(normalize_missing_values(df[[column]])[column], errors="coerce")
            for column in selected_columns
        }
    ).dropna(axis=1, how="all")

    if len(numeric_df.columns) < 2:
        raise ValueError("Correlation heatmap requires at least two numeric variables.")

    corr = numeric_df.corr(method=method)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "Correlation"},
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title="", yaxis_title="")
    _style_figure(fig)
    return fig


def plot_missingness_bar(
    df: pd.DataFrame,
    title: str = "Missingness by column",
) -> go.Figure:
    normalized = normalize_missing_values(df)
    row_count = len(normalized)
    missing_df = pd.DataFrame(
        {
            "column_name": normalized.columns.astype(str),
            "missing_percent": [
                round((int(normalized[column].isna().sum()) / row_count * 100) if row_count else 0.0, 2)
                for column in normalized.columns
            ],
        }
    ).sort_values("missing_percent", ascending=False)

    fig = px.bar(
        missing_df,
        x="column_name",
        y="missing_percent",
        title=title,
        labels={"column_name": "Column", "missing_percent": "Missing (%)"},
    )
    fig.update_yaxes(title_text="Missing (%)", range=[0, 100])
    _style_figure(fig)
    return fig


def plot_missingness_heatmap(
    df: pd.DataFrame,
    max_rows: int = 250,
    title: str = "Missingness heatmap",
) -> go.Figure:
    missing = normalize_missing_values(df).head(max_rows).isna().astype(int)
    display = missing.T.replace({0: "Present", 1: "Missing"})
    fig = go.Figure(
        data=go.Heatmap(
            z=missing.T.values,
            x=missing.index.astype(str).tolist(),
            y=missing.columns.astype(str).tolist(),
            zmin=0,
            zmax=1,
            colorscale=[
                [0, "#f7fbff"],
                [0.499, "#f7fbff"],
                [0.5, "#d7301f"],
                [1, "#d7301f"],
            ],
            colorbar={"title": "Value", "tickvals": [0, 1], "ticktext": ["Present", "Missing"]},
            customdata=display.values,
            hovertemplate="Row %{x}<br>Column %{y}<br>%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Dataset row",
        yaxis_title="Column",
        height=max(320, min(900, 120 + 24 * len(missing.columns))),
    )
    _style_figure(fig)
    return fig


def build_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str | None = None,
    y_col: str | None = None,
    color_col: str | None = None,
    profile_df: pd.DataFrame | None = None,
    max_category_levels: int = 20,
    include_missing: bool = True,
    normalize: bool = False,
) -> dict[str, Any]:
    warnings: list[str] = []
    resolved_chart_type = (
        recommend_chart_type(x_col, y_col, df, profile_df)
        if chart_type == "auto"
        else chart_type
    )

    try:
        if resolved_chart_type == "missingness_bar":
            return {
                "fig": plot_missingness_bar(df),
                "chart_type": resolved_chart_type,
                "warnings": warnings,
            }

        if resolved_chart_type == "correlation_heatmap":
            return {
                "fig": plot_correlation_heatmap(df),
                "chart_type": resolved_chart_type,
                "warnings": warnings,
            }

        variable_warnings = _selected_variable_warnings(
            df,
            [x_col, y_col, color_col],
            profile_df,
            max_category_levels,
        )
        warnings.extend(variable_warnings)

        if resolved_chart_type == "histogram":
            if x_col is None or get_chart_variable_type(x_col, df, profile_df) != "numeric":
                warnings.append("Histogram requires a numeric X variable.")
                return {"fig": None, "chart_type": resolved_chart_type, "warnings": warnings}
            fig = plot_histogram(
                df,
                x_col,
                color_col=color_col,
                max_levels=max_category_levels,
                include_missing=include_missing,
            )

        elif resolved_chart_type == "bar":
            if x_col is None or get_chart_variable_type(x_col, df, profile_df) != "categorical":
                warnings.append("Bar chart requires a categorical X variable.")
                return {"fig": None, "chart_type": resolved_chart_type, "warnings": warnings}
            fig = plot_bar_chart(
                df,
                x_col,
                color_col=color_col,
                max_levels=max_category_levels,
                include_missing=include_missing,
            )

        elif resolved_chart_type == "box":
            numeric_col, category_col = _resolve_box_columns(df, x_col, y_col, profile_df)
            if numeric_col is None or category_col is None:
                warnings.append("Box plot requires one numeric variable and one categorical variable.")
                return {"fig": None, "chart_type": resolved_chart_type, "warnings": warnings}
            fig = plot_box_plot(
                df,
                numeric_col=numeric_col,
                category_col=category_col,
                color_col=color_col,
                max_levels=max_category_levels,
                include_missing=include_missing,
            )

        elif resolved_chart_type == "scatter":
            if (
                x_col is None
                or y_col is None
                or get_chart_variable_type(x_col, df, profile_df) != "numeric"
                or get_chart_variable_type(y_col, df, profile_df) != "numeric"
            ):
                warnings.append("Scatter plot requires numeric X and Y variables.")
                return {"fig": None, "chart_type": resolved_chart_type, "warnings": warnings}
            fig = plot_scatter(
                df,
                x_col,
                y_col,
                color_col=color_col,
                max_levels=max_category_levels,
                include_missing=include_missing,
            )

        elif resolved_chart_type == "stacked_bar":
            if (
                x_col is None
                or y_col is None
                or get_chart_variable_type(x_col, df, profile_df) != "categorical"
                or get_chart_variable_type(y_col, df, profile_df) != "categorical"
            ):
                warnings.append("Stacked bar chart requires categorical X and Y variables.")
                return {"fig": None, "chart_type": resolved_chart_type, "warnings": warnings}
            fig = plot_stacked_bar(
                df,
                x_col,
                y_col,
                max_levels=max_category_levels,
                normalize=normalize,
                include_missing=include_missing,
            )

        else:
            warnings.append("Select compatible variables to create a chart.")
            return {"fig": None, "chart_type": resolved_chart_type, "warnings": warnings}

    except ValueError as exc:
        warnings.append(str(exc))
        return {"fig": None, "chart_type": resolved_chart_type, "warnings": warnings}

    if include_missing is False:
        fig.update_layout(meta={"include_missing": False})

    return {"fig": fig, "chart_type": resolved_chart_type, "warnings": warnings}


def _prepare_numeric_plot_dataframe(
    df: pd.DataFrame,
    numeric_cols: list[str],
    color_col: str | None = None,
    max_levels: int = 20,
    include_missing: bool = True,
) -> pd.DataFrame:
    color_col = color_col if color_col not in numeric_cols else None
    columns = list(dict.fromkeys(numeric_cols + ([color_col] if color_col else [])))
    plot_df = normalize_missing_values(df[columns])

    for column in numeric_cols:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    drop_columns = list(numeric_cols)
    if color_col and not include_missing:
        drop_columns.append(color_col)
    plot_df = plot_df.dropna(subset=drop_columns)

    if color_col:
        color_type = get_chart_variable_type(color_col, df)
        if color_type == "numeric":
            plot_df[color_col] = pd.to_numeric(plot_df[color_col], errors="coerce")
        else:
            plot_df[color_col] = prepare_categorical_series(
                plot_df,
                color_col,
                max_levels=max_levels,
                include_missing=include_missing,
            )

    return plot_df.reset_index(drop=True)


def _prepare_categorical_plot_dataframe(
    df: pd.DataFrame,
    categorical_cols: list[str],
    max_levels: int,
    include_missing: bool,
) -> pd.DataFrame:
    selected_columns = list(dict.fromkeys(categorical_cols))
    working = normalize_missing_values(df[selected_columns])
    if not include_missing:
        working = working.dropna(subset=selected_columns)

    prepared = pd.DataFrame(index=working.index)
    for column in selected_columns:
        prepared[column] = prepare_categorical_series(
            working,
            column,
            max_levels=max_levels,
            include_missing=include_missing,
        )

    return prepared.reset_index(drop=True)


def _resolve_box_columns(
    df: pd.DataFrame,
    x_col: str | None,
    y_col: str | None,
    profile_df: pd.DataFrame | None,
) -> tuple[str | None, str | None]:
    if x_col is None or y_col is None:
        return None, None

    x_type = get_chart_variable_type(x_col, df, profile_df)
    y_type = get_chart_variable_type(y_col, df, profile_df)
    if x_type == "numeric" and y_type == "categorical":
        return x_col, y_col
    if x_type == "categorical" and y_type == "numeric":
        return y_col, x_col
    return None, None


def _numeric_chart_columns(df: pd.DataFrame) -> list[str]:
    profile = profile_dataframe(df)
    return [
        str(column)
        for column in df.columns
        if get_chart_variable_type(str(column), df, profile) == "numeric"
    ]


def _selected_variable_warnings(
    df: pd.DataFrame,
    columns: list[str | None],
    profile_df: pd.DataFrame | None,
    max_category_levels: int,
) -> list[str]:
    warnings = []
    seen: set[str] = set()

    for column in columns:
        if column is None or column not in df.columns or column in seen:
            continue

        seen.add(column)
        variable_type = get_chart_variable_type(column, df, profile_df)
        if variable_type == "id":
            warnings.append(f"{column} looks like an ID column and is not suitable for charting.")
        elif variable_type == "text":
            warnings.append(f"{column} looks like free text and is not suitable for charting.")
        elif variable_type == "categorical":
            unique_count = int(normalize_missing_values(df[[column]])[column].dropna().nunique())
            if unique_count > max_category_levels:
                warnings.append(f"{column} has many levels; rare levels were collapsed into Other.")

    return warnings


def _profile_row_for_column(
    column_name: str,
    df: pd.DataFrame,
    profile_df: pd.DataFrame | None,
) -> dict[str, Any]:
    if profile_df is not None and not profile_df.empty and "column_name" in profile_df.columns:
        matching = profile_df[profile_df["column_name"].astype(str) == str(column_name)]
        if not matching.empty:
            return matching.iloc[0].to_dict()

    return profile_dataframe(df[[column_name]]).iloc[0].to_dict()


def _looks_like_id_column(
    column_name: str,
    unique_ratio: float,
    non_missing_count: int,
    is_id_like: bool,
    numeric_parse_rate: float,
) -> bool:
    if _has_id_name_hint(column_name):
        return True

    if numeric_parse_rate >= 0.95:
        return is_id_like

    return is_id_like or (unique_ratio >= 0.95 and non_missing_count > 10)


def _has_id_name_hint(column_name: str) -> bool:
    normalized_name = column_name.lower().replace("-", "_").replace(" ", "_")
    tokens = {token for token in re.split(r"[^a-z0-9]+", column_name.lower()) if token}
    return "id" in tokens or any(
        hint in normalized_name
        for hint in ID_NAME_HINTS
        if hint != "id"
    )


def _has_categorical_name_hint(column_name: str) -> bool:
    normalized_name = column_name.lower().replace("-", "_").replace(".", "_")
    return any(hint in normalized_name for hint in CATEGORICAL_NAME_HINTS)


def _has_continuous_name_hint(column_name: str) -> bool:
    normalized_name = column_name.lower().replace("-", "_").replace(".", "_")
    return any(hint in normalized_name for hint in CONTINUOUS_NAME_HINTS)


def _looks_like_free_text(series: pd.Series) -> bool:
    text_values = series.dropna().astype(str)
    if text_values.empty:
        return False

    unique_ratio = text_values.nunique(dropna=True) / len(text_values)
    return text_values.str.len().mean() > TEXT_LENGTH_THRESHOLD or unique_ratio > 0.8


def _style_figure(fig: go.Figure) -> None:
    fig.update_layout(
        margin={"l": 48, "r": 24, "t": 64, "b": 48},
        legend_title_text="Group",
    )
