import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


def plot_km_curve(
    curve_df: pd.DataFrame,
    title: str = "Kaplan-Meier Survival Curve",
    time_unit: str = "unknown",
    show_ci: bool = True,
    show_censors: bool = True,
) -> go.Figure:
    fig = go.Figure()

    if curve_df.empty:
        _style_figure(fig, title, time_unit)
        return fig

    for index, (group, group_df) in enumerate(curve_df.groupby("group", sort=False)):
        color = qualitative.Plotly[index % len(qualitative.Plotly)]
        ordered_group_df = group_df.sort_values("time")

        if show_ci:
            fig.add_trace(
                go.Scatter(
                    x=ordered_group_df["time"].tolist()
                    + ordered_group_df["time"].iloc[::-1].tolist(),
                    y=ordered_group_df["ci_upper"].tolist()
                    + ordered_group_df["ci_lower"].iloc[::-1].tolist(),
                    fill="toself",
                    fillcolor=_hex_to_rgba(color, 0.18),
                    line={"color": "rgba(255,255,255,0)"},
                    hoverinfo="skip",
                    name=f"{group} CI",
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=ordered_group_df["time"],
                y=ordered_group_df["survival"],
                mode="lines",
                line={"color": color, "shape": "hv", "width": 2},
                name=str(group),
                hovertemplate=(
                    "Time: %{x}<br>"
                    "Survival probability: %{y:.3f}"
                    f"<extra>{group}</extra>"
                ),
            )
        )

        if show_censors and "censored" in ordered_group_df:
            censored = ordered_group_df[ordered_group_df["censored"] > 0]
            if not censored.empty:
                fig.add_trace(
                    go.Scatter(
                        x=censored["time"],
                        y=censored["survival"],
                        mode="markers",
                        marker={"color": color, "size": 10, "symbol": "line-ns-open"},
                        name=f"{group} censored",
                        showlegend=False,
                        customdata=censored["censored"],
                        hovertemplate=(
                            "Censored at %{x}: %{customdata}<br>"
                            "Survival probability: %{y:.3f}"
                            f"<extra>{group}</extra>"
                        ),
                    )
                )

    _style_figure(fig, title, time_unit)
    return fig


def _style_figure(fig: go.Figure, title: str, time_unit: str) -> None:
    x_axis_title = "Time" if time_unit == "unknown" else f"Time ({time_unit})"
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title="Survival probability",
        hovermode="x unified",
        legend_title_text="Group",
        margin={"l": 48, "r": 24, "t": 64, "b": 48},
    )
    fig.update_yaxes(range=[0, 1.05])


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    stripped = hex_color.lstrip("#")
    red = int(stripped[0:2], 16)
    green = int(stripped[2:4], 16)
    blue = int(stripped[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"
