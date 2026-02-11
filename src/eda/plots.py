# -*- coding: utf-8 -*-

'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "EDA plotting utilities: bar charts, pie charts, histograms and visual distributions."
'''

from __future__ import annotations

## Standard library imports
from pathlib import Path
from typing import Any

## Third-party imports
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot

## ============================================================
## GENERIC PLOT BUILDERS
## ============================================================
def _add_bar_trace(
    fig: go.Figure,
    x_values: list[Any],
    y_values: list[Any],
    title: str,
    row: int,
    col: int,
) -> go.Figure:
    """
        Add a bar trace to a subplot figure

        Args:
            fig: Plotly figure
            x_values: X-axis values
            y_values: Y-axis values
            title: Plot title
            row: Subplot row index
            col: Subplot column index

        Returns:
            Updated figure
    """
    
    fig.add_trace(
        go.Bar(x=x_values, y=y_values, name=title),
        row=row,
        col=col,
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
        }
    )

    return fig

def _add_pie_trace(
    fig: go.Figure,
    labels: list[Any],
    values: list[Any],
    title: str,
    row: int,
    col: int,
) -> go.Figure:
    """
        Add a pie trace to a subplot figure

        Args:
            fig: Plotly figure
            labels: Pie labels
            values: Pie values
            title: Plot title
            row: Subplot row index
            col: Subplot column index

        Returns:
            Updated figure
    """
    
    fig.add_trace(
        go.Pie(labels=labels, values=values, name=title),
        row=row,
        col=col,
    )

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
        }
    )

    return fig

## ============================================================
## BAR & PIE DISTRIBUTIONS
## ============================================================
def plot_value_counts_bar(
    df: pd.DataFrame,
    column: str,
    threshold: int = 10,
    output_path: str | Path | None = None,
) -> go.Figure:
    """
        Plot value count distribution as a bar chart

        Args:
            df: Input dataframe
            column: Target column
            threshold: Minimum count threshold
            output_path: Optional HTML export path

        Returns:
            Plotly figure
    """
    
    counts = df[column].value_counts()
    df_plot = pd.DataFrame(
        {"value": counts.index, "count": counts.values}
    )

    ## Aggregate small categories
    small_sum = df_plot[df_plot["count"] < threshold]["count"].sum()
    df_plot = df_plot[df_plot["count"] >= threshold]

    if small_sum > 0:
        df_plot.loc[len(df_plot)] = ["all_other", small_sum]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "bar"}]])
    fig = _add_bar_trace(
        fig,
        df_plot["value"].tolist(),
        df_plot["count"].tolist(),
        f"{column} distribution",
        1,
        1,
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig

def plot_null_percentage_pie(
    df: pd.DataFrame,
    target_column: str,
    group_by_column: str,
    threshold: float = 5.0,
    output_path: str | Path | None = None,
) -> go.Figure:
    """
        Plot percentage of empty values grouped by another column

        Args:
            df: Input dataframe
            target_column: Column to analyze
            group_by_column: Grouping column
            threshold: Minimum percentage threshold
            output_path: Optional HTML export path

        Returns:
            Plotly figure
    """
    
    total_empty = (
        df[target_column].isin(["", "NaN", "nan", np.nan]).sum()
    )

    grouped = (
        df[target_column]
        .isin(["", "NaN", "nan", np.nan])
        .groupby(df[group_by_column])
        .sum()
    )

    percentages = round((grouped / max(total_empty, 1)) * 100, 2)

    df_plot = pd.DataFrame(
        {"value": percentages.index, "percentage": percentages.values}
    )

    df_plot = df_plot[df_plot["percentage"] > threshold]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "pie"}]])
    fig = _add_pie_trace(
        fig,
        df_plot["value"].tolist(),
        df_plot["percentage"].tolist(),
        f"{target_column} empty %",
        1,
        1,
    )

    if output_path:
        fig.write_html(str(output_path))

    return fig

## ============================================================
## HISTOGRAMS
## ============================================================
def plot_age_histogram(
    df: pd.DataFrame,
    birth_date_column: str = "birth_date",
    output_path: str | Path | None = None,
) -> go.Figure:
    """
        Plot age distribution histogram

        Args:
            df: Input dataframe
            birth_date_column: Birth date column name
            output_path: Optional HTML export path

        Returns:
            Plotly figure
    """
    
    df_tmp = df[df[birth_date_column] != ""].copy()
    df_tmp = df_tmp.dropna(subset=[birth_date_column])

    df_tmp["Age"] = round(
        (dt.datetime.today() - pd.to_datetime(df_tmp[birth_date_column]))
        .dt.days
        / 365,
        0,
    )

    fig = px.histogram(df_tmp, x="Age")

    if output_path:
        fig.write_html(str(output_path))

    return fig

## ============================================================
## CLUSTER VISUALIZATION
## ============================================================
def plot_cluster_confidence(
    df: pd.DataFrame,
    confidence_column: str = "confidence_score",
    output_path: str | Path | None = None,
) -> go.Figure:
    """
        Plot histogram of cluster confidence scores

        Args:
            df: Clustered dataframe
            confidence_column: Confidence score column
            output_path: Optional HTML export path

        Returns:
            Plotly figure
    """
    
    fig = px.histogram(df, x=confidence_column)

    if output_path:
        fig.write_html(str(output_path))

    return fig
    
## ============================================================
## HIGH-LEVEL EDA PLOTS
## ============================================================
def statistics_plots(
    df: pd.DataFrame,
    *,
    output_dir: str | Path = ".",
    open_in_browser: bool = True,
) -> None:
    """
        Generate core EDA plots for a dataframe

        High-level workflow:
            1) Age distribution from birth_date
            2) City distribution (bar)
            3) Sex distribution (bar)
            4) Missingness by origin for selected columns (pie)

        Args:
            df: Input dataframe (expected to be already cleaned)
            output_dir: Folder where HTML plots are written
            open_in_browser: If True, opens plots in a browser
    """

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ## 1) Age histogram
    if "birth_date" in df.columns:
        fig = plot_age_histogram(df, birth_date_column="birth_date", output_path=out_dir / "histogram_age_distribution.html")
        if open_in_browser:
            plot(fig)

    ## 2) City bar
    if "addresses_city" in df.columns:
        fig = plot_value_counts_bar(df, "addresses_city", threshold=10, output_path=out_dir / "city_distribution_count.html")
        if open_in_browser:
            plot(fig)

    ## 3) Sex bar
    if "sex" in df.columns:
        fig = plot_value_counts_bar(df, "sex", threshold=10, output_path=out_dir / "sex_distribution_count.html")
        if open_in_browser:
            plot(fig)

    ## 4) Missingness pies by origin
    if "origin" in df.columns:
        for col, fname in [
            ("sex", "sex_empty_vals_percentage.html"),
            ("social_security_number", "ssn_empty_vals_percentage.html"),
            ("addresses_city", "city_empty_vals_percentage.html"),
            ("telephones", "phones_vals_percentage.html"),
            ("emails", "emails_vals_percentage.html"),
        ]:
            if col not in df.columns:
                continue

            fig = plot_null_percentage_pie(
                df,
                target_column=col,
                group_by_column="origin",
                threshold=2.0,
                output_path=out_dir / fname,
            )
            if open_in_browser:
                plot(fig)