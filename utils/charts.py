# ============================================================
# utils/charts.py
# ============================================================
# PURPOSE: Generates all visualizations for the app.
# Uses matplotlib and plotly — no AI logic here, just charting.
# Each function returns a Plotly figure that Streamlit can render.
# ============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from utils.config import Config


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing count of missing values per column.
    Only shows columns that have at least 1 missing value.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="✅ No missing values found!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="green")
        )
        fig.update_layout(title="Missing Values Analysis")
        return fig

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x": "Column", "y": "Missing Count"},
        title="Missing Values per Column",
        color=missing.values,
        color_continuous_scale="Reds",
        text=missing.values
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_histograms(df: pd.DataFrame) -> list:
    """
    Generates a histogram for each numeric column.
    Returns a list of Plotly figures.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    figures = []

    for col in numeric_cols:
        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"Distribution of '{col}'",
            marginal="box",        # Adds a box plot on top — looks professional
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        figures.append((col, fig))

    return figures


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Correlation matrix heatmap for numeric columns.
    Correlation shows how strongly two variables move together.
    Value of 1 = perfect positive correlation, -1 = perfect negative.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # Limit columns to avoid a huge unreadable chart
    if numeric_df.shape[1] > Config.MAX_HEATMAP_COLS:
        numeric_df = numeric_df.iloc[:, : Config.MAX_HEATMAP_COLS]

    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 numeric columns for correlation.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    corr = numeric_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap (Numeric Columns)",
        aspect="auto"
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_bar_charts(df: pd.DataFrame) -> list:
    """
    Bar charts showing value counts for categorical columns.
    Only shows top N categories to avoid clutter.
    Returns a list of (column_name, figure) tuples.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    figures = []

    for col in cat_cols:
        # Skip columns with too many unique values (not useful to visualize)
        if df[col].nunique() > Config.MAX_CATEGORIES:
            continue

        value_counts = df[col].value_counts().head(Config.MAX_CATEGORIES)

        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={"x": col, "y": "Count"},
            title=f"Value Counts for '{col}'",
            color=value_counts.values,
            color_continuous_scale="Blues",
            text=value_counts.values
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        figures.append((col, fig))

    return figures


def plot_scatter_matrix(df: pd.DataFrame) -> go.Figure:
    """
    Scatter matrix (pair plot) for numeric columns.
    Helps spot relationships between multiple variables at once.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Limit to 5 columns max for readability
    cols_to_use = numeric_cols[:5]

    if len(cols_to_use) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 numeric columns for scatter matrix.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Find a categorical column to use as color (optional)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    color_col = cat_cols[0] if cat_cols else None

    fig = px.scatter_matrix(
        df[cols_to_use + ([color_col] if color_col else [])].dropna(),
        dimensions=cols_to_use,
        color=color_col,
        title="Scatter Matrix (Relationships Between Numeric Columns)",
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.6))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
