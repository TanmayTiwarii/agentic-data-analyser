# ============================================================
# agents/visualization_agent.py
# ============================================================
# PURPOSE: The "Visualization Agent" — decides which charts to
# generate based on the dataset structure, then creates them.
#
# WHAT IT DOES:
#   1. Inspects the dataset (how many numeric/categorical cols?)
#   2. Decides which chart types are appropriate
#   3. Generates those charts by calling utils/charts.py
#   4. Optionally uses LLM to add a text caption for each chart
#
# INTERVIEW TIP:
#   "The Visualization Agent uses a rule-based decision layer to
#    select appropriate chart types, then delegates rendering to
#    the charts utility module. This separation of concerns keeps
#    the agent logic clean and the charting logic reusable."
# ============================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage

from utils import charts as chart_module
from agents.llm_client import get_llm


def run_visualization_agent(df: pd.DataFrame) -> dict:
    """
    Visualization Agent: Selects and generates appropriate charts.

    Args:
        df: The uploaded DataFrame

    Returns:
        dict with keys:
            - 'charts': list of (title, figure, description) tuples
            - 'chart_plan': what charts were decided upon and why
    """

    # ── Step 1: Decide what charts to generate ────────────────
    chart_plan = _decide_chart_plan(df)

    # ── Step 2: Generate the charts ───────────────────────────
    charts_output = _generate_charts(df, chart_plan)

    return {
        "charts": charts_output,
        "chart_plan": chart_plan
    }


def _decide_chart_plan(df: pd.DataFrame) -> dict:
    """
    Rule-based decision logic for which charts to generate.
    This is deterministic (no LLM needed) — just checks data types.

    Returns a dict describing the plan.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    plan = {
        "generate_missing_chart": True,   # Always show missing values chart
        "generate_histograms": len(numeric_cols) > 0,
        "generate_correlation": len(numeric_cols) >= 2,
        "generate_bar_charts": len(cat_cols) > 0,
        "generate_scatter_matrix": len(numeric_cols) >= 2,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "notes": []
    }

    # Add notes explaining decisions (great for transparency)
    if len(numeric_cols) == 0:
        plan["notes"].append("No numeric columns — skipping histogram and correlation charts.")
    if len(numeric_cols) < 2:
        plan["notes"].append("Less than 2 numeric columns — skipping correlation heatmap.")
    if len(cat_cols) == 0:
        plan["notes"].append("No categorical columns — skipping bar charts.")

    return plan


def _generate_charts(df: pd.DataFrame, plan: dict) -> List[Tuple[str, go.Figure, str]]:
    """
    Generates charts based on the plan.

    Returns a list of tuples: (chart_title, plotly_figure, description)
    """
    charts = []

    # ── Missing Values Chart ───────────────────────────────────
    if plan["generate_missing_chart"]:
        fig = chart_module.plot_missing_values(df)
        charts.append((
            "Missing Values Analysis",
            fig,
            "Shows which columns have missing data and how much. "
            "Red bars indicate more missing values — these columns need attention."
        ))

    # ── Histograms ────────────────────────────────────────────
    if plan["generate_histograms"]:
        hist_figs = chart_module.plot_histograms(df)
        for col_name, fig in hist_figs:
            charts.append((
                f"Distribution: {col_name}",
                fig,
                f"Histogram showing the frequency distribution of '{col_name}'. "
                "The box plot on top shows median, quartiles, and outliers."
            ))

    # ── Correlation Heatmap ───────────────────────────────────
    if plan["generate_correlation"]:
        fig = chart_module.plot_correlation_heatmap(df)
        charts.append((
            "Correlation Heatmap",
            fig,
            "Shows correlation between all numeric columns. "
            "Values close to 1 (dark red) = strong positive relationship. "
            "Values close to -1 (dark blue) = strong inverse relationship."
        ))

    # ── Bar Charts for Categorical Columns ────────────────────
    if plan["generate_bar_charts"]:
        bar_figs = chart_module.plot_bar_charts(df)
        for col_name, fig in bar_figs:
            charts.append((
                f"Value Counts: {col_name}",
                fig,
                f"Bar chart showing frequency of each category in '{col_name}'. "
                "Taller bars = more frequent categories."
            ))

    # ── Scatter Matrix ────────────────────────────────────────
    if plan["generate_scatter_matrix"] and len(plan["numeric_cols"]) >= 2:
        fig = chart_module.plot_scatter_matrix(df)
        charts.append((
            "Scatter Matrix",
            fig,
            "Shows pairwise relationships between all numeric columns. "
            "Look for diagonal patterns (positive correlation) or "
            "horizontal patterns (no correlation)."
        ))

    return charts
