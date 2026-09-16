# ============================================================
# agents/analysis_agent.py
# ============================================================
# PURPOSE: The "Analysis Agent" — performs actual EDA (Exploratory
# Data Analysis) using pandas, then asks the LLM to interpret results.
#
# WHAT IT DOES:
#   1. Uses pandas to compute real statistics (no hallucination!)
#   2. Sends those statistics to the LLM for interpretation
#   3. Returns both raw stats and AI interpretation
#
# KEY INSIGHT for interviews:
#   We do NOT let the LLM do the math. Pandas does the math.
#   The LLM only interprets what pandas calculated.
#   This avoids "LLM hallucination" on numbers.
#
# INTERVIEW TIP:
#   "The Analysis Agent follows a tool-augmented pattern — it uses
#    pandas as a reliable computation tool, then uses the LLM only
#    for language tasks like explaining the results in plain English."
# ============================================================

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from agents.llm_client import get_llm


def run_analysis_agent(df: pd.DataFrame, dataset_context: str) -> dict:
    """
    Analysis Agent: Performs EDA and generates AI-powered interpretation.

    Args:
        df: The uploaded DataFrame
        dataset_context: Pre-built text summary of the dataset

    Returns:
        dict with keys:
            - 'stats': raw computed statistics (dict)
            - 'cleaning_suggestions': list of cleaning suggestions
            - 'analysis_text': LLM-generated interpretation of the data
    """

    # ── Step 1: Compute real statistics using pandas ───────────
    # This is pure Python — no AI involved here.
    stats = _compute_statistics(df)

    # ── Step 2: Generate cleaning suggestions ─────────────────
    cleaning_suggestions = _generate_cleaning_suggestions(df, stats)

    # ── Step 3: Ask LLM to interpret the statistics ───────────
    analysis_text = _interpret_with_llm(df, dataset_context, stats, cleaning_suggestions)

    return {
        "stats": stats,
        "cleaning_suggestions": cleaning_suggestions,
        "analysis_text": analysis_text
    }


def _compute_statistics(df: pd.DataFrame) -> dict:
    """
    Computes all important EDA statistics using pandas.
    Returns a structured dictionary of results.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(include=["object", "category"])

    stats = {
        "shape": df.shape,
        "total_cells": df.size,
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().mean().mean() * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_col_count": len(numeric_df.columns),
        "categorical_col_count": len(cat_df.columns),
    }

    # Numeric statistics
    if not numeric_df.empty:
        desc = numeric_df.describe()
        stats["numeric_stats"] = desc.to_dict()

        # Find highly skewed columns (skewness > 1 means right-skewed)
        skewness = numeric_df.skew().round(2)
        stats["skewed_columns"] = skewness[abs(skewness) > 1].to_dict()

        # Find potential outlier columns using IQR method
        outlier_cols = []
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((numeric_df[col] < Q1 - 1.5 * IQR) |
                             (numeric_df[col] > Q3 + 1.5 * IQR)).sum()
            if outlier_count > 0:
                outlier_cols.append(f"{col} ({int(outlier_count)} outliers)")
        stats["outlier_columns"] = outlier_cols

        # Correlation summary (strongest correlations)
        if len(numeric_df.columns) >= 2:
            corr = numeric_df.corr().abs()
            # Get upper triangle only (avoid duplicates)
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            top_corr = upper.stack().sort_values(ascending=False).head(5)
            stats["top_correlations"] = {
                f"{idx[0]} vs {idx[1]}": round(val, 3)
                for idx, val in top_corr.items()
            }

    # Categorical statistics
    if not cat_df.empty:
        cat_summary = {}
        for col in cat_df.columns:
            cat_summary[col] = {
                "unique_values": int(df[col].nunique()),
                "top_value": str(df[col].mode().iloc[0]) if not df[col].mode().empty else "N/A",
                "top_value_count": int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0
            }
        stats["categorical_stats"] = cat_summary

    return stats


def _generate_cleaning_suggestions(df: pd.DataFrame, stats: dict) -> list:
    """
    Generates rule-based data cleaning suggestions.
    These are deterministic (no AI needed for this part).

    Returns a list of suggestion strings.
    """
    suggestions = []

    # ── Check for missing values ───────────────────────────────
    missing_per_col = df.isnull().sum()
    for col, count in missing_per_col.items():
        if count == 0:
            continue
        pct = round((count / len(df)) * 100, 1)

        if pct > 50:
            suggestions.append(
                f"🔴 Column '{col}' has {pct}% missing data — consider DROPPING this column."
            )
        elif pct > 20:
            suggestions.append(
                f"🟠 Column '{col}' has {pct}% missing data — consider imputation "
                f"(median for numeric, mode for categorical)."
            )
        else:
            dtype = str(df[col].dtype)
            if "float" in dtype or "int" in dtype:
                suggestions.append(
                    f"🟡 Column '{col}' has {pct}% missing values — fill with median."
                )
            else:
                suggestions.append(
                    f"🟡 Column '{col}' has {pct}% missing values — fill with mode or 'Unknown'."
                )

    # ── Check for duplicate rows ───────────────────────────────
    if stats["duplicate_rows"] > 0:
        suggestions.append(
            f"🔴 Found {stats['duplicate_rows']} duplicate rows — run df.drop_duplicates() to remove them."
        )

    # ── Check for skewed numeric columns ──────────────────────
    if "skewed_columns" in stats and stats["skewed_columns"]:
        for col, skew in stats["skewed_columns"].items():
            suggestions.append(
                f"📊 Column '{col}' is highly skewed (skewness={skew}) — "
                f"consider log transformation to normalize distribution."
            )

    # ── Check for outliers ────────────────────────────────────
    if "outlier_columns" in stats and stats["outlier_columns"]:
        for col_info in stats["outlier_columns"]:
            suggestions.append(
                f"⚠️  Outliers detected in '{col_info}' — "
                f"investigate using box plots and consider capping (winsorization)."
            )

    # ── Check for high-cardinality categorical columns ────────
    cat_cols = df.select_dtypes(include=["object", "category"])
    for col in cat_cols.columns:
        if df[col].nunique() / len(df) > 0.5 and len(df) > 100:
            suggestions.append(
                f"🔵 Column '{col}' has very high cardinality ({df[col].nunique()} unique values) "
                f"— may need encoding strategy or consider dropping."
            )

    if not suggestions:
        suggestions.append("✅ Dataset looks clean! No major cleaning issues detected.")

    return suggestions


def _interpret_with_llm(df: pd.DataFrame, dataset_context: str,
                        stats: dict, cleaning_suggestions: list) -> str:
    """
    Uses the LLM to interpret the computed statistics in plain English.
    The LLM doesn't compute anything — it just explains what pandas found.
    """
    llm = get_llm(temperature=0.4)

    # Format cleaning suggestions into a string for the prompt
    cleaning_str = "\n".join(cleaning_suggestions[:5])  # Limit to avoid token overflow

    # Format key stats for the LLM
    corr_str = ""
    if "top_correlations" in stats:
        corr_str = "\n".join([f"  - {k}: {v}" for k, v in stats["top_correlations"].items()])

    outlier_str = ""
    if "outlier_columns" in stats:
        outlier_str = ", ".join(stats["outlier_columns"][:5])

    system_prompt = """You are an expert data analyst providing insights to a business team.

Your job is to explain data analysis results in clear, non-technical language.
Focus on:
- What the data tells us about the business
- Key patterns and trends
- Potential risks or opportunities
- Actionable recommendations

Use bullet points for clarity. Be concise but insightful.
"""

    human_prompt = f"""
Analyze this dataset and provide business insights:

{dataset_context}

COMPUTED STATISTICS:
- Missing data: {stats['missing_pct']}% of all cells
- Duplicate rows: {stats['duplicate_rows']}
- Numeric columns: {stats['numeric_col_count']}
- Categorical columns: {stats['categorical_col_count']}

TOP CORRELATIONS FOUND:
{corr_str if corr_str else 'No strong correlations found'}

OUTLIER COLUMNS:
{outlier_str if outlier_str else 'No significant outliers'}

CLEANING RECOMMENDATIONS:
{cleaning_str}

Please provide:
1. A 2-sentence executive summary of what this dataset represents
2. 3-5 key statistical observations
3. 2-3 potential business insights
4. Overall data quality assessment (1-10 score with brief explanation)
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm.invoke(messages)
    return response.content
