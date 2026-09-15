# ============================================================
# utils/data_loader.py
# ============================================================
# PURPOSE: Handles loading and basic inspection of uploaded CSV files.
# This is a "utility" — it does not contain AI logic, just
# pure data manipulation with pandas.
# ============================================================

import pandas as pd
import numpy as np
from io import BytesIO


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV file from a Streamlit UploadedFile object.

    Args:
        uploaded_file: Streamlit file object from st.file_uploader()

    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Read the uploaded file bytes directly into pandas
    df = pd.read_csv(uploaded_file)
    return df


def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Compute basic dataset statistics that the AI agents will use.

    Returns a dictionary with:
    - shape: (rows, columns)
    - dtypes: column data types
    - missing values count per column
    - duplicate row count
    - numeric columns list
    - categorical columns list
    - sample data (first 5 rows as string)
    """

    # Separate columns by data type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Count missing values per column (only columns that have at least 1 missing)
    missing_counts = df.isnull().sum()
    missing_dict = missing_counts[missing_counts > 0].to_dict()

    # Missing percentages (rounded to 2 decimal places)
    missing_pct = {
        col: round((count / len(df)) * 100, 2)
        for col, count in missing_dict.items()
    }

    return {
        "shape": df.shape,                             # (rows, cols)
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),     # {col: dtype}
        "missing_counts": missing_dict,                 # {col: count}
        "missing_pct": missing_pct,                     # {col: pct}
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


def get_numeric_summary(df: pd.DataFrame) -> str:
    """
    Returns a string summary of numeric columns using pandas describe().
    This will be injected into LLM prompts so the AI has real statistics.
    """
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        return "No numeric columns found."

    # describe() gives count, mean, std, min, 25%, 50%, 75%, max
    summary = numeric_cols.describe().round(2)
    return summary.to_string()


def get_categorical_summary(df: pd.DataFrame) -> str:
    """
    Returns a string summary of categorical columns.
    Shows top value counts for each categorical column.
    """
    cat_cols = df.select_dtypes(include=["object", "category"])
    if cat_cols.empty:
        return "No categorical columns found."

    lines = []
    for col in cat_cols.columns:
        # Top 5 most frequent values
        top_values = df[col].value_counts().head(5)
        lines.append(f"\nColumn: '{col}' (unique: {df[col].nunique()})")
        lines.append(top_values.to_string())

    return "\n".join(lines)


def prepare_llm_context(df: pd.DataFrame, info: dict) -> str:
    """
    Builds a compact text representation of the dataset for LLM prompts.
    We can't send the whole DataFrame to an LLM — this summarizes it.

    This is the "context" that gets injected into every agent's prompt.
    """
    context = f"""
DATASET OVERVIEW:
- Total Rows: {info['rows']}
- Total Columns: {info['columns']}
- Column Names: {', '.join(info['column_names'])}
- Numeric Columns: {', '.join(info['numeric_cols']) or 'None'}
- Categorical Columns: {', '.join(info['categorical_cols']) or 'None'}
- Missing Values: {info['missing_counts'] if info['missing_counts'] else 'None'}
- Duplicate Rows: {info['duplicate_rows']}
- Memory Usage: {info['memory_usage_kb']} KB

NUMERIC STATISTICS:
{get_numeric_summary(df)}

CATEGORICAL STATISTICS:
{get_categorical_summary(df)}

SAMPLE DATA (first 5 rows):
{df.head(5).to_string(index=False)}
"""
    return context.strip()
