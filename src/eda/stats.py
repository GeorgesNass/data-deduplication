# -*- coding: utf-8 -*-

'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "EDA core stats utilities: null distribution, memory/dtypes tables, uniqueness metrics, and dataframe summary helpers."
'''

from __future__ import annotations

## Standard library imports
from typing import Any

## Third-party imports
import numpy as np
import pandas as pd

## Local imports
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger("eda.stats")

## ============================================================
## CONSTANTS
## ============================================================
DEFAULT_MISSING_TOKENS = {"", "nan", "NaN", "None", "none"} ## NOTE: We treat these string tokens as missing values in EDA stats

## ============================================================
## BACKWARD-COMPAT ALIASES (expected by reports.py)
## ============================================================
def build_data_types_table(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    return build_dtypes_table(df_raw=df_raw, df_clean=df_clean)

def build_missingness_table(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    return build_null_percentage_table(df_raw=df_raw, df_clean=df_clean)

def build_generic_describe_table(df: pd.DataFrame) -> pd.DataFrame:
    return build_generic_report(df=df)

## ============================================================
## MISSINGNESS HELPERS
## ============================================================
def count_missing_like(
    series: pd.Series,
    *,
    missing_tokens: set[str] | None = None,
) -> int:
    """
        Count missing-like values in a pandas Series

        Missing-like definition:
            - NaN / None
            - Empty string
            - Common string tokens (nan, NaN, none, None)

        Args:
            series: Input pandas Series
            missing_tokens: Tokens to treat as missing (case-sensitive)

        Returns:
            Number of missing-like cells
    """

    tokens = missing_tokens or DEFAULT_MISSING_TOKENS

    ## NOTE: We keep it robust even if dtype is not string
    s = series.copy()

    ## Missing by pandas definition
    is_missing = s.isna()

    ## Missing by token (string comparison)
    s_str = s.astype(str)
    is_token_missing = s_str.isin(tokens)

    return int((is_missing | is_token_missing).sum())

def get_df_null_distribution(
    df: pd.DataFrame,
    *,
    missing_tokens: set[str] | None = None,
) -> list[float]:
    """
        Compute missing-like percentage per column

        Args:
            df: Input dataframe
            missing_tokens: Tokens to treat as missing (case-sensitive)

        Returns:
            List of missing-like percentages aligned with df.columns
    """

    if df.shape[0] == 0:
        return [0.0 for _ in df.columns]

    out: list[float] = []
    for col in df.columns:
        cnt = count_missing_like(df[col], missing_tokens=missing_tokens)
        out.append((cnt / len(df)) * 100.0)

    return out

## ============================================================
## TABLE BUILDERS
## ============================================================
def build_memory_usage_table(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
        Build a table comparing memory usage per column for raw vs cleaned dataframes

        Args:
            df_raw: Raw dataframe
            df_clean: Cleaned dataframe

        Returns:
            A dataframe with memory usage comparison
    """

    s_mem_raw = df_raw.memory_usage(deep=True)
    s_mem_clean = df_clean.memory_usage(deep=True)

    df_mem_raw = pd.DataFrame(
        {
            "Column_name": s_mem_raw.index.astype(str),
            "Memory_bytes_raw": s_mem_raw.values,
        }
    )
    
    df_mem_clean = pd.DataFrame(
        {
            "Column_name": s_mem_clean.index.astype(str),
            "Memory_bytes_clean": s_mem_clean.values,
        }
    )

    df_out = df_mem_raw.merge(df_mem_clean, on="Column_name", how="left")
    
    return df_out

def build_dtypes_table(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
        Build a table comparing dtypes per column for raw vs cleaned dataframes

        Args:
            df_raw: Raw dataframe
            df_clean: Cleaned dataframe

        Returns:
            A dataframe with dtype comparison
    """

    s_dtypes_raw = df_raw.dtypes
    s_dtypes_clean = df_clean.dtypes

    df_out = pd.DataFrame(
        {
            "Column_name": s_dtypes_raw.index.astype(str),
            "Data_types_raw": s_dtypes_raw.astype(str).values,
        }
    )
    df_out["Data_types_clean"] = (
        s_dtypes_clean.reindex(s_dtypes_raw.index).astype(str).values
    )

    return df_out

def build_unique_values_table(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
        Build a table comparing number of unique values per column

        Args:
            df_raw: Raw dataframe
            df_clean: Cleaned dataframe

        Returns:
            A dataframe with nunique comparison
    """

    ## NOTE: We treat empty-like as NaN for uniqueness computation to avoid counting empty strings
    s_nunique_raw = (
        df_raw.astype(str)
        .replace({"": np.nan, "NaN": np.nan, "nan": np.nan})
        .nunique(dropna=True)
    )
    s_nunique_clean = (
        df_clean.astype(str)
        .replace({"": np.nan, "NaN": np.nan, "nan": np.nan})
        .nunique(dropna=True)
    )

    df_out = pd.DataFrame(
        {
            "Column_name": s_nunique_raw.index.astype(str),
            "Unique_values_raw": s_nunique_raw.values,
            "Unique_values_clean": s_nunique_clean.reindex(s_nunique_raw.index).values,
        }
    )

    return df_out

def build_null_percentage_table(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    *,
    missing_tokens: set[str] | None = None,
) -> pd.DataFrame:
    """
        Build a table comparing missing-like percentages per column

        Args:
            df_raw: Raw dataframe
            df_clean: Cleaned dataframe
            missing_tokens: Tokens to treat as missing (case-sensitive)

        Returns:
            A dataframe with missing-like percentage comparison
    """

    raw_pct = get_df_null_distribution(df_raw, missing_tokens=missing_tokens)
    clean_pct = get_df_null_distribution(df_clean, missing_tokens=missing_tokens)

    df_out = pd.DataFrame(
        {
            "Column_name": df_raw.columns.astype(str).to_list(),
            "Null_values_raw": raw_pct,
            "Null_values_clean": clean_pct,
        }
    )

    return df_out

## ============================================================
## SUMMARY HELPERS
## ============================================================
def build_generic_report(df: pd.DataFrame) -> pd.DataFrame:
    """
        Build a generic descriptive report using pandas describe

        Args:
            df: Input dataframe

        Returns:
            Transposed describe(include='all') dataframe
    """

    if df.shape[0] == 0:
        return pd.DataFrame()

    try:
        return df.describe(include="all").T
    except Exception as exc:
        LOGGER.warning("Failed to build generic report: %s", exc)
        return pd.DataFrame()

def print_df_statistics(df: pd.DataFrame, message: str) -> None:
    """
        Print a dataframe with a header message

        Args:
            df: Dataframe to print
            message: Title message
    """

    print(f"\n--------------------------- {message} ---------------------------\n")
    print(df)

def statistics_tables(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    *,
    missing_tokens: set[str] | None = None,
    sort_nulls_desc: bool = True,
) -> dict[str, pd.DataFrame]:
    """
        Compute core EDA tables for raw vs cleaned dataframes

        High-level workflow:
            1) Memory usage comparison
            2) Dtypes comparison
            3) Unique values comparison
            4) Missing-like percentage comparison
            5) Generic report on cleaned dataframe

        Args:
            df_raw: Raw dataframe
            df_clean: Cleaned dataframe
            missing_tokens: Tokens to treat as missing (case-sensitive)
            sort_nulls_desc: If True, sorts nulls table by cleaned null percentage desc

        Returns:
            Dict of named dataframes
    """

    ## NOTE: We keep this function pure (no file I/O), it only returns dataframes
    mem_table = build_memory_usage_table(df_raw, df_clean)
    dtype_table = build_dtypes_table(df_raw, df_clean)
    unique_table = build_unique_values_table(df_raw, df_clean)
    null_table = build_null_percentage_table(df_raw, df_clean, missing_tokens=missing_tokens)
    generic_report = build_generic_report(df_clean)

    if sort_nulls_desc and not null_table.empty:
        null_table = null_table.sort_values(by="Null_values_clean", ascending=False).reset_index(drop=True)

    return {
        "memory_usage": mem_table,
        "dtypes": dtype_table,
        "unique_values": unique_table,
        "null_percentages": null_table,
        "generic_report": generic_report,
    }