# -*- coding: utf-8 -*-

'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "EDA reporting helpers: high-level orchestration, summary tables, and optional Plotly HTML exports."
'''

from __future__ import annotations

## Standard library imports
from pathlib import Path
from typing import Any

## Third-party imports
import pandas as pd

## Local imports
from src.eda.plots import statistics_plots
from src.eda.stats import (
    build_data_types_table,
    build_memory_usage_table,
    build_missingness_table,
    build_unique_values_table,
    build_generic_describe_table,
)
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger("eda.reports")

## ============================================================
## DISPLAY HELPERS
## ============================================================
def print_df_statistics(df: pd.DataFrame, message: str) -> None:
    """
        Print a dataframe with a formatted header

        Args:
            df: Dataframe to print
            message: Header message
    """
    
    print(f"\n--------------------------- {message} ---------------------------\n")
    print(df)

## ============================================================
## HIGH-LEVEL TABLE REPORTS
## ============================================================
def statistics_tables(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    *,
    print_tables: bool = True,
) -> dict[str, pd.DataFrame]:
    """
        Build core EDA summary tables comparing raw vs cleaned dataframes

        High-level workflow:
            1) Memory usage per column
            2) Data types per column
            3) Unique values per column
            4) Missingness percentage per column
            5) Generic describe table (clean)

        Args:
            df_raw: Raw dataframe
            df_clean: Clean dataframe
            print_tables: If True, prints each table to stdout

        Returns:
            Dict of named dataframes
    """

    ## Build tables
    df_mem = build_memory_usage_table(df_raw=df_raw, df_clean=df_clean)
    df_dtypes = build_data_types_table(df_raw=df_raw, df_clean=df_clean)
    df_unique = build_unique_values_table(df_raw=df_raw, df_clean=df_clean)
    df_missing = build_missingness_table(df_raw=df_raw, df_clean=df_clean)
    df_describe = build_generic_describe_table(df=df_clean)

    tables = {
        "memory_usage": df_mem,
        "data_types": df_dtypes,
        "unique_values": df_unique,
        "missingness": df_missing,
        "describe": df_describe,
    }

    ## Optional printing
    if print_tables:
        print_df_statistics(df_mem, "Memory usage per variable")
        print_df_statistics(df_dtypes, "Data types")
        print_df_statistics(df_unique, "Unique values per variable")
        print_df_statistics(df_missing.sort_values(by="Null_values_clean", ascending=False), "Null values percentage")
        print("\n--------------------------- Generic report (clean dataframe) ---------------------------\n")
        print(df_describe)

    return tables

## ============================================================
## HIGH-LEVEL EDA ORCHESTRATION
## ============================================================
def run_eda_analysis(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    *,
    output_dir: str | Path = ".",
    enable_plots: bool = True,
    open_in_browser: bool = True,
    enable_tables: bool = True,
    print_tables: bool = True,
) -> dict[str, Any]:
    """
        Run EDA analysis on a dataset

        High-level workflow:
            1) Build summary tables (optional)
            2) Generate Plotly HTML plots (optional)

        Args:
            df_raw: Raw dataframe
            df_clean: Clean dataframe
            output_dir: Output folder for HTML plots
            enable_plots: If True, generate Plotly HTML plots
            open_in_browser: If True, opens plots in a browser
            enable_tables: If True, compute table reports
            print_tables: If True, prints tables to stdout

        Returns:
            Dict containing optional "tables" and "plots" metadata
    """

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("EDA analysis: start")

    result: dict[str, Any] = {"meta": {"output_dir": str(out_dir)}}

    ## Tables
    if enable_tables:
        try:
            result["tables"] = statistics_tables(df_raw=df_raw, df_clean=df_clean, print_tables=print_tables)
        except Exception as exc:
            LOGGER.warning("EDA tables failed: %s", exc)
            result["tables_error"] = str(exc)

    ## Plots
    if enable_plots:
        try:
            statistics_plots(df_clean, output_dir=out_dir, open_in_browser=open_in_browser)
            result["plots"] = {"enabled": True}
        except Exception as exc:
            LOGGER.warning("EDA plots failed: %s", exc)
            result["plots"] = {"enabled": False, "error": str(exc)}

    LOGGER.info("EDA analysis: completed")
    
    return result