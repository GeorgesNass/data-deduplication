'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Candidate generation helpers used for deduplication: full-name and address candidate lists."
'''

from __future__ import annotations

## Standard library imports
import itertools
from typing import Any, Iterable

## Third-party imports
import pandas as pd

## Local imports
from src.model.cleaning import ensure_columns, normalize_text

## ============================================================
## CANDIDATE GENERATION HELPERS
## ============================================================
def unique_non_empty(values: Iterable[Any]) -> list[str]:
    """
        Normalize values and keep unique non-empty tokens

        Args:
            values: Iterable of values

        Returns:
            Unique normalized strings (stable order)
    """
    
    out: list[str] = []
    seen = set()

    for v in values:
        token = normalize_text(v)
        if token and token not in seen:
            seen.add(token)
            out.append(token)

    return out

def row_full_name_candidates(row: pd.Series) -> list[str]:
    """
        Build full name candidates for a single row

        Fields used:
            - birth_family_name, usage_family_name, birth_family_name_list
            - birth_first_name, usage_first_name, birth_first_name_list

        Args:
            row: Row of a dataframe

        Returns:
            List of candidate full names (stable unique)
    """
    
    ## Collect candidate tokens
    fam_candidates = unique_non_empty(
        [
            row.get("birth_family_name"),
            row.get("usage_family_name"),
            *(row.get("birth_family_name_list") or []),
        ]
    )
    first_candidates = unique_non_empty(
        [
            row.get("birth_first_name"),
            row.get("usage_first_name"),
            *(row.get("birth_first_name_list") or []),
        ]
    )

    ## Create combinations
    combos: list[str] = []
    for fam, first in itertools.product(fam_candidates, first_candidates):
        candidate = f"{fam} {first}".strip()
        if candidate:
            combos.append(candidate)

    ## Unique stable
    out: list[str] = []
    seen = set()
    for c in combos:
        if c not in seen:
            seen.add(c)
            out.append(c)

    return out

def build_full_name_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
        Build full name candidates for all rows

        Output:
            - all_full_names: list[str]

        Args:
            df: Input dataframe

        Returns:
            Dataframe with all_full_names column
    """
    
    ## Ensure required columns exist
    required = [
        "birth_family_name",
        "usage_family_name",
        "birth_first_name",
        "usage_first_name",
        "birth_family_name_list",
        "birth_first_name_list",
    ]
    
    ensure_columns(df, required)

    ## Apply per-row builder
    df["all_full_names"] = df.apply(row_full_name_candidates, axis=1)
    return df

def row_address_candidates(row: pd.Series) -> list[str]:
    """
        Build address candidates for a single row

        Fields used:
            - addresses_street, addresses_complement, addresses_city, addresses_postal_code

        Notes:
            - This can be combinatorial if lists are large
            - Prefer keeping lists small during preprocessing

        Args:
            row: Row of a dataframe

        Returns:
            List of candidate addresses (stable unique)
    """
    
    ## Normalize components and provide safe defaults
    street = unique_non_empty(row.get("addresses_street") or []) or [""]
    comp = unique_non_empty(row.get("addresses_complement") or []) or [""]
    city = unique_non_empty(row.get("addresses_city") or []) or [""]
    postal = unique_non_empty(row.get("addresses_postal_code") or []) or [""]

    ## Build combinations
    combos: list[str] = []
    for a, b, c, d in itertools.product(street, comp, city, postal):
        candidate = " ".join([x for x in [a, b, c, d] if x]).strip()
        if candidate:
            combos.append(candidate)

    ## Unique stable
    out: list[str] = []
    seen = set()
    for cnd in combos:
        if cnd not in seen:
            seen.add(cnd)
            out.append(cnd)

    return out

def build_address_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
        Build address candidates for all rows

        Output:
            - all_addresses: list[str]

        Args:
            df: Input dataframe

        Returns:
            Dataframe with all_addresses column
    """
    
    ## Ensure required columns exist
    required = [
        "addresses_street",
        "addresses_complement",
        "addresses_city",
        "addresses_postal_code",
    ]
    ensure_columns(df, required)

    ## Apply per-row builder
    df["all_addresses"] = df.apply(row_address_candidates, axis=1)
    
    return df