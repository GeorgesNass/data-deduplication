'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Dataset preprocessing for deduplication: normalization, nested-list parsing, and lightweight field enrichment."
'''

from __future__ import annotations

## Standard library imports
import re
from typing import Any, Dict, Iterable, Sequence

## Third-party imports
import pandas as pd
from unidecode import unidecode

## Local imports
from src.core.config import load_data_control_config
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger("model.cleaning")

## ============================================================
## CONSTANTS
## ============================================================
DEFAULT_NESTED_LIST_SEPARATOR = "@@@"

## NOTE: Heuristic list-like columns used when we need to create missing columns
DEFAULT_LIST_COLUMNS = {
    "birth_family_name_list",
    "birth_first_name_list",
    "addresses_street",
    "addresses_complement",
    "addresses_city",
    "addresses_postal_code",
    "emails",
    "telephones",
    "all_full_names",
    "all_addresses",
}

## ============================================================
## TEXT NORMALIZATION
## ============================================================
def normalize_text(value: Any) -> str:
    """
        Normalize a value into a canonical string suitable for fuzzy matching

        High-level rules:
            - Cast to string
            - Remove accents
            - Remove known junk patterns
            - Replace separators by spaces
            - Normalize +33 to 0 (France)
            - Collapse spaces, trim, lowercase

        Args:
            value: Any input value

        Returns:
            Normalized string
    """
    
    ## Handle missing values early
    if value is None:
        return ""

    ## Cast and remove accents
    s = unidecode(str(value))

    ## Remove common junk patterns
    junk_patterns = [
        r"\bnone\b",
        r"\[\]",
        r"\[''\]",
        "T00 00 00.000Z",
        "T00:00:00.000Z",
        "00 00 00.000",
        "00:00:00.000",
    ]
    
    for pat in junk_patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)

    ## Replace separators by spaces
    s = re.sub(r"[,/\n:]+", " ", s)

    ## Normalize French phone format
    s = re.sub(r"\+33", "0", s)

    ## Normalize quotes + casing + spaces
    s = s.strip().strip('"').strip("'").lower()
    s = " ".join(s.split())

    return s

def clean_cell(value: Any) -> Any:
    """
        Clean a dataframe cell (string or list)

        Rules:
            - list -> normalize each element, drop empties, return list or ""
            - scalar -> normalize string, return "" if empty

        Args:
            value: Input value

        Returns:
            Cleaned value (list[str] or str)
    """
    
    ## List case
    if isinstance(value, list):
        cleaned_list = [normalize_text(x) for x in value]
        cleaned_list = [x for x in cleaned_list if x and x != "none"]
        return cleaned_list if cleaned_list else ""

    ## Scalar case
    cleaned_scalar = normalize_text(value)
    return "" if cleaned_scalar in ("", "none") else cleaned_scalar

## ============================================================
## LIST PARSING
## ============================================================
def parse_nested_list(value: Any, separator: str = DEFAULT_NESTED_LIST_SEPARATOR) -> list[str]:
    """
        Parse a nested-list-like cell into a list of normalized strings

        Supported inputs:
            - list -> normalized unique
            - string -> split using separator -> normalized unique
            - empty/none -> []

        Args:
            value: Cell content
            separator: Split token

        Returns:
            List of unique normalized strings (stable order)
    """
    
    ## Missing values
    if value is None:
        return []

    ## Convert input to a raw list
    if isinstance(value, list):
        raw_items = value
    else:
        raw_str = str(value).strip()
        if raw_str == "" or raw_str.lower() == "none":
            return []
        raw_items = raw_str.split(separator)

    ## Normalize + unique (stable order)
    out: list[str] = []
    seen = set()
    for item in raw_items:
        token = normalize_text(item)
        if token and token != "none" and token not in seen:
            seen.add(token)
            out.append(token)

    return out

## ============================================================
## DATAFRAME STRUCTURE HELPERS
## ============================================================
def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """
        Ensure required columns exist in the dataframe

        Missing columns are created with safe defaults:
            - list-like columns -> []
            - scalar columns -> ""

        Args:
            df: Input dataframe
            required: Required column names

        Returns:
            Dataframe with missing columns added
    """
    
    ## Create missing columns using safe defaults
    for col in required:
        if col in df.columns:
            continue

        ## Default to list if the column is known list-like
        if col in DEFAULT_LIST_COLUMNS:
            df[col] = [[] for _ in range(len(df))]
        else:
            df[col] = ["" for _ in range(len(df))]

    return df

def clean_scalar_columns(df: pd.DataFrame, nested_cols: Sequence[str]) -> pd.DataFrame:
    """
        Clean scalar columns in a dataframe

        Args:
            df: Input dataframe
            nested_cols: Columns treated as nested lists (skip scalar cleaning)

        Returns:
            Dataframe with cleaned scalar columns
    """
    
    ## Apply normalization to non-nested columns only
    for col in df.columns:
        if col in nested_cols:
            continue
        df[col] = df[col].apply(normalize_text)

    return df

def parse_nested_columns(df: pd.DataFrame, nested_cols: Sequence[str]) -> pd.DataFrame:
    """
        Parse and normalize nested list columns

        Args:
            df: Input dataframe
            nested_cols: Columns that contain nested list content

        Returns:
            Dataframe with parsed nested list columns
    """
    ## Parse only configured columns that exist
    for col in nested_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(parse_nested_list)

    return df

## ============================================================
## FIELD ENRICHMENT
## ============================================================
def enrich_emails(email_list: Any) -> list[str]:
    """
        Enrich emails by adding local-part tokens (before '@')

        Example:
            ["john.doe@gmail.com"] -> ["john.doe@gmail.com", "john.doe"]

        Args:
            email_list: List of emails or any value

        Returns:
            Unique normalized tokens (stable order)
    """
    
    ## Validate type
    if not isinstance(email_list, list):
        return []

    ## Build base + local-part tokens
    tokens: list[str] = []
    for email in email_list:
        email_norm = normalize_text(email)
        if not email_norm or "@" not in email_norm:
            continue

        local_part = email_norm.split("@")[0].strip()
        if local_part:
            tokens.append(email_norm)
            tokens.append(local_part)

    ## Unique stable
    out: list[str] = []
    seen = set()
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    return out

def normalize_birth_date(value: Any) -> str:
    """
        Normalize a birth date field

        Rule:
            - normalize text
            - keep only date part (drop time if present)

        Args:
            value: Input value

        Returns:
            Normalized date string or ""
    """
    
    s = normalize_text(value)
    if not s:
        return ""
    
    return s.split(" ")[0]

## ============================================================
## MAIN PREPROCESS ENTRYPOINT
## ============================================================
def preprocess_dataset(df: pd.DataFrame, data_control: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
        Preprocess dataset for deduplication and linkage

        High-level workflow:
            1) Load data_control.json (optional)
            2) Extract list nested columns from config
            3) Ensure nested columns exist
            4) Clean scalar columns
            5) Parse nested columns
            6) Enrich high-signal fields (emails, birth_date)

        Args:
            df: Raw input dataframe
            data_control: Optional dict loaded from data_control.json

        Returns:
            Preprocessed dataframe ready for fuzzy matching
    """
    
    ## Load config if not provided
    if data_control is None:
        try:
            data_control = load_data_control_config()
        except Exception as exc:
            LOGGER.warning("Could not load data_control.json: %s", exc)
            data_control = {}

    ## Extract nested columns definition
    nested_cols = data_control.get("list_nested_columns", [])
    if not isinstance(nested_cols, list):
        nested_cols = []

    ## Ensure nested columns exist
    ensure_columns(df, nested_cols)

    ## Clean scalar columns
    clean_scalar_columns(df, nested_cols)

    ## Parse nested list columns
    parse_nested_columns(df, nested_cols)

    ## Enrich emails
    if "emails" in df.columns:
        df["emails"] = df["emails"].apply(enrich_emails)

    ## Normalize birth_date
    if "birth_date" in df.columns:
        df["birth_date"] = df["birth_date"].apply(normalize_birth_date)

    return df
