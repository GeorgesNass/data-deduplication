'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Core helper utilities: timeouts, safe casting, text normalization, iteration helpers, timing, and JSON payload parsing."
'''

from __future__ import annotations

## Standard library imports
import random
import string
import time
import json
import unicodedata
from threading import Thread
from typing import Any, Callable, Iterator, List, Sequence, TypeVar

## Local imports
from src.core.errors import ValidationError

## ============================================================
## GLOBALS
## ============================================================

T = TypeVar("T")

## ============================================================
## TIMEOUT DECORATOR
## ============================================================
def timeout(seconds: int) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
        Timeout decorator using a daemon thread

        Limitations:
            - The target function is executed in a thread
            - Hard-kill is not possible in Python threads, so we only stop waiting
            - Suitable for I/O bound tasks or best-effort execution guards

        Args:
            seconds: Timeout in seconds

        Returns:
            A decorator that raises TimeoutError if exceeded
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result: List[Any] = [TimeoutError(
                f"Function '{func.__name__}' timeout exceeded ({seconds}s)"
            )]

            ## Run the function in a daemon thread
            def run() -> None:
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as exc:
                    result[0] = exc

            thread = Thread(target=run, daemon=True)
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(
                    f"Function '{func.__name__}' timeout exceeded ({seconds}s)"
                )

            if isinstance(result[0], BaseException):
                raise result[0]

            return result[0]

        return wrapper

    return decorator

## ============================================================
## SAFE CASTING
## ============================================================
def safe_int(value: Any, default: int | None = None) -> int | None:
    """
        Safely cast a value to int

        Args:
            value: Input value to cast
            default: Fallback if casting fails

        Returns:
            int value or default
    """
    
    if value is None:
        return default

    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value: Any, default: float | None = None) -> float | None:
    """
        Safely cast a value to float

        Args:
            value: Input value to cast
            default: Fallback if casting fails

        Returns:
            float value or default
    """
    
    if value is None:
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_str(value: Any, default: str = "") -> str:
    """
        Safely cast a value to string

        Args:
            value: Input value to cast
            default: Fallback if value is None

        Returns:
            String representation
    """
    
    if value is None:
        return default

    return str(value)

## ============================================================
## TEXT NORMALIZATION
## ============================================================
def is_ascii(text: str) -> bool:
    """
        Check whether a string contains only ASCII characters

        Args:
            text: Input string

        Returns:
            True if ASCII only
    """
    
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

def remove_accents(text: str) -> str:
    """
        Remove accents/diacritics from a string

        Args:
            text: Input string

        Returns:
            Normalized string without accents
    """
    
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKD", text)
    
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))

def normalize_no_accents(text: str) -> str:
    """
        Normalize text for matching without accents

        Steps:
            - Trim
            - Lowercase
            - Remove accents
            - Collapse multiple spaces

        Args:
            text: Input string

        Returns:
            Normalized string
    """
    
    if not text:
        return ""

    text_norm = remove_accents(text.strip().lower())
    text_norm = " ".join(text_norm.split())
    
    return text_norm

def list_substrs_included(substrings: Sequence[str], text: str) -> List[str]:
    """
        Return substrings that are included in a target string

        Args:
            substrings: List of candidate substrings
            text: Target string

        Returns:
            List of substrings present in text
    """
    
    if not text:
        return []

    found: List[str] = []
    for sub in substrings:
        if sub and sub in text:
            found.append(sub)

    return found

## ============================================================
## RANDOM / IDS
## ============================================================
def get_random_string(length: int = 12) -> str:
    """
        Generate a random alphanumeric string

        Args:
            length: String length

        Returns:
            Random string
    """
    
    alphabet = string.ascii_letters + string.digits
    
    return "".join(random.choice(alphabet) for _ in range(max(1, length)))

## ============================================================
## ITERATION HELPERS
## ============================================================
def chunk_list(items: Sequence[T], chunk_size: int) -> Iterator[List[T]]:
    """
        Yield successive chunks from a sequence

        Args:
            items: Input sequence
            chunk_size: Chunk size

        Yields:
            Lists of size <= chunk_size
    """
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    for i in range(0, len(items), chunk_size):
        yield list(items[i:i + chunk_size])

## ============================================================
## SIMPLE TIMING HELPERS
## ============================================================
def now_ts() -> float:
    """
        Return current timestamp (seconds)

        Returns:
            Current time.time value
    """
    
    return time.time()

def elapsed_seconds(start_ts: float) -> float:
    """
        Compute elapsed seconds since start

        Args:
            start_ts: Start timestamp (seconds)

        Returns:
            Elapsed seconds
    """
    
    return max(0.0, time.time() - start_ts)

## ============================================================
## REQUEST / PAYLOAD HELPERS
## ============================================================
def parse_request_payload(raw_bytes: bytes | None) -> dict:
    """
        Parse a request payload expected to be JSON

        Args:
            raw_bytes: Raw request body bytes

        Returns:
            Parsed JSON payload as dict (empty dict if missing)

        Raises:
            ValidationError: If payload is not valid JSON
    """

    if not raw_bytes:
        return {}

    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        ## Try to decode with default json behavior as a fallback
        try:
            return json.loads(raw_bytes)
        except Exception:
            raise ValidationError(
                message="Request payload must be valid JSON",
                details=str(exc),
            )