'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Application entrypoint: environment setup, validation and FastAPI bootstrap."
'''

from __future__ import annotations

## Standard library
import argparse
import os
import sys
import time
from typing import Optional

## Local imports
from src.core.config import ensure_directories_exist
from src.utils.logging_utils import get_logger

## ============================================================
## CONSTANTS
## ============================================================
APP_VERSION = "1.0.0"
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

LOGGER = get_logger("data_deduplication.main")

## ============================================================
## ARG PARSER
## ============================================================
def _build_parser() -> argparse.ArgumentParser:
    """
        Build CLI parser for bootstrap and validation

        Returns:
            ArgumentParser instance
    """

    parser = argparse.ArgumentParser(
        description="Data Deduplication bootstrap",
        add_help=True,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APP_VERSION}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment without executing bootstrap",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate environment and exit",
    )

    return parser

## ============================================================
## VALIDATION
## ============================================================
def _validate_environment() -> dict:
    """
        Validate runtime environment

        Returns:
            Validation summary
    """

    return {
        "cwd": os.getcwd(),
        "env": os.getenv("ENV", "dev"),
        "python": sys.executable,
    }

## ============================================================
## SUMMARY
## ============================================================
def _build_summary(
    action: str,
    success: bool,
    start: float,
    details: Optional[dict] = None,
) -> dict:
    """
        Build execution summary

        Args:
            action: Action name
            success: Status
            start: Start time
            details: Optional details

        Returns:
            Summary dictionary
    """

    return {
        "action": action,
        "success": success,
        "duration_seconds": round(time.monotonic() - start, 3),
        "details": details or {},
    }

## ============================================================
## MAIN LOGIC
## ============================================================
def main() -> int:
    """
        Application bootstrap entry point

        Responsibilities:
            - Validate environment
            - Ensure required directories exist
            - Provide standardized CLI behavior

        Returns:
            Exit code
    """

    start_time = time.monotonic()

    parser = _build_parser()
    args = parser.parse_args()

    try:
        ## Validate environment
        env_summary = _validate_environment()

        if args.validate_config:
            LOGGER.info("Environment validation OK | %s", env_summary)
            LOGGER.info(
                "Summary | %s",
                _build_summary("validate-config", True, start_time),
            )
            return EXIT_SUCCESS

        if args.dry_run:
            LOGGER.info("Dry-run | bootstrap would execute")
            LOGGER.info(
                "Summary | %s",
                _build_summary("dry-run", True, start_time, env_summary),
            )
            return EXIT_SUCCESS

        ## Ensure directories
        ensure_directories_exist()

        ## Log context
        LOGGER.info("Application bootstrap completed")
        LOGGER.info("ENV=%s", os.getenv("ENV", "dev"))

        ## API instruction (kept explicit)
        LOGGER.info(
            "Start API with: uvicorn src.core.service:app --host 0.0.0.0 --port 8080"
        )

        LOGGER.info(
            "Summary | %s",
            _build_summary("bootstrap", True, start_time, env_summary),
        )

        return EXIT_SUCCESS

    except KeyboardInterrupt:
        LOGGER.warning("Execution interrupted by user")
        LOGGER.warning(
            "Summary | %s",
            _build_summary("interrupt", False, start_time),
        )
        return EXIT_FAILURE

    except Exception as exc:
        LOGGER.exception("Unhandled exception: %s", exc)
        LOGGER.error(
            "Summary | %s",
            _build_summary("exception", False, start_time),
        )
        return EXIT_FAILURE

## ============================================================
## ENTRYPOINT
## ============================================================
if __name__ == "__main__":
    sys.exit(main())