'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Application entrypoint: environment setup and FastAPI service launcher."
'''

from __future__ import annotations

## Standard library imports
import os

## Local imports
from src.core.config import ensure_directories_exist
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger("main")

## ============================================================
## BOOTSTRAP
## ============================================================
def main() -> None:
    """
        Application bootstrap

        Responsibilities:
            - Ensure runtime directories exist (logs, data/raw, data/active_learning)
            - Provide a clear entrypoint for Docker/CLI usage

        Notes:
            - FastAPI is served via uvicorn:
                uvicorn src.core.service:app --host 0.0.0.0 --port 8080
            - This script keeps bootstrap responsibilities only
    """
    
    ## Ensure filesystem prerequisites
    ensure_directories_exist()

    ## Log environment context (best-effort)
    LOGGER.info("Starting application")
    LOGGER.info("ENV=%s", os.getenv("ENV", "dev"))

    ## No direct server start here for FastAPI
    LOGGER.info("Use uvicorn to start the API: uvicorn src.core.service:app --host 0.0.0.0 --port 8080")

## ============================================================
## ENTRYPOINT
## ============================================================
if __name__ == "__main__":
    main()