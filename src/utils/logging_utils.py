'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized logging utilities: consistent console/file handlers, formatting, and safe logger retrieval."
'''

from __future__ import annotations

## Standard library imports
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

## ============================================================
## CONSTANTS
## ============================================================
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_DIRNAME = os.getenv("LOG_DIR", "logs")
DEFAULT_LOG_FILENAME = os.getenv("LOG_FILE", "app.log")

DEFAULT_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "5242880"))  ## 5 MB
DEFAULT_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

## ============================================================
## INTERNAL HELPERS
## ============================================================
def _resolve_log_dir(log_dir: str | Path | None = None) -> Path:
    """
        Resolve and create the log directory

        Args:
            log_dir: Optional explicit directory path

        Returns:
            A Path to an existing directory
    """
    
    ## Use default if not provided
    log_dir_path = Path(DEFAULT_LOG_DIRNAME) if log_dir is None else Path(log_dir)

    ## Ensure directory exists
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    return log_dir_path.resolve()

def _parse_log_level(level: str | int | None) -> int:
    """
        Parse a log level into a logging integer

        Args:
            level: Log level name (INFO, DEBUG) or logging int

        Returns:
            A valid logging level integer
    """
    
    ## Default level from env
    if level is None:
        return logging._nameToLevel.get(DEFAULT_LOG_LEVEL, logging.INFO)

    ## Integer level provided
    if isinstance(level, int):
        return int(level)

    ## String level provided
    normalized = str(level).upper().strip()
    
    return logging._nameToLevel.get(normalized, logging.INFO)

def _build_formatter() -> logging.Formatter:
    """
        Build the default formatter

        Returns:
            A configured logging.Formatter
    """
    
    return logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

def _has_handler_type(logger: logging.Logger, handler_type: type) -> bool:
    """
        Check whether a logger already has a handler of a given type

        Args:
            logger: Logger instance
            handler_type: Handler class

        Returns:
            True if at least one handler matches
    """
    
    return any(isinstance(h, handler_type) for h in logger.handlers)

def _has_rotating_file_for_path(logger: logging.Logger, target_path: Path) -> bool:
    """
        Check whether a logger already has a RotatingFileHandler targeting a given file

        Args:
            logger: Logger instance
            target_path: Target log file path

        Returns:
            True if a matching RotatingFileHandler exists
    """
    
    target = str(target_path.resolve())
    
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == target:
            return True
    
    return False

def _ensure_console_handler(logger: logging.Logger, level: int) -> None:
    """
        Ensure a single StreamHandler is attached to the logger

        Args:
            logger: Logger instance
            level: Logger/handler level
    """
    
    ## Prevent duplicates
    if _has_handler_type(logger, logging.StreamHandler):
        return

    ## Attach stream handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_build_formatter())
    logger.addHandler(handler)

def _ensure_file_handler(
    logger: logging.Logger,
    log_dir: Path,
    filename: str,
    level: int,
    max_bytes: int,
    backup_count: int,
) -> None:
    """
        Ensure a single rotating file handler is attached to the logger

        Args:
            logger: Logger instance
            log_dir: Existing directory for log file
            filename: Log filename
            level: Logger/handler level
            max_bytes: Max file size before rotation
            backup_count: Number of backups to keep
    """
    
    ## Build log path
    log_path = (log_dir / filename).resolve()

    ## Prevent duplicates for the same file
    if _has_rotating_file_for_path(logger, log_path):
        return

    ## Attach rotating file handler
    handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(_build_formatter())
    logger.addHandler(handler)

## ============================================================
## PUBLIC API
## ============================================================
def get_logger(
    name: str,
    level: str | int | None = None,
    log_dir: str | Path | None = None,
    filename: Optional[str] = None,
    enable_file: bool = True,
    enable_console: bool = True,
    propagate: bool = False,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    clear_handlers: bool = False,
) -> logging.Logger:
    """
        Get a configured logger with consistent handlers and formatting

        Design goals:
            - Idempotent: calling get_logger multiple times does not duplicate handlers
            - Safe defaults: console + rotating file handler (unless disabled)
            - Test-friendly: optional clear_handlers to avoid pytest duplication

        Args:
            name: Logger name (namespace)
            level: Log level (string like INFO/DEBUG or int)
            log_dir: Directory where logs are written (default: ./logs)
            filename: Log filename (default: app.log)
            enable_file: If True, attach a rotating file handler
            enable_console: If True, attach a console (stream) handler
            propagate: Whether to propagate to parent loggers
            max_bytes: Rotating file max size (bytes)
            backup_count: Rotating file backup count
            clear_handlers: If True, remove existing handlers before attaching new ones

        Returns:
            A configured logging.Logger instance
    """
    
    ## Resolve final config values
    resolved_level = _parse_log_level(level)
    resolved_filename = filename or DEFAULT_LOG_FILENAME
    resolved_max_bytes = max_bytes if max_bytes is not None else DEFAULT_MAX_BYTES
    resolved_backup_count = backup_count if backup_count is not None else DEFAULT_BACKUP_COUNT

    ## Get logger instance
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = propagate

    ## Optional cleanup (useful for tests / hot reload)
    if clear_handlers:
        logger.handlers.clear()

    ## Attach console handler
    if enable_console:
        _ensure_console_handler(logger, resolved_level)

    ## Attach file handler
    if enable_file:
        resolved_log_dir = _resolve_log_dir(log_dir)
        _ensure_file_handler(
            logger=logger,
            log_dir=resolved_log_dir,
            filename=resolved_filename,
            level=resolved_level,
            max_bytes=resolved_max_bytes,
            backup_count=resolved_backup_count,
        )

    return logger