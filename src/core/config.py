'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized configuration management: paths, environment parsing, and JSON/YAML config helpers."
'''

from __future__ import annotations

## Standard library imports
import json
import os
from pathlib import Path
from typing import Any, Dict

## ============================================================
## PROJECT ROOT & PATHS
## ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

CONFIG_DIR = ARTIFACTS_DIR / "config"
EXAMPLES_DIR = ARTIFACTS_DIR / "examples"

RAW_DATA_DIR = DATA_DIR / "raw"
ACTIVE_LEARNING_DIR = DATA_DIR / "active_learning"

## ============================================================
## DEFAULT FILENAMES
## ============================================================
DEFAULT_SWAGGER_FILENAME = "swagger.yaml"
DEFAULT_DATA_CONTROL_FILENAME = "data_control.json"

## ============================================================
## ENVIRONMENT VARIABLES
## ============================================================
ENV = os.getenv("ENV", "dev")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "8053"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}

## ============================================================
## CONFIG FILE PATHS
## ============================================================
SWAGGER_CONFIG_PATH = CONFIG_DIR / DEFAULT_SWAGGER_FILENAME
DATA_CONTROL_CONFIG_PATH = CONFIG_DIR / DEFAULT_DATA_CONTROL_FILENAME

## ============================================================
## ENV HELPERS
## ============================================================
def get_env_str(key: str, default: str | None = None) -> str | None:
    """
        Retrieve an environment variable as string

        Args:
            key: Environment variable name
            default: Default value if missing

        Returns:
            The string value or default
    """
    
    return os.getenv(key, default)

def get_env_int(key: str, default: int) -> int:
    """
        Retrieve an environment variable as integer

        Args:
            key: Environment variable name
            default: Default integer value

        Returns:
            Parsed integer, or default if parsing fails
    """
    
    raw = os.getenv(key)
    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError:
        return default

def get_env_bool(key: str, default: bool = False) -> bool:
    """
        Retrieve an environment variable as boolean

        Accepted true values:
            - 1, true, yes, y, on

        Accepted false values:
            - 0, false, no, n, off

        Args:
            key: Environment variable name
            default: Default boolean value

        Returns:
            Parsed boolean value
    """
    
    raw = os.getenv(key)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    return default

## ============================================================
## FILE HELPERS
## ============================================================
def resolve_path(path: str | Path) -> Path:
    """
        Resolve a path into an absolute Path

        Args:
            path: Path-like input

        Returns:
            Absolute resolved Path
    """
    
    return Path(path).expanduser().resolve()

def ensure_dir(path: str | Path) -> Path:
    """
        Ensure a directory exists

        Args:
            path: Directory path

        Returns:
            Absolute resolved Path to the directory
    """
    
    dir_path = resolve_path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
   
    return dir_path

def ensure_file_exists(path: str | Path) -> Path:
    """
        Ensure a file exists on disk

        Args:
            path: File path

        Returns:
            Absolute resolved Path to the file

        Raises:
            FileNotFoundError: If the file does not exist
    """
    
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return file_path

def load_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """
        Load a text file from disk

        Args:
            path: Text file path
            encoding: File encoding

        Returns:
            File content as text
    """
    
    file_path = ensure_file_exists(path)
    
    return file_path.read_text(encoding=encoding)

def save_text_file(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """
        Save text content to disk

        Args:
            path: Output file path
            content: Text content to write
            encoding: File encoding

        Returns:
            Absolute resolved Path to the written file
    """
    
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding=encoding)
    
    return file_path

## ============================================================
## JSON HELPERS
## ============================================================
def load_json_config(path: str | Path) -> Dict[str, Any]:
    """
        Load a JSON file from disk

        Args:
            path: Path to a JSON file

        Returns:
            Parsed JSON content

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file content is not valid JSON
    """
    
    file_path = ensure_file_exists(path)

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file: {file_path}") from exc

def save_json_file(path: str | Path, payload: Dict[str, Any], indent: int = 2) -> Path:
    """
        Save a JSON-serializable dict to disk

        Args:
            path: Output JSON path
            payload: JSON-serializable dictionary
            indent: JSON indentation

        Returns:
            Absolute resolved Path to the written file
    """
    
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=indent)

    return file_path

## ============================================================
## PROJECT-SPECIFIC LOADERS
## ============================================================
def load_data_control_config() -> Dict[str, Any]:
    """
        Load data validation and control rules

        Returns:
            Dictionary containing data control rules
    """
    
    return load_json_config(DATA_CONTROL_CONFIG_PATH)

def load_swagger_config_path() -> Path:
    """
        Get the Swagger/OpenAPI configuration path

        Returns:
            Absolute Path to swagger.yaml (may not exist)
    """
    
    return resolve_path(SWAGGER_CONFIG_PATH)

def list_example_payloads() -> list[Path]:
    """
        List available example JSON payload files

        Returns:
            Sorted list of example JSON paths (existing files only)
    """
    
    if not EXAMPLES_DIR.exists():
        return []

    return sorted([p for p in EXAMPLES_DIR.glob("*.json") if p.is_file()])

def get_project_paths() -> Dict[str, Path]:
    """
        Provide a centralized dictionary of project paths

        Returns:
            Mapping of well-known path names to Path objects
    """
    
    return {
        "project_root": PROJECT_ROOT,
        "artifacts_dir": ARTIFACTS_DIR,
        "config_dir": CONFIG_DIR,
        "examples_dir": EXAMPLES_DIR,
        "data_dir": DATA_DIR,
        "raw_data_dir": RAW_DATA_DIR,
        "active_learning_dir": ACTIVE_LEARNING_DIR,
        "logs_dir": LOGS_DIR,
        "swagger_config_path": SWAGGER_CONFIG_PATH,
        "data_control_config_path": DATA_CONTROL_CONFIG_PATH,
    }

## ============================================================
## INITIALIZATION CHECKS
## ============================================================
def ensure_directories_exist() -> None:
    """
        Ensure required runtime directories exist

        Directories:
            - logs
            - data/raw
            - data/active_learning
            - artifacts/examples
            - artifacts/config
    """
    
    ensure_dir(LOGS_DIR)
    ensure_dir(RAW_DATA_DIR)
    ensure_dir(ACTIVE_LEARNING_DIR)
    ensure_dir(EXAMPLES_DIR)
    ensure_dir(CONFIG_DIR)