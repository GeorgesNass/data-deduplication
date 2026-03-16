'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Unified configuration loader for data-deduplication: dotenv, env parsing, paths, profiles, files, JSON/YAML helpers, secrets and runtime metadata."
'''

from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

try:
    from src.core.errors import ConfigurationError
except Exception:
    class ConfigurationError(ValueError):
        """
            Fallback configuration error when the project error module is unavailable
        """

## ============================================================
## PLACEHOLDER TOKENS
## ============================================================
PLACEHOLDER_PREFIXES: Tuple[str, ...] = ("<YOUR_", "YOUR_", "CHANGE_ME", "REPLACE_ME", "TODO")

## ============================================================
## OS / SYSTEM CONSTANTS
## ============================================================
SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"
IS_LINUX = SYSTEM_NAME == "linux"
IS_MACOS = SYSTEM_NAME == "darwin"
DEFAULT_ENCODING = "utf-8"
CSV_SEPARATOR = ";"

## ============================================================
## STABLE DOMAIN CONSTANTS
## ============================================================
DEFAULT_APP_NAME = "data-deduplication"
DEFAULT_APP_VERSION = "1.0.0"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_PROFILE = "api"

DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_SECRETS_DIR = "secrets"
DEFAULT_CONFIG_DIR = "artifacts/config"
DEFAULT_EXAMPLES_DIR = "artifacts/examples"
DEFAULT_RAW_DATA_DIR = "data/raw"
DEFAULT_ACTIVE_LEARNING_DIR = "data/active_learning"

DEFAULT_SWAGGER_FILENAME = "swagger.yaml"
DEFAULT_DATA_CONTROL_FILENAME = "data_control.json"
DEFAULT_FLASK_HOST = "0.0.0.0"
DEFAULT_FLASK_PORT = 8053

SUPPORTED_INPUT_EXTENSIONS = (".csv", ".json", ".txt", ".xlsx", ".xls")
SUPPORTED_EXPORT_EXTENSIONS = (".json", ".yaml", ".yml", ".txt")

## ============================================================
## CONFIG MODELS
## ============================================================
@dataclass(frozen=True)
class ExecutionMetadata:
    """
        Execution metadata

        Args:
            run_id: Unique runtime identifier
            started_at_utc: UTC timestamp when config was built
            hostname: Current host name
            platform_name: Current operating system name
            profile: Active runtime profile
            environment: Active environment
    """

    run_id: str
    started_at_utc: str
    hostname: str
    platform_name: str
    profile: str
    environment: str

@dataclass(frozen=True)
class PathsConfig:
    """
        Filesystem paths configuration

        Args:
            project_root: Project root directory
            src_dir: Source directory
            data_dir: Data root directory
            raw_data_dir: Raw data directory
            active_learning_dir: Active learning data directory
            artifacts_dir: Artifacts root directory
            config_dir: Config artifact directory
            examples_dir: Example payload directory
            logs_dir: Logs directory
            secrets_dir: Secrets directory
            swagger_config_path: Swagger/OpenAPI file path
            data_control_config_path: Data control JSON path
    """

    project_root: Path
    src_dir: Path
    data_dir: Path
    raw_data_dir: Path
    active_learning_dir: Path
    artifacts_dir: Path
    config_dir: Path
    examples_dir: Path
    logs_dir: Path
    secrets_dir: Path
    swagger_config_path: Path
    data_control_config_path: Path

@dataclass(frozen=True)
class RuntimeConfig:
    """
        Runtime configuration

        Args:
            environment: Environment name
            profile: Active runtime profile
            debug: Whether debug mode is enabled
            log_level: Logging level
            flask_host: Flask bind host
            flask_port: Flask bind port
            max_workers: Maximum worker count
            batch_size: Batch size for internal operations
            batch_sleep_seconds: Sleep delay between batches
            request_timeout_seconds: Request timeout
            allowed_origins: Allowed HTTP origins for future usage
    """

    environment: str
    profile: str
    debug: bool
    log_level: str
    flask_host: str
    flask_port: int
    max_workers: int
    batch_size: int
    batch_sleep_seconds: float
    request_timeout_seconds: int
    allowed_origins: list[str]

@dataclass(frozen=True)
class DeduplicationConfig:
    """
        Deduplication configuration

        Args:
            default_similarity_threshold: Default matching threshold
            active_learning_enabled: Whether active learning is enabled
            save_examples: Whether example payload export is enabled
            export_format: Default export format
    """

    default_similarity_threshold: float
    active_learning_enabled: bool
    save_examples: bool
    export_format: str

@dataclass(frozen=True)
class SecretsConfig:
    """
        Secret values resolved from env or files

        Args:
            api_key: Optional application API key
    """

    api_key: str

@dataclass(frozen=True)
class AppConfig:
    """
        Unified application configuration

        Args:
            app_name: Application name
            app_version: Application version
            execution: Execution metadata
            paths: Filesystem paths configuration
            runtime: Runtime configuration
            deduplication: Deduplication configuration
            secrets: Secret values
    """

    app_name: str
    app_version: str
    execution: ExecutionMetadata
    paths: PathsConfig
    runtime: RuntimeConfig
    deduplication: DeduplicationConfig
    secrets: SecretsConfig

## ============================================================
## DOTENV / ENV HELPERS
## ============================================================
def _resolve_project_root() -> Path:
    """
        Resolve the project root path

        Returns:
            Absolute project root path
    """

    ## Prefer explicit project root override when available
    project_root_raw = os.getenv("PROJECT_ROOT", "").strip()
    return Path(project_root_raw).expanduser().resolve() if project_root_raw else Path(__file__).resolve().parents[2]

def _load_dotenv_if_present() -> None:
    """
        Load a local .env file if available

        Returns:
            None
    """

    ## Import dotenv lazily
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    ## Load project-level .env when present
    env_path = _resolve_project_root() / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

def _is_placeholder(value: str) -> bool:
    """
        Detect placeholder-like values

        Args:
            value: Raw environment value

        Returns:
            True if the value looks like a placeholder
    """

    ## Normalize before inspection
    normalized = value.strip().upper()
    return any(token in normalized for token in PLACEHOLDER_PREFIXES)

def get_env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """
        Retrieve an environment variable as string

        Args:
            key: Environment variable name
            default: Default value if missing

        Returns:
            The normalized string value or default
    """

    ## Read env and normalize whitespace
    value = os.getenv(key, default)
    if value is None:
        return default
    return value.strip()

def get_env_int(key: str, default: int) -> int:
    """
        Retrieve an environment variable as integer

        Args:
            key: Environment variable name
            default: Default integer value

        Returns:
            Parsed integer value

        Raises:
            ConfigurationError: If the value is invalid
    """

    ## Parse integer strictly
    raw = get_env_str(key, str(default))
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be an integer") from exc

def get_env_float(key: str, default: float) -> float:
    """
        Retrieve an environment variable as float

        Args:
            key: Environment variable name
            default: Default float value

        Returns:
            Parsed float value

        Raises:
            ConfigurationError: If the value is invalid
    """

    ## Parse float strictly
    raw = get_env_str(key, str(default))
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be a float") from exc

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

        Raises:
            ConfigurationError: If the value is invalid
    """

    ## Read and normalize raw value
    raw = get_env_str(key, str(default))
    normalized = raw.strip().lower() if raw is not None else str(default).lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value for {key}: {normalized}")

def get_env_list(key: str, default: Optional[list[str]] = None, *, separator: str = ",") -> list[str]:
    """
        Retrieve a list-like environment variable

        Args:
            key: Environment variable name
            default: Default list value
            separator: Raw value separator

        Returns:
            Parsed list of strings
    """

    ## Read raw list-like value
    raw = get_env_str(key, "")
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(separator) if item.strip()]

def _expand_env_vars(value: str) -> str:
    """
        Expand shell variables and user home in a string

        Args:
            value: Raw string value

        Returns:
            Expanded string
    """

    ## Expand shell variables
    return os.path.expandvars(value)

## ============================================================
## FILE HELPERS
## ============================================================
def resolve_path(path: str | Path, project_root: Optional[Path] = None) -> Path:
    """
        Resolve a path into an absolute Path

        Args:
            path: Path-like input
            project_root: Optional project root for relative paths

        Returns:
            Absolute resolved Path
    """

    ## Expand shell variables and user home
    path_obj = Path(_expand_env_vars(str(path))).expanduser()
    if path_obj.is_absolute():
        return path_obj.resolve()

    ## Resolve relative path from project root
    root = project_root or _resolve_project_root()
    return (root / path_obj).resolve()

def get_env_path(key: str, default: str, project_root: Path) -> Path:
    """
        Read and resolve a path environment variable

        Args:
            key: Environment variable name
            default: Default path value
            project_root: Project root directory

        Returns:
            Resolved path
    """

    ## Resolve env override or default path
    return resolve_path(get_env_str(key, default) or default, project_root)

def ensure_dir(path: str | Path) -> Path:
    """
        Ensure a directory exists

        Args:
            path: Directory path

        Returns:
            Absolute resolved Path to the directory
    """

    ## Resolve and create directory
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

    ## Resolve target file path
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path

def load_text_file(path: str | Path, encoding: str = DEFAULT_ENCODING) -> str:
    """
        Load a text file from disk

        Args:
            path: Text file path
            encoding: File encoding

        Returns:
            File content as text
    """

    ## Ensure the file exists before reading
    file_path = ensure_file_exists(path)
    return file_path.read_text(encoding=encoding)

def save_text_file(path: str | Path, content: str, encoding: str = DEFAULT_ENCODING) -> Path:
    """
        Save text content to disk

        Args:
            path: Output file path
            content: Text content to write
            encoding: File encoding

        Returns:
            Absolute resolved Path to the written file
    """

    ## Resolve output path and create parent directory
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding=encoding)
    return file_path

## ============================================================
## JSON / YAML HELPERS
## ============================================================
def load_json_config(path: str | Path) -> dict[str, Any]:
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

    ## Ensure file exists before parsing
    file_path = ensure_file_exists(path)
    try:
        with file_path.open("r", encoding=DEFAULT_ENCODING) as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in file: {file_path}") from exc

def save_json_file(path: str | Path, payload: dict[str, Any], indent: int = 2) -> Path:
    """
        Save a JSON-serializable dict to disk

        Args:
            path: Output JSON path
            payload: JSON-serializable dictionary
            indent: JSON indentation

        Returns:
            Absolute resolved Path to the written file
    """

    ## Resolve output path and ensure parent directory
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding=DEFAULT_ENCODING) as file:
        json.dump(payload, file, ensure_ascii=False, indent=indent)
    return file_path

def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
        Load a YAML file from disk

        Args:
            path: Path to a YAML file

        Returns:
            Parsed YAML content

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If the file does not exist
            ValueError: If the file content is not valid YAML
    """

    ## Require PyYAML only when YAML is used
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML files")

    ## Ensure file exists before parsing
    file_path = ensure_file_exists(path)
    try:
        with file_path.open("r", encoding=DEFAULT_ENCODING) as file:
            content = yaml.safe_load(file)
        return content if isinstance(content, dict) else {}
    except Exception as exc:
        raise ValueError(f"Invalid YAML in file: {file_path}") from exc

def save_yaml_file(path: str | Path, payload: dict[str, Any]) -> Path:
    """
        Save a YAML-serializable dict to disk

        Args:
            path: Output YAML path
            payload: YAML-serializable dictionary

        Returns:
            Absolute resolved Path to the written file

        Raises:
            ImportError: If PyYAML is not installed
    """

    ## Require PyYAML only when YAML is used
    if yaml is None:
        raise ImportError("PyYAML is required to save YAML files")

    ## Resolve output path and ensure parent directory
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding=DEFAULT_ENCODING) as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)
    return file_path

## ============================================================
## PROFILE / SECRET HELPERS
## ============================================================
def _read_secret_value(direct_key: str, file_key: str, *, project_root: Path, default: str = "") -> str:
    """
        Read a secret from env directly or from a file path

        Args:
            direct_key: Environment variable containing the secret
            file_key: Environment variable containing the secret file path
            project_root: Project root directory
            default: Default fallback value

        Returns:
            Secret value or default
    """

    ## Prefer direct env secret value first
    direct_value = get_env_str(direct_key, default) or default
    if direct_value and not _is_placeholder(direct_value):
        return direct_value

    ## Fallback to file-based secret
    secret_file_raw = get_env_str(file_key, "") or ""
    if not secret_file_raw:
        return default

    ## Resolve and read secret file when available
    secret_file = resolve_path(secret_file_raw, project_root)
    if secret_file.exists() and secret_file.is_file():
        return secret_file.read_text(encoding=DEFAULT_ENCODING).strip()
    return default

def _get_profiled_env(name: str, default: str, profile: str) -> str:
    """
        Read an env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Resolved string value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return get_env_str(override_key, default) if os.getenv(override_key) is not None else (get_env_str(name, default) or default)

def _get_profiled_env_bool(name: str, default: bool, profile: str) -> bool:
    """
        Read a boolean env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed boolean value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return get_env_bool(override_key, default) if os.getenv(override_key) is not None else get_env_bool(name, default)

def _get_profiled_env_int(name: str, default: int, profile: str) -> int:
    """
        Read an integer env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed integer value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return get_env_int(override_key, default) if os.getenv(override_key) is not None else get_env_int(name, default)

def _get_profiled_env_float(name: str, default: float, profile: str) -> float:
    """
        Read a float env value with optional profile override

        Args:
            name: Base environment variable name
            default: Default fallback value
            profile: Active runtime profile

        Returns:
            Parsed float value
    """

    ## Prefer profile-specific override when present
    override_key = f"{profile.upper()}_{name}"
    return get_env_float(override_key, default) if os.getenv(override_key) is not None else get_env_float(name, default)

## ============================================================
## VALIDATION / BUILD HELPERS
## ============================================================
def _validate_required_placeholders(keys: list[str]) -> None:
    """
        Validate that required values are not unresolved placeholders

        Args:
            keys: Environment keys to inspect

        Returns:
            None

        Raises:
            ConfigurationError: If placeholders are detected
    """

    ## Collect placeholder-based invalid keys
    invalid_keys = [key for key in keys if (value := get_env_str(key, "")) and _is_placeholder(value)]
    if invalid_keys:
        raise ConfigurationError("Placeholder values detected for: " + ", ".join(invalid_keys))

def _validate_positive_int(value: int, field_name: str) -> None:
    """
        Validate that an integer is strictly positive

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject non-positive integers
    if value <= 0:
        raise ConfigurationError(f"{field_name} must be > 0. Got: {value}")

def _validate_non_negative_float(value: float, field_name: str) -> None:
    """
        Validate that a float is non-negative

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject negative floats
    if value < 0.0:
        raise ConfigurationError(f"{field_name} must be >= 0. Got: {value}")

def _validate_probability(value: float, field_name: str) -> None:
    """
        Validate that a float is inside [0, 1]

        Args:
            value: Value to validate
            field_name: Human-readable field name

        Returns:
            None

        Raises:
            ConfigurationError: If invalid
    """

    ## Reject invalid probability values
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{field_name} must be in [0, 1]. Got: {value}")

def _ensure_directories_exist(paths: list[Path]) -> None:
    """
        Ensure required runtime directories exist

        Args:
            paths: Directories to create if missing

        Returns:
            None
    """

    ## Create all required directories
    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)

def _validate_export_format(value: str) -> str:
    """
        Validate default export format

        Args:
            value: Raw export format

        Returns:
            Validated export format

        Raises:
            ConfigurationError: If unsupported
    """

    ## Normalize and validate the chosen export format
    normalized = value.strip().lower()
    if normalized not in {"json", "yaml", "yml", "txt"}:
        raise ConfigurationError("DEFAULT_EXPORT_FORMAT must be one of: json, yaml, yml, txt")
    return normalized

def _validate_config(config: AppConfig) -> None:
    """
        Validate the final structured configuration

        Args:
            config: Structured configuration

        Returns:
            None
        """

    ## Validate runtime numeric parameters
    _validate_positive_int(config.runtime.flask_port, "FLASK_PORT")
    _validate_positive_int(config.runtime.max_workers, "MAX_WORKERS")
    _validate_positive_int(config.runtime.batch_size, "BATCH_SIZE")
    _validate_positive_int(config.runtime.request_timeout_seconds, "REQUEST_TIMEOUT_SECONDS")
    _validate_non_negative_float(config.runtime.batch_sleep_seconds, "BATCH_SLEEP_SECONDS")

    ## Validate deduplication parameters
    _validate_probability(config.deduplication.default_similarity_threshold, "DEFAULT_SIMILARITY_THRESHOLD")

    ## Validate fixed config files
    if config.paths.swagger_config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("SWAGGER_CONFIG_PATH must point to a YAML file")
    if config.paths.data_control_config_path.suffix.lower() != ".json":
        raise ConfigurationError("DATA_CONTROL_CONFIG_PATH must point to a JSON file")

## ============================================================
## PROJECT-SPECIFIC LOADERS
## ============================================================
def load_data_control_config() -> dict[str, Any]:
    """
        Load data validation and control rules

        Returns:
            Dictionary containing data control rules
    """

    ## Load the JSON control configuration
    return load_json_config(CONFIG.paths.data_control_config_path)

def load_swagger_config_path() -> Path:
    """
        Get the Swagger/OpenAPI configuration path

        Returns:
            Absolute Path to swagger.yaml
    """

    ## Return the resolved swagger path
    return CONFIG.paths.swagger_config_path

def list_example_payloads() -> list[Path]:
    """
        List available example JSON payload files

        Returns:
            Sorted list of example JSON paths
    """

    ## Return only existing JSON files from examples directory
    if not CONFIG.paths.examples_dir.exists():
        return []
    return sorted([p for p in CONFIG.paths.examples_dir.glob("*.json") if p.is_file()])

def get_project_paths() -> dict[str, Path]:
    """
        Provide a centralized dictionary of project paths

        Returns:
            Mapping of well-known path names to Path objects
    """

    ## Expose well-known paths with backward-compatible keys
    return {
        "project_root": CONFIG.paths.project_root,
        "artifacts_dir": CONFIG.paths.artifacts_dir,
        "config_dir": CONFIG.paths.config_dir,
        "examples_dir": CONFIG.paths.examples_dir,
        "data_dir": CONFIG.paths.data_dir,
        "raw_data_dir": CONFIG.paths.raw_data_dir,
        "active_learning_dir": CONFIG.paths.active_learning_dir,
        "logs_dir": CONFIG.paths.logs_dir,
        "swagger_config_path": CONFIG.paths.swagger_config_path,
        "data_control_config_path": CONFIG.paths.data_control_config_path,
    }

def ensure_directories_exist() -> None:
    """
        Ensure required runtime directories exist

        Returns:
            None
    """

    ## Recreate the standard runtime directories
    _ensure_directories_exist([
        CONFIG.paths.logs_dir,
        CONFIG.paths.raw_data_dir,
        CONFIG.paths.active_learning_dir,
        CONFIG.paths.examples_dir,
        CONFIG.paths.config_dir,
    ])

## ============================================================
## EXPORT HELPERS
## ============================================================
def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """
        Convert AppConfig into a serializable dictionary

        Args:
            config: Structured configuration object

        Returns:
            Serializable dictionary
    """

    ## Convert dataclass tree into a plain dictionary
    payload = asdict(config)

    ## Normalize Path objects recursively
    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: _normalize(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    return _normalize(payload)

def config_to_json(config: AppConfig) -> str:
    """
        Convert AppConfig into a JSON string

        Args:
            config: Structured configuration object

        Returns:
            JSON string
    """

    ## Serialize normalized configuration to JSON
    return json.dumps(config_to_dict(config), indent=2, ensure_ascii=False)

## ============================================================
## CONFIG FACTORY
## ============================================================
@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
        Build full application configuration from environment variables

        High-level workflow:
            1) Load optional project-level .env
            2) Resolve project root and active profile
            3) Build execution, paths, runtime and deduplication sections
            4) Resolve optional secrets
            5) Validate and cache the final AppConfig

        Returns:
            AppConfig instance
    """

    ## Load optional local .env file first
    _load_dotenv_if_present()

    ## Resolve project root and active runtime profile
    project_root = _resolve_project_root()
    environment = (get_env_str("ENV", DEFAULT_ENVIRONMENT) or DEFAULT_ENVIRONMENT).lower()
    profile = (get_env_str("PROFILE", DEFAULT_PROFILE) or DEFAULT_PROFILE).lower()

    ## Validate placeholder values where relevant
    _validate_required_placeholders(["ENV", "PROFILE", "API_KEY"])

    ## Build execution metadata
    execution = ExecutionMetadata(
        run_id=get_env_str("RUN_ID", str(uuid.uuid4())) or str(uuid.uuid4()),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        platform_name=SYSTEM_NAME,
        profile=profile,
        environment=environment,
    )

    ## Resolve main folders
    data_dir = get_env_path("DATA_DIR", DEFAULT_DATA_DIR, project_root)
    artifacts_dir = get_env_path("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR, project_root)
    logs_dir = get_env_path("LOGS_DIR", DEFAULT_LOGS_DIR, project_root)
    secrets_dir = get_env_path("SECRETS_DIR", DEFAULT_SECRETS_DIR, project_root)

    ## Build structured paths section
    paths = PathsConfig(
        project_root=project_root,
        src_dir=(project_root / "src").resolve(),
        data_dir=data_dir,
        raw_data_dir=get_env_path("RAW_DATA_DIR", DEFAULT_RAW_DATA_DIR, project_root),
        active_learning_dir=get_env_path("ACTIVE_LEARNING_DIR", DEFAULT_ACTIVE_LEARNING_DIR, project_root),
        artifacts_dir=artifacts_dir,
        config_dir=get_env_path("CONFIG_DIR", DEFAULT_CONFIG_DIR, project_root),
        examples_dir=get_env_path("EXAMPLES_DIR", DEFAULT_EXAMPLES_DIR, project_root),
        logs_dir=logs_dir,
        secrets_dir=secrets_dir,
        swagger_config_path=get_env_path("SWAGGER_CONFIG_PATH", f"{DEFAULT_CONFIG_DIR}/{DEFAULT_SWAGGER_FILENAME}", project_root),
        data_control_config_path=get_env_path("DATA_CONTROL_CONFIG_PATH", f"{DEFAULT_CONFIG_DIR}/{DEFAULT_DATA_CONTROL_FILENAME}", project_root),
    )

    ## Ensure runtime directories exist
    _ensure_directories_exist([
        paths.data_dir,
        paths.raw_data_dir,
        paths.active_learning_dir,
        paths.artifacts_dir,
        paths.config_dir,
        paths.examples_dir,
        paths.logs_dir,
        paths.secrets_dir,
    ])

    ## Build runtime section
    runtime = RuntimeConfig(
        environment=environment,
        profile=profile,
        debug=_get_profiled_env_bool("FLASK_DEBUG", environment == "dev", profile),
        log_level=_get_profiled_env("LOG_LEVEL", "INFO", profile),
        flask_host=_get_profiled_env("FLASK_HOST", DEFAULT_FLASK_HOST, profile),
        flask_port=_get_profiled_env_int("FLASK_PORT", DEFAULT_FLASK_PORT, profile),
        max_workers=_get_profiled_env_int("MAX_WORKERS", 4, profile),
        batch_size=_get_profiled_env_int("BATCH_SIZE", 32, profile),
        batch_sleep_seconds=_get_profiled_env_float("BATCH_SLEEP_SECONDS", 0.0, profile),
        request_timeout_seconds=_get_profiled_env_int("REQUEST_TIMEOUT_SECONDS", 120, profile),
        allowed_origins=get_env_list("ALLOWED_ORIGINS", ["*"]),
    )

    ## Build deduplication section
    deduplication = DeduplicationConfig(
        default_similarity_threshold=_get_profiled_env_float("DEFAULT_SIMILARITY_THRESHOLD", 0.85, profile),
        active_learning_enabled=_get_profiled_env_bool("ACTIVE_LEARNING_ENABLED", True, profile),
        save_examples=_get_profiled_env_bool("SAVE_EXAMPLES", True, profile),
        export_format=_validate_export_format(_get_profiled_env("DEFAULT_EXPORT_FORMAT", "json", profile)),
    )

    ## Resolve optional secrets
    secrets = SecretsConfig(
        api_key=_read_secret_value("API_KEY", "API_KEY_FILE", project_root=project_root),
    )

    ## Build final config
    config = AppConfig(
        app_name=get_env_str("APP_NAME", DEFAULT_APP_NAME) or DEFAULT_APP_NAME,
        app_version=get_env_str("APP_VERSION", DEFAULT_APP_VERSION) or DEFAULT_APP_VERSION,
        execution=execution,
        paths=paths,
        runtime=runtime,
        deduplication=deduplication,
        secrets=secrets,
    )

    ## Validate final configuration
    _validate_config(config)
    return config

def load_config() -> AppConfig:
    """
        Backward-compatible alias for configuration loading

        Returns:
            AppConfig instance
    """

    ## Keep compatibility with existing imports
    return get_config()

def build_config() -> AppConfig:
    """
        Backward-compatible config builder

        Returns:
            AppConfig instance
    """

    ## Preserve an additional public entrypoint
    return get_config()

## ============================================================
## PUBLIC SINGLETONS
## ============================================================
CONFIG: AppConfig = get_config()
config = CONFIG