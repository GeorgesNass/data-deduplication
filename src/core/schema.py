'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic schemas for data deduplication API, training, linkage, batch processing, metrics, and pipeline contracts."
'''

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover
    BaseSettings = BaseModel  # type: ignore[misc, assignment]
    SettingsConfigDict = dict  # type: ignore[misc, assignment]

## ============================================================
## COMMON TYPES AND PATTERNS
## ============================================================
JobStatusName = Literal["pending", "running", "success", "failed", "cancelled"]
TaskTypeName = Literal[
    "train_model",
    "dataset_deduplication",
    "record_linkage",
    "evaluation",
    "export",
]
ResponseTypeName = Literal["success", "error", "warning", "info"]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
FileFormatName = Literal["csv", "json", "parquet"]

SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._:/\-]+$")
SAFE_FILE_PATTERN = re.compile(r"^[a-zA-Z0-9._/\-]+$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/=\s]+$")

## ============================================================
## BASE SCHEMAS
## ============================================================
class BaseSchema(BaseModel):
    """
        Base schema with shared validation and serialization helpers

        Returns:
            A reusable Pydantic base model
    """

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
        "str_strip_whitespace": True,
    }

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the model to a Python dictionary

            Returns:
                Serialized model as dictionary
        """

        return self.model_dump()

    def to_json(self) -> str:
        """
            Convert the model to a JSON string

            Returns:
                Serialized model as JSON
        """

        return self.model_dump_json()

    def to_record(self) -> dict[str, Any]:
        """
            Convert the model to a row-oriented dictionary

            Returns:
                Flat dictionary representation
        """

        return self.model_dump(mode="json")

    def to_pandas(self) -> Any:
        """
            Convert the model to a one-row pandas DataFrame

            Returns:
                A pandas DataFrame with one row
        """

        ## Import pandas lazily to avoid a hard dependency at import time
        import pandas as pd

        return pd.DataFrame([self.to_record()])

class WarningMixin(BaseSchema):
    """
        Mixin exposing warnings in response payloads

        Args:
            warnings: Warning messages list
    """

    warnings: list[str] = Field(default_factory=list)

## ============================================================
## SETTINGS AND CONFIG SCHEMAS
## ============================================================
@dataclass(frozen=True)
class DeduplicationRuntimeConfig:
    """
        Typed runtime configuration for deduplication workflows

        Args:
            default_model_id: Default model identifier
            default_encoding: Default file encoding
            max_limit: Maximum allowed row limit
            default_num_processes: Default worker count
            enable_active_learning: Whether active learning is enabled by default
    """

    default_model_id: int
    default_encoding: str
    max_limit: int
    default_num_processes: int
    enable_active_learning: bool

    def to_dict(self) -> dict[str, Any]:
        """
            Convert the dataclass to a dictionary

            Returns:
                Serialized dataclass as dictionary
        """

        return asdict(self)

class AppSettings(BaseSettings):
    """
        Settings model for data-deduplication

        Args:
            app_name: Application name
            environment: Runtime environment
            default_model_id: Default deduplication model id
            default_encoding: Default file encoding
            max_limit: Maximum allowed row limit
            max_num_processes: Maximum allowed worker count
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="DEDUP_",
        case_sensitive=False,
    )

    app_name: str = "data-deduplication"
    environment: str = "dev"
    default_model_id: int = Field(default=1, ge=1)
    default_encoding: str = "utf-8"
    max_limit: int = Field(default=1_000_000, ge=1)
    max_num_processes: int = Field(default=64, ge=1, le=1024)

class PipelineConfig(BaseSchema):
    """
        Pipeline execution configuration schema

        Args:
            job_name: Pipeline job name
            batch_size: Batch size
            max_workers: Number of workers
            retry_count: Retry count
            overwrite: Whether outputs can be overwritten
    """

    job_name: str = Field(default="data-deduplication-job", min_length=1)
    batch_size: int = Field(default=1000, ge=1, le=1_000_000)
    max_workers: int = Field(default=1, ge=1, le=512)
    retry_count: int = Field(default=0, ge=0, le=20)
    overwrite: bool = False

## ============================================================
## COMMON OPERATIONAL SCHEMAS
## ============================================================
class HealthResponse(BaseSchema):
    """
        Healthcheck response model

        Args:
            status: Service status
            service: Service name
            version: Application version
    """

    status: str = Field(default="ok", min_length=1)
    service: str = Field(default="data-deduplication", min_length=1)
    version: str = Field(default="1.0.0", min_length=1)

class ErrorResponse(BaseSchema):
    """
        Standard API error response

        Args:
            error: Normalized error code
            message: Human-readable message
            origin: Component where the error happened
            details: Diagnostic details
            request_id: Optional request correlation id
    """

    error: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    origin: str = Field(default="unknown", min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default="n/a", min_length=1)

class ApiEnvelope(BaseSchema):
    """
        Standard API envelope schema

        Args:
            code: Business or HTTP-like code
            type: Response type
            message: Human-readable message
            data: Payload data
    """

    code: str = Field(..., min_length=1)
    type: ResponseTypeName
    message: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

class StatusResponse(BaseSchema):
    """
        Generic status response schema

        Args:
            status: Current status
            message: Optional message
            progress: Optional progress between 0 and 100
            metadata: Optional metadata
    """

    status: str = Field(..., min_length=1)
    message: str = Field(default="")
    progress: float | None = Field(default=None, ge=0.0, le=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

class StructuredLogEvent(BaseSchema):
    """
        Structured log schema

        Args:
            level: Log level
            event: Event name
            message: Human-readable message
            logger_name: Logger name
            context: Additional context
    """

    level: LogLevelName
    event: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    logger_name: str = Field(default="data-deduplication", min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)

class QueueEvent(BaseSchema):
    """
        Message queue or event bus schema

        Args:
            event_id: Unique event identifier
            event_type: Event type
            source: Event source
            payload: Event payload
    """

    event_id: str = Field(..., min_length=1)
    event_type: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "event_type", "source")
    @classmethod
    def validate_safe_names(cls, value: str) -> str:
        """
            Validate safe identifier-like strings

            Args:
                value: Candidate identifier string

            Returns:
                The validated identifier string

            Raises:
                ValueError: If the value contains unsupported characters
        """

        ## Ensure identifiers remain API and filesystem friendly
        if not SAFE_NAME_PATTERN.match(value):
            raise ValueError("value contains unsupported characters")
        return value

class MetricPoint(BaseSchema):
    """
        Monitoring metric point schema

        Args:
            name: Metric name
            value: Metric value
            unit: Optional metric unit
            tags: Optional metric tags
    """

    name: str = Field(..., min_length=1)
    value: float
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

class MonitoringResponse(WarningMixin):
    """
        Monitoring response schema

        Args:
            metrics: Metric points list
            summary: Aggregated summary
    """

    metrics: list[MetricPoint] = Field(default_factory=list)
    summary: dict[str, float] = Field(default_factory=dict)

## ============================================================
## DATASET AND PIPELINE SCHEMAS
## ============================================================
class DatasetRecord(BaseSchema):
    """
        Generic dataset record schema

        Args:
            record_id: Record identifier
            payload: Raw row payload
            metadata: Optional metadata
    """

    record_id: str = Field(..., min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, value: str) -> str:
        """
            Validate record identifier format

            Args:
                value: Candidate record identifier

            Returns:
                The validated record identifier

            Raises:
                ValueError: If the identifier contains unsupported characters
        """

        ## Keep record identifiers safe for logs and exports
        if not SAFE_NAME_PATTERN.match(value):
            raise ValueError("record_id contains unsupported characters")
        return value

class DatasetInput(BaseSchema):
    """
        Dataset input schema

        Args:
            name: Dataset name
            records: Dataset records
    """

    name: str = Field(..., min_length=1)
    records: list[DatasetRecord] = Field(default_factory=list)

    @field_validator("records")
    @classmethod
    def validate_non_empty_records(
        cls, value: list[DatasetRecord]
    ) -> list[DatasetRecord]:
        """
            Validate that the dataset contains at least one record

            Args:
                value: Dataset records

            Returns:
                The validated records list

            Raises:
                ValueError: If the records list is empty
        """

        ## Prevent empty dataset payloads
        if not value:
            raise ValueError("records must contain at least one item")
        return value

class DatasetOutput(BaseSchema):
    """
        Dataset output schema

        Args:
            name: Dataset name
            row_count: Number of rows
            artifacts: Generated artifacts list
    """

    name: str = Field(..., min_length=1)
    row_count: int = Field(..., ge=0)
    artifacts: list[str] = Field(default_factory=list)

    @field_validator("artifacts")
    @classmethod
    def validate_artifacts(cls, value: list[str]) -> list[str]:
        """
            Validate artifact path strings

            Args:
                value: Candidate artifact path list

            Returns:
                The validated artifact path list

            Raises:
                ValueError: If one artifact path contains unsupported characters
        """

        ## Ensure artifact paths remain safe to persist and log
        for artifact_path in value:
            if not SAFE_FILE_PATTERN.match(artifact_path):
                raise ValueError("artifacts contain unsupported path characters")
        return value

class PipelineTask(BaseSchema):
    """
        Pipeline task schema

        Args:
            task_id: Task identifier
            task_type: Task type
            status: Task status
            progress: Task progress percentage
            input_payload: Task input payload
            output_payload: Task output payload
    """

    task_id: str = Field(..., min_length=1)
    task_type: TaskTypeName
    status: JobStatusName = "pending"
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)

class PipelineJob(BaseSchema):
    """
        Pipeline job schema

        Args:
            job_id: Job identifier
            status: Job status
            tasks: Job tasks
            progress: Job progress percentage
            metadata: Job metadata
    """

    job_id: str = Field(..., min_length=1)
    status: JobStatusName = "pending"
    tasks: list[PipelineTask] = Field(default_factory=list)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_job_progress(self) -> "PipelineJob":
        """
            Validate progress consistency between the job and its tasks

            Returns:
                The validated pipeline job

            Raises:
                ValueError: If job progress is below the minimum task progress
        """

        ## Keep parent progress coherent with child task progress
        if self.tasks and self.progress < min(task.progress for task in self.tasks):
            raise ValueError("job progress cannot be below the minimum task progress")
        return self

## ============================================================
## FILE INPUT PAYLOADS
## ============================================================
class BasePayload(BaseSchema):
    """
        Base payload shared by train-model and dataset-deduplication endpoints

        Args:
            model_id: Deduplication model identifier
            input_path: Path to CSV or JSON file
            raw_dir_filename: Filename located in data/raw
            filename: Original filename
            base64_file: Base64-encoded file content
            id_column: Primary key column
            encoding: File encoding
            limit: Optional maximum number of rows
            num_processes: Number of processes
            enable_active_learning: Whether active learning is enabled
            use_existing_training: Whether existing training is reused
            enable_candidates: Whether candidate generation is enabled
    """

    model_id: int = Field(..., description="Deduplication model identifier", ge=1)
    input_path: str | None = Field(None, description="Path to CSV/JSON file")
    raw_dir_filename: str | None = Field(
        None, description="Filename located in data/raw/"
    )
    filename: str | None = Field(None, description="Original filename")
    base64_file: str | None = Field(None, description="Base64 encoded file content")
    id_column: str = Field("id", description="Primary key column", min_length=1)
    encoding: str = Field("utf-8", description="File encoding", min_length=1)
    limit: int | None = Field(None, description="Optional max number of rows", ge=1)
    num_processes: int = Field(1, description="Number of processes", ge=1, le=1024)
    enable_active_learning: bool = Field(False)
    use_existing_training: bool = Field(True)
    enable_candidates: bool = Field(True)

    @field_validator("input_path", "raw_dir_filename", "filename")
    @classmethod
    def validate_optional_paths(cls, value: str | None) -> str | None:
        """
            Validate optional path-like string fields

            Args:
                value: Candidate path-like value

            Returns:
                The validated value or None

            Raises:
                ValueError: If the path contains unsupported characters
        """

        ## Keep incoming path-like values safe for filesystem operations
        if value is not None and not SAFE_FILE_PATTERN.match(value):
            raise ValueError("path-like fields contain unsupported characters")
        return value

    @field_validator("id_column")
    @classmethod
    def validate_id_column(cls, value: str) -> str:
        """
            Validate primary key column name

            Args:
                value: Candidate column name

            Returns:
                The validated column name

            Raises:
                ValueError: If the column name contains unsupported characters
        """

        ## Restrict column names to safe identifier-like characters
        if not SAFE_NAME_PATTERN.match(value):
            raise ValueError("id_column contains unsupported characters")
        return value

    @field_validator("base64_file")
    @classmethod
    def validate_base64_file(cls, value: str | None) -> str | None:
        """
            Validate optional base64 file content

            Args:
                value: Candidate base64 string

            Returns:
                The validated base64 string or None

            Raises:
                ValueError: If the content format is invalid
        """

        ## Apply lightweight base64-like validation only
        if value is not None and not BASE64_PATTERN.match(value):
            raise ValueError("base64_file is not a valid base64-like string")
        return value

    @model_validator(mode="after")
    def validate_input_sources(self) -> "BasePayload":
        """
            Validate file source fields consistency

            Returns:
                The validated payload

            Raises:
                ValueError: If no input source is provided
        """

        ## Require at least one input source among supported file inputs
        if not any(
            [
                self.input_path,
                self.raw_dir_filename,
                self.filename,
                self.base64_file,
            ]
        ):
            raise ValueError(
                "at least one of input_path, raw_dir_filename, filename or "
                "base64_file must be provided"
            )

        return self

## ============================================================
## TRAIN MODEL
## ============================================================
class TrainModelPayload(BasePayload):
    """
        Payload for POST /train-model

        Args:
            model_id: Deduplication model identifier
            input_path: Path to CSV or JSON file
            raw_dir_filename: Filename located in data/raw
            filename: Original filename
            base64_file: Base64-encoded file content
            id_column: Primary key column
            encoding: File encoding
            limit: Optional maximum number of rows
            num_processes: Number of processes
            enable_active_learning: Whether active learning is enabled
            use_existing_training: Whether existing training is reused
            enable_candidates: Whether candidate generation is enabled
    """

class TrainModelResponse(BaseSchema):
    """
        Response schema for model training

        Args:
            model_id: Trained model identifier
            records_count: Number of records used during training
            timestamp: Training completion timestamp
    """

    model_id: int = Field(..., ge=1)
    records_count: int = Field(..., ge=0)
    timestamp: str = Field(..., min_length=1)

## ============================================================
## DATASET DEDUPLICATION
## ============================================================
class DatasetDeduplicationPayload(BasePayload):
    """
        Payload for POST /dataset-deduplication

        Args:
            model_id: Deduplication model identifier
            input_path: Path to CSV or JSON file
            raw_dir_filename: Filename located in data/raw
            filename: Original filename
            base64_file: Base64-encoded file content
            id_column: Primary key column
            encoding: File encoding
            limit: Optional maximum number of rows
            num_processes: Number of processes
            enable_active_learning: Whether active learning is enabled
            use_existing_training: Whether existing training is reused
            enable_candidates: Whether candidate generation is enabled
            cluster_threshold: Clustering similarity threshold
    """

    cluster_threshold: float = Field(
        0.5,
        description="Clustering similarity threshold",
        ge=0.0,
        le=1.0,
    )

class DatasetDeduplicationResponse(ApiEnvelope):
    """
        Response schema for dataset deduplication

        Args:
            code: Business or HTTP-like code
            type: Response type
            message: Human-readable message
            data: Payload data
    """

## ============================================================
## RECORD TO DATASET LINKAGE
## ============================================================
class RecordContent(BaseSchema):
    """
        Single record content used for linkage

        Args:
            family_name_list: Candidate family names
            first_name_list: Candidate first names
            sex: Optional sex field
            birth_date: Optional birth date
            address_complete_list: Optional full addresses
            emails: Optional emails list
            telephones: Optional phone numbers list
    """

    family_name_list: list[str]
    first_name_list: list[str]
    sex: str | None = None
    birth_date: str | None = None
    address_complete_list: list[str] | None = None
    emails: list[str] | None = None
    telephones: list[int] | None = None

    @field_validator("family_name_list", "first_name_list")
    @classmethod
    def validate_required_name_lists(cls, value: list[str]) -> list[str]:
        """
            Validate required name lists

            Args:
                value: Candidate names list

            Returns:
                The validated names list

            Raises:
                ValueError: If the list is empty or contains empty values
        """

        ## Require at least one non-empty name value
        if not value:
            raise ValueError("name lists must contain at least one item")

        for item in value:
            if not item.strip():
                raise ValueError("name lists must not contain empty values")

        return value

    @field_validator("address_complete_list")
    @classmethod
    def validate_address_list(
        cls, value: list[str] | None
    ) -> list[str] | None:
        """
            Validate optional address list

            Args:
                value: Candidate addresses list

            Returns:
                The validated addresses list or None

            Raises:
                ValueError: If an address value is empty
        """

        ## Validate optional addresses without over-constraining format
        if value is None:
            return value

        for item in value:
            if not item.strip():
                raise ValueError("address_complete_list must not contain empty values")

        return value

    @field_validator("emails")
    @classmethod
    def validate_emails(cls, value: list[str] | None) -> list[str] | None:
        """
            Validate optional emails list

            Args:
                value: Candidate email list

            Returns:
                The validated email list or None

            Raises:
                ValueError: If one email is invalid
        """

        ## Validate each optional email value
        if value is None:
            return value

        for item in value:
            if not EMAIL_PATTERN.match(item):
                raise ValueError("emails contains an invalid email address")

        return value

    @field_validator("birth_date")
    @classmethod
    def validate_birth_date(cls, value: str | None) -> str | None:
        """
            Validate optional birth date format

            Args:
                value: Candidate birth date

            Returns:
                The validated birth date or None

            Raises:
                ValueError: If the date format is invalid
        """

        ## Accept only ISO-like YYYY-MM-DD dates
        if value is not None and not DATE_PATTERN.match(value):
            raise ValueError("birth_date must match YYYY-MM-DD")
        return value

class RecordLinkagePayload(BaseSchema):
    """
        Payload for POST /record-to-dataset-linkage

        Args:
            model_id: Deduplication model identifier
            confidence_filter: Confidence threshold for filtering matches
            record_info: Single record content to link
    """

    model_id: int = Field(..., ge=1)
    confidence_filter: float = Field(0.65, ge=0.0, le=1.0)
    record_info: RecordContent

class RecordLinkageResponse(ApiEnvelope):
    """
        Response schema for record-to-dataset linkage

        Args:
            code: Business or HTTP-like code
            type: Response type
            message: Human-readable message
            data: Payload data
    """

## ============================================================
## MODELS INFO
## ============================================================
class ModelsInfoResponse(ApiEnvelope):
    """
        Response schema for GET /get-models-info

        Args:
            code: Business or HTTP-like code
            type: Response type
            message: Human-readable message
            data: Payload data
    """

## ============================================================
## OPTIONAL BATCH, EVALUATION, AND SUMMARY SCHEMAS
## ============================================================
class DeduplicationPair(BaseSchema):
    """
        Pairwise deduplication match schema

        Args:
            left_id: Left-side record id
            right_id: Right-side record id
            score: Similarity score
            is_duplicate: Whether the pair is considered duplicate
    """

    left_id: str = Field(..., min_length=1)
    right_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    is_duplicate: bool

    @field_validator("left_id", "right_id")
    @classmethod
    def validate_pair_ids(cls, value: str) -> str:
        """
            Validate pair record identifiers

            Args:
                value: Candidate record identifier

            Returns:
                The validated record identifier

            Raises:
                ValueError: If the identifier contains unsupported characters
        """

        ## Keep pair identifiers safe for logs and exports
        if not SAFE_NAME_PATTERN.match(value):
            raise ValueError("pair identifiers contain unsupported characters")
        return value

class DeduplicationMetrics(BaseSchema):
    """
        Deduplication metrics schema

        Args:
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            duplicates_found: Number of duplicates found
    """

    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    duplicates_found: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_metric_consistency(self) -> "DeduplicationMetrics":
        """
            Validate deduplication metric consistency

            Returns:
                The validated metrics object
        """

        ## Keep the schema extensible while accepting externally computed metrics
        return self