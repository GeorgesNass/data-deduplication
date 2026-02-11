'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pydantic schemas for request and response validation (FastAPI)."
'''

from __future__ import annotations

## Standard library imports
from typing import Any, Dict, List, Optional

## Third-party imports
from pydantic import BaseModel, Field

## ============================================================
## BASE PAYLOAD
## ============================================================
class BasePayload(BaseModel):
    """
        Base payload shared by train-model and dataset-deduplication endpoints
    """

    model_id: int = Field(..., description="Deduplication model identifier")

    input_path: Optional[str] = Field(
        None, description="Path to CSV/JSON file"
    )
    raw_dir_filename: Optional[str] = Field(
        None, description="Filename located in data/raw/"
    )
    filename: Optional[str] = Field(
        None, description="Original filename"
    )
    base64_file: Optional[str] = Field(
        None, description="Base64 encoded file content"
    )

    id_column: str = Field("id", description="Primary key column")
    encoding: str = Field("utf-8", description="File encoding")
    limit: Optional[int] = Field(
        None, description="Optional max number of rows"
    )
    num_processes: int = Field(
        1, description="Number of processes"
    )

    enable_active_learning: bool = Field(False)
    use_existing_training: bool = Field(True)
    enable_candidates: bool = Field(True)

    class Config:
        extra = "forbid"


## ============================================================
## TRAIN MODEL
## ============================================================
class TrainModelPayload(BasePayload):
    """
        Payload for POST /train-model
    """
    pass


class TrainModelResponse(BaseModel):
    model_id: int
    records_count: int
    timestamp: str


## ============================================================
## DATASET DEDUPLICATION
## ============================================================

class DatasetDeduplicationPayload(BasePayload):
    """
        Payload for POST /dataset-deduplication
    """

    cluster_threshold: float = Field(
        0.5, description="Clustering similarity threshold"
    )


class DatasetDeduplicationResponse(BaseModel):
    code: str
    type: str
    message: Optional[str] = None
    data: Dict[str, Any]


## ============================================================
## RECORD TO DATASET LINKAGE
## ============================================================
class RecordContent(BaseModel):
    """
        Single record content used for linkage
    """

    family_name_list: List[str]
    first_name_list: List[str]

    sex: Optional[str] = None
    birth_date: Optional[str] = None
    address_complete_list: Optional[List[str]] = None
    emails: Optional[List[str]] = None
    telephones: Optional[List[int]] = None

    class Config:
        extra = "forbid"


class RecordLinkagePayload(BaseModel):
    """
        Payload for POST /record-to-dataset-linkage
    """

    model_id: int
    confidence_filter: float = Field(0.65)
    record_info: RecordContent

    class Config:
        extra = "forbid"


class RecordLinkageResponse(BaseModel):
    code: str
    type: str
    message: Optional[str] = None
    data: Dict[str, Any]


## ============================================================
## MODELS INFO
## ============================================================
class ModelsInfoResponse(BaseModel):
    """
        Response for GET /get-models-info
    """

    code: str
    type: str
    message: Optional[str] = None
    data: Dict[str, Any]
