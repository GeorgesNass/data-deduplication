'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Complete unit test suite covering core, pipeline, service, and model layers (FastAPI)."
'''

from __future__ import annotations

## Standard library imports
from pathlib import Path
from typing import Any, Dict

## Third-party imports
import pytest
from fastapi.testclient import TestClient

## Local imports
from src import pipeline
from src.core.controls import build_error_response, build_success_response, safe_run
from src.core.errors import ApplicationError, ValidationError
from src.core.schema import (
    DatasetDeduplicationPayload,
    RecordLinkagePayload,
    TrainModelPayload,
)
from src.core.service import app as fastapi_app
from src.model import active_learning
from src.model.active_learning import load_active_learning_model, save_active_learning_model
from src.model.cleaning import (
    enrich_emails,
    normalize_birth_date,
    normalize_text,
    parse_nested_list,
)
from src.pipeline import run_pipeline

## ============================================================
## FIXTURES
## ============================================================
@pytest.fixture(scope="module")
def client() -> TestClient:
    """
        Build a FastAPI TestClient

        Returns:
            FastAPI TestClient instance
    """

    return TestClient(fastapi_app)

## ============================================================
## CORE.ERRORS TESTS
## ============================================================
def test_application_error_to_dict() -> None:
    """
        Validate serialization of ApplicationError

        Returns:
            None
    """

    err = ApplicationError(
        message="Something went wrong",
        code="500",
        error_type="TEST_ERROR",
        details="details here",
    )

    payload = err.to_dict()

    assert payload["code"] == "500"
    assert payload["type"] == "TEST_ERROR"
    assert payload["message"] == "Something went wrong"
    assert payload["details"] == "details here"

def test_validation_error_defaults() -> None:
    """
        ValidationError should use code 400 and VALIDATION_ERROR type

        Returns:
            None
    """

    err = ValidationError(message="Invalid input")

    assert err.code == "400"
    assert err.error_type == "VALIDATION_ERROR"

## ============================================================
## CORE.CONTROLS TESTS
## ============================================================
def test_build_success_response() -> None:
    """
        Validate standardized success response builder

        Returns:
            None
    """

    payload = build_success_response(message="OK", data={"x": 1})

    assert payload["code"] == "200"
    assert payload["type"] == "SUCCESS"
    assert payload["data"]["x"] == 1

def test_build_error_response() -> None:
    """
        Validate standardized error response builder

        Returns:
            None
    """

    payload = build_error_response(
        code="400",
        error_type="VALIDATION_ERROR",
        message="Bad request",
        details="missing field",
    )

    assert payload["code"] == "400"
    assert payload["type"] == "VALIDATION_ERROR"
    assert payload["details"] == "missing field"

def test_safe_run_success() -> None:
    """
        safe_run should return handler output when no exception occurs

        Returns:
            None
    """

    def handler(x: int) -> Dict[str, Any]:
        """
            Build a simple payload for safe_run success path

            Args:
                x: Integer input value

            Returns:
                Dictionary containing x
        """

        return {"x": x}

    result = safe_run(handler, 10)

    assert result["x"] == 10

def test_safe_run_validation_error() -> None:
    """
        safe_run should normalize ValidationError to error payload

        Returns:
            None
    """

    def handler() -> Dict[str, Any]:
        """
            Raise a validation error for safe_run testing

            Raises:
                ValidationError: Always raised in this test helper

            Returns:
                Never returns
        """

        raise ValidationError(message="Invalid input")

    result = safe_run(handler)

    assert result["code"] == "400"
    assert result["type"] == "VALIDATION_ERROR"

def test_safe_run_unexpected_error() -> None:
    """
        safe_run should handle unexpected exceptions

        Returns:
            None
    """

    def handler() -> Dict[str, Any]:
        """
            Raise an unexpected runtime error for safe_run testing

            Raises:
                RuntimeError: Always raised in this test helper

            Returns:
                Never returns
        """

        raise RuntimeError("boom")

    result = safe_run(handler)

    assert result["code"] in {"500", 500}

## ============================================================
## SCHEMA (PYDANTIC) TESTS
## ============================================================
def test_train_model_payload_validation() -> None:
    """
        TrainModelPayload should validate required fields

        Returns:
            None
    """

    payload = TrainModelPayload(model_id=1)

    assert payload.model_id == 1
    assert payload.encoding == "utf-8"

def test_dataset_deduplication_payload_defaults() -> None:
    """
        DatasetDeduplicationPayload should set defaults correctly

        Returns:
            None
    """

    payload = DatasetDeduplicationPayload(model_id=1)

    assert payload.cluster_threshold == 0.5
    assert payload.num_processes == 1

def test_record_linkage_payload_requires_names() -> None:
    """
        RecordLinkagePayload should require family_name_list and first_name_list

        Returns:
            None
    """

    payload = RecordLinkagePayload(
        model_id=1,
        record_info={
            "family_name_list": ["Doe"],
            "first_name_list": ["John"],
        },
    )

    assert payload.model_id == 1
    assert payload.record_info.family_name_list == ["Doe"]

## ============================================================
## MODEL.CLEANING TESTS
## ============================================================
def test_normalize_text_basic() -> None:
    """
        normalize_text should normalize accents and casing

        Returns:
            None
    """

    assert normalize_text("ÉLÈVE  ") == "eleve"

def test_normalize_text_empty() -> None:
    """
        normalize_text should handle empty input

        Returns:
            None
    """

    assert normalize_text("") == ""

def test_parse_nested_list() -> None:
    """
        parse_nested_list should split and normalize values

        Returns:
            None
    """

    result = parse_nested_list("John@@@Doe@@@John")

    assert "john" in result
    assert "doe" in result
    assert len(result) == 2

def test_enrich_emails() -> None:
    """
        enrich_emails should add local-part tokens

        Returns:
            None
    """

    emails = ["John.Doe@gmail.com"]
    enriched = enrich_emails(emails)

    assert "john.doe@gmail.com" in enriched
    assert "john.doe" in enriched

def test_enrich_emails_empty() -> None:
    """
        enrich_emails should handle empty list

        Returns:
            None
    """

    assert enrich_emails([]) == []

def test_normalize_birth_date() -> None:
    """
        normalize_birth_date should drop time suffix

        Returns:
            None
    """

    assert normalize_birth_date("1990-01-01 00:00:00") == "1990-01-01"

## ============================================================
## MODEL.ACTIVE_LEARNING TESTS
## ============================================================
def test_active_learning_save_and_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
        Active learning model should be saved and reloaded correctly

        Args:
            tmp_path: Temporary directory
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            None
    """

    monkeypatch.setattr(active_learning, "ACTIVE_LEARNING_DIR", tmp_path)

    model_id = 99
    payload = {"foo": "bar"}

    save_active_learning_model(model_id, payload)
    loaded = load_active_learning_model(model_id)

    assert loaded == payload

## ============================================================
## PIPELINE TESTS
## ============================================================
def test_run_pipeline_unknown_function() -> None:
    """
        Unknown pipeline function should return an error payload

        Returns:
            None
    """

    result = run_pipeline("unknownFunction", payload={})

    assert result["code"] == "404"

def test_run_pipeline_none_payload() -> None:
    """
        Pipeline should handle None payload

        Returns:
            None
    """

    result = run_pipeline("unknownFunction", payload=None)

    assert "code" in result

def test_run_pipeline_mocked_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
        Pipeline should route to mocked handler correctly

        Args:
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            None
    """

    def fake_handler(_: Any) -> Dict[str, Any]:
        """
            Return a fake pipeline result without heavy processing

            Args:
                _: Ignored payload

            Returns:
                Dictionary with a fake marker
        """

        return {"fake": True}

    monkeypatch.setitem(
        pipeline.FUNCTION_DISPATCH["datasetDeduplicationCluster"],
        "handler",
        fake_handler,
    )

    result = run_pipeline("datasetDeduplicationCluster", payload={"x": 1})

    assert result["code"] == "200"
    assert result["data"]["result"]["fake"] is True

## ============================================================
## SERVICE (FASTAPI) TESTS
## ============================================================
def test_get_models_info_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
        GET /get-models-info should return a standardized response

        Args:
            client: FastAPI TestClient
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            None
    """

    def fake_handler(_: Any) -> Dict[str, Any]:
        """
            Return fake model information payload

            Args:
                _: Ignored payload

            Returns:
                Dictionary with fake model state
        """

        return {"model_id": 1, "state": {"ok": True}}

    monkeypatch.setitem(
        pipeline.FUNCTION_DISPATCH["getModelsInfo"],
        "handler",
        fake_handler,
    )

    response = client.get("/get-models-info")

    assert response.status_code == 200

    payload = response.json()

    assert payload["code"] == "200"
    assert payload["type"] == "SUCCESS"

def test_dataset_deduplication_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
        POST /dataset-deduplication should validate payload and return a response

        Args:
            client: FastAPI TestClient
            monkeypatch: Pytest monkeypatch fixture

        Returns:
            None
    """

    def fake_handler(_: Any) -> Dict[str, Any]:
        """
            Return fake deduplication clusters payload

            Args:
                _: Ignored payload

            Returns:
                Dictionary with fake clusters
        """

        return {"clusters": [{"id": 1, "members": [1, 2]}]}

    monkeypatch.setitem(
        pipeline.FUNCTION_DISPATCH["datasetDeduplicationCluster"],
        "handler",
        fake_handler,
    )

    response = client.post(
        "/dataset-deduplication",
        json={"model_id": 1},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["code"] == "200"
    assert payload["type"] == "SUCCESS"

def test_dataset_deduplication_invalid_payload(client: TestClient) -> None:
    """
        Endpoint should reject invalid payload

        Args:
            client: FastAPI TestClient

        Returns:
            None
    """

    response = client.post("/dataset-deduplication", json={})

    assert response.status_code in {400, 422}