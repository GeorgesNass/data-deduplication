'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "FastAPI service layer: routes, external swagger UI, payload validation, and pipeline execution."
'''

from __future__ import annotations

## Standard library imports
from typing import Optional

## Third-party imports
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import Response
from fastapi.responses import JSONResponse

## Local imports
from src.core.config import load_swagger_config_path
from src.core.schema import (
    DatasetDeduplicationPayload,
    RecordLinkagePayload,
    TrainModelPayload,
)
from src.pipeline import run_pipeline
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger("core.service")

## ============================================================
## APP FACTORY
## ============================================================
def create_app() -> FastAPI:
    """
        Create and configure the FastAPI application

        High-level workflow:
            1) Configure FastAPI instance (disable built-in docs)
            2) Add CORS middleware
            3) Serve external swagger.yaml for documentation
            4) Register business routes using Pydantic payloads

        Returns:
            Configured FastAPI app instance
    """
    
    app = FastAPI(
        title="Data Deduplication API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    ## CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ## Swagger UI (external swagger.yaml)
    swagger_path = load_swagger_config_path()

    if swagger_path.exists():

        @app.get("/api/doc", include_in_schema=False)
        async def swagger_ui() -> Response:
            """
                Serve Swagger UI pointing to the external OpenAPI YAML

                Returns:
                    Swagger UI HTML
            """
            
            return get_swagger_ui_html(
                openapi_url="/api/openapi.yaml",
                title="API documentation",
            )

        @app.get("/api/openapi.yaml", include_in_schema=False)
        async def openapi_yaml() -> Response:
            """
                Serve the external swagger.yaml as the OpenAPI specification

                Returns:
                    YAML content as HTTP response
            """
            
            content = swagger_path.read_text(encoding="utf-8")
            
            return Response(content=content, media_type="application/yaml")

    else:
        LOGGER.warning("Swagger config not found: %s", swagger_path)

    ## ROUTES (Pydantic-validated)
    @app.post("/train-model")
    async def train_model(payload: TrainModelPayload) -> JSONResponse:
        """
            Train model artifacts (train-on-demand compatible)

            Args:
                payload: Validated training payload

            Returns:
                Standardized API response
        """
        
        result = run_pipeline("trainModel", payload.model_dump())
        
        return JSONResponse(content=result)

    @app.post("/dataset-deduplication")
    async def dataset_deduplication(payload: DatasetDeduplicationPayload) -> JSONResponse:
        """
            Run dataset deduplication and clustering

            Args:
                payload: Validated deduplication payload

            Returns:
                Standardized API response
        """
        
        result = run_pipeline("datasetDeduplicationCluster", payload.model_dump())
        
        return JSONResponse(content=result)

    @app.post("/record-to-dataset-linkage")
    async def record_to_dataset_linkage(payload: RecordLinkagePayload) -> JSONResponse:
        """
            Link a single record to an existing dataset

            Args:
                payload: Validated linkage payload

            Returns:
                Standardized API response
        """
        
        result = run_pipeline("recordDatasetLinkage", payload.model_dump())
        
        return JSONResponse(content=result)

    @app.get("/get-models-info")
    async def get_models_info(
        model_id: Optional[int] = Query(default=1, description="Optional model identifier"),
    ) -> JSONResponse:
        """
            Get meta info about persisted active learning artifacts

            Args:
                model_id: Optional model identifier

            Returns:
                Standardized API response
        """
        
        result = run_pipeline("getModelsInfo", {"model_id": model_id})
        
        return JSONResponse(content=result)

    return app

## ============================================================
## ASGI APP (uvicorn entrypoint)
## ============================================================
app = create_app()