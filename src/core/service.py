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
from fastapi import FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer

## JWT / SECURITY IMPORTS
from core.auth import (
    login_user,
    refresh_access_token,
    logout_user,
    get_current_active_user,
)
from core.security import (
    JWTMiddleware,
    require_roles,
)

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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

## Fake DB (TO REPLACE)
fake_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$examplehash",
        "roles": ["admin"],
        "scopes": ["all"],
        "is_active": True,
    }
}

## ============================================================
## APP FACTORY
## ============================================================
def create_app() -> FastAPI:
    """
        Create and configure the FastAPI application

        Returns:
            FastAPI app
    """
    
    app = FastAPI(
        title="Data Deduplication API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    ## MIDDLEWARE CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ## JWT middleware
    app.add_middleware(JWTMiddleware)

    ## AUTH ENDPOINTS
    @app.post("/login")
    async def login(data: dict):
        """
            Authenticate user and return tokens

            Args:
                data: username/password

            Returns:
                JWT tokens
        """
        ## Login
        return login_user(data["username"], data["password"], fake_db)

    @app.post("/refresh")
    async def refresh(data: dict):
        """
            Refresh access token

            Args:
                data: refresh_token

            Returns:
                New tokens
        """
        return refresh_access_token(data["refresh_token"])

    @app.post("/logout")
    async def logout(token: str = Depends(oauth2_scheme)):
        """
            Logout user

            Args:
                token: JWT token

            Returns:
                Status
        """
        
        logout_user(token)
        return {"status": "logged_out"}

    ## SWAGGER
    swagger_path = load_swagger_config_path()

    if swagger_path.exists():

        @app.get("/api/doc", include_in_schema=False)
        async def swagger_ui() -> Response:
            """
                Serve Swagger UI

                Returns:
                    HTML response
            """
            
            ## Serve Swagger UI
            return get_swagger_ui_html(
                openapi_url="/api/openapi.yaml",
                title="API documentation",
            )

        @app.get("/api/openapi.yaml", include_in_schema=False)
        async def openapi_yaml() -> Response:
            """
                Serve OpenAPI YAML

                Returns:
                    YAML response
            """
            
            ## Read YAML
            content = swagger_path.read_text(encoding="utf-8")

            return Response(content=content, media_type="application/yaml")

    else:
        LOGGER.warning("Swagger config not found: %s", swagger_path)

    ## ROUTES
    @app.post("/train-model")
    async def train_model(
        payload: TrainModelPayload,
        user=Depends(require_roles(["admin"])),
    ) -> JSONResponse:
        """
            Train model

            Args:
                payload: Training payload
                user: Authenticated user

            Returns:
                API response
        """

        result = run_pipeline("trainModel", payload.model_dump())

        return JSONResponse(content=result)

    @app.post("/dataset-deduplication")
    async def dataset_deduplication(
        payload: DatasetDeduplicationPayload,
        user=Depends(get_current_active_user),
    ) -> JSONResponse:
        """
            Run dataset deduplication

            Args:
                payload: Deduplication payload
                user: Authenticated user

            Returns:
                API response
        """

        result = run_pipeline("datasetDeduplicationCluster", payload.model_dump())

        return JSONResponse(content=result)

    @app.post("/record-to-dataset-linkage")
    async def record_to_dataset_linkage(
        payload: RecordLinkagePayload,
        user=Depends(get_current_active_user),
    ) -> JSONResponse:
        """
            Link record to dataset

            Args:
                payload: Linkage payload
                user: Authenticated user

            Returns:
                API response
        """
        
        result = run_pipeline("recordDatasetLinkage", payload.model_dump())

        return JSONResponse(content=result)

    @app.get("/get-models-info")
    async def get_models_info(
        model_id: Optional[int] = Query(default=1),
        user=Depends(get_current_active_user),
    ) -> JSONResponse:
        """
            Get model info

            Args:
                model_id: Model identifier
                user: Authenticated user

            Returns:
                API response
        """
        
        result = run_pipeline("getModelsInfo", {"model_id": model_id})

        return JSONResponse(content=result)

    return app

## ============================================================
## ASGI APP
## ============================================================
app = create_app()