"""ASGI application exposing the Agent Memory Leaderboard contract."""
from __future__ import annotations

import sqlite3

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from memoria.auth import require_authorization
from memoria.config import Settings
from memoria.embeddings import EmbeddingUnavailable
from memoria.runtime import create_runtime_store
from memoria.schemas import AddRequest, AddResponse, HealthResponse, SearchRequest, SearchResponse
from memoria.store import IdempotencyConflict, MemoryStore


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create a fully initialized service instance for production or tests."""

    runtime_settings = settings or Settings.from_env()
    store = create_runtime_store(runtime_settings)

    app = FastAPI(title="memoria", version="0.4.0", docs_url=None, redoc_url=None)
    app.state.settings = runtime_settings
    app.state.store = store

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": {"reason": "invalid request"}},
        )

    @app.exception_handler(IdempotencyConflict)
    async def idempotency_conflict_handler(_: Request, __: IdempotencyConflict) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": {"reason": "request_id already exists with different content"}},
        )

    @app.exception_handler(sqlite3.Error)
    async def sqlite_error_handler(_: Request, __: sqlite3.Error) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": {"reason": "storage is temporarily unavailable"}},
        )

    @app.exception_handler(EmbeddingUnavailable)
    async def embedding_error_handler(_: Request, __: EmbeddingUnavailable) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": {"reason": "semantic retrieval service is temporarily unavailable"}},
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        if not request.app.state.store.healthcheck():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"reason": "storage is not ready"},
            )
        return HealthResponse(status="ok")

    @app.post("/v1/add", response_model=AddResponse)
    def add_memory(request: Request, payload: AddRequest) -> AddResponse:
        require_authorization(request, request.app.state.settings)
        return request.app.state.store.add(payload)

    @app.post("/v1/search", response_model=SearchResponse)
    def search_memories(request: Request, payload: SearchRequest) -> SearchResponse:
        require_authorization(request, request.app.state.settings)
        if payload.top_k > request.app.state.settings.max_top_k:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"reason": "top_k exceeds configured maximum"},
            )
        return SearchResponse(
            data=request.app.state.store.search(
                query=payload.query,
                options=payload.options,
                user_id=payload.user_id,
                top_k=payload.top_k,
            )
        )

    return app
