"""Vector Workbench FastAPI entrypoint and application factory."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, benchmarks, diagnostics, inference
from app.core.config import settings
from app.models.database import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup safety check: refuse demo mode in production
    settings.ensure_safe_mode()

    # Record initialization in audit log
    db.log_audit(
        user_id="system",
        action="APP_STARTUP",
        resource="system",
        status="SUCCESS",
        details=f"Vector Workbench {settings.version} initialized in {settings.environment} mode",
    )
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    application = FastAPI(
        title="Vector Workbench",
        description="1-bit LLM ternary inference framework, FFN activation validator, and in-memory runtime server.",
        version=settings.version,
        lifespan=lifespan,
    )

    # CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @application.get("/api/v1/health", tags=["Health"])
    def health_check():
        return {
            "status": "healthy",
            "version": settings.version,
            "environment": settings.environment,
            "demo_mode": settings.demo_mode,
        }

    # Mount API routers
    application.include_router(auth.router, prefix="/api/v1")
    application.include_router(inference.router, prefix="/api/v1")
    application.include_router(benchmarks.router, prefix="/api/v1")
    application.include_router(diagnostics.router, prefix="/api/v1")

    return application


app = create_app()
