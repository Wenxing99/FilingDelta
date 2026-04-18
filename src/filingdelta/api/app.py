from __future__ import annotations

from fastapi import FastAPI

from filingdelta.api.routes.system import router as system_router
from filingdelta.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Traceable disclosure reading and diff analysis for A-share, "
            "H-share, and ADR filings."
        ),
    )
    app.include_router(system_router)
    return app
