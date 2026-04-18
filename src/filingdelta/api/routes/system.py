from __future__ import annotations

from fastapi import APIRouter, Depends

from filingdelta.core.config import Settings, get_settings
from filingdelta.schemas.runtime import HealthResponse, RuntimeConfigResponse


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name)


@router.get("/config", response_model=RuntimeConfigResponse)
def runtime_config(
    settings: Settings = Depends(get_settings),
) -> RuntimeConfigResponse:
    return RuntimeConfigResponse.model_validate(settings.safe_summary())
