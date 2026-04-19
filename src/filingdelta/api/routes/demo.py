from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from filingdelta.schemas.demo import (
    CreateDemoRunRequest,
    DemoDocumentListResponse,
    DemoRunFeedbackActionRequest,
    DemoRunIssueActionRequest,
    DemoRunResponse,
)
from filingdelta.services.demo_documents import (
    get_demo_document_source,
    list_demo_documents,
)
from filingdelta.services.demo_runs import get_demo_run_manager


router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/documents", response_model=DemoDocumentListResponse)
def demo_documents() -> DemoDocumentListResponse:
    return DemoDocumentListResponse(documents=list_demo_documents())


@router.get("/documents/{document_id}/source")
def demo_document_source(document_id: str) -> FileResponse:
    try:
        source = get_demo_document_source(document_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    media_type = _guess_media_type(source.source_path)
    return FileResponse(
        source.source_path,
        media_type=media_type,
        filename=source.source_path.name,
        content_disposition_type="inline",
    )


@router.post("/runs", response_model=DemoRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_demo_run(payload: CreateDemoRunRequest) -> DemoRunResponse:
    try:
        source = get_demo_document_source(payload.document_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    manager = get_demo_run_manager()
    run = await manager.create_run(document_id=payload.document_id, source=source)
    return DemoRunResponse(run=run)


@router.get("/runs/{run_id}", response_model=DemoRunResponse)
async def get_demo_run(run_id: str) -> DemoRunResponse:
    manager = get_demo_run_manager()
    try:
        run = await manager.get_run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return DemoRunResponse(run=run)


@router.post("/runs/{run_id}/issues/approve", response_model=DemoRunResponse)
async def approve_demo_run_issue(run_id: str, payload: DemoRunIssueActionRequest) -> DemoRunResponse:
    manager = get_demo_run_manager()
    try:
        run = await manager.approve_issue(run_id=run_id, item_key=payload.item_key)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return DemoRunResponse(run=run)


@router.post("/runs/{run_id}/issues/rerun", response_model=DemoRunResponse)
async def rerun_demo_run_issue(run_id: str, payload: DemoRunIssueActionRequest) -> DemoRunResponse:
    manager = get_demo_run_manager()
    try:
        run = await manager.rerun_issue(run_id=run_id, item_key=payload.item_key)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return DemoRunResponse(run=run)


@router.post("/runs/{run_id}/feedback", response_model=DemoRunResponse)
async def rerun_demo_run_feedback(run_id: str, payload: DemoRunFeedbackActionRequest) -> DemoRunResponse:
    manager = get_demo_run_manager()
    try:
        run = await manager.rerun_feedback(run_id=run_id, feedback_category=payload.feedback_category)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return DemoRunResponse(run=run)


def _guess_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    return "application/octet-stream"
