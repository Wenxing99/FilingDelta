from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from filingdelta.schemas.demo import (
    DemoChatRequest,
    DemoChatResponse,
    CreateDemoRunRequest,
    DemoDocumentListResponse,
    DemoRunFeedbackActionRequest,
    DemoRunIssueActionRequest,
    DemoDocument,
    DemoRunResponse,
)
from filingdelta.services.demo_documents import (
    get_demo_document_source,
    import_demo_document,
    list_demo_documents,
)
from filingdelta.services.chat_qa import get_chat_qa_service
from filingdelta.services.demo_runs import get_demo_run_manager
from filingdelta.services.kb_financial_facts import KbFinancialFactsChatService


router = APIRouter(prefix="/api/demo", tags=["demo"])
_kb_financial_facts_chat_service = KbFinancialFactsChatService()


@router.get("/documents", response_model=DemoDocumentListResponse)
def demo_documents() -> DemoDocumentListResponse:
    return DemoDocumentListResponse(documents=list_demo_documents())


@router.post("/documents/import", response_model=DemoDocument, status_code=status.HTTP_201_CREATED)
async def import_demo_source(file: UploadFile = File(...)) -> DemoDocument:
    try:
        content = await file.read()
        document = import_demo_document(file.filename or "", content)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    finally:
        await file.close()

    return document


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


@router.post("/chat", response_model=DemoChatResponse)
async def demo_chat(payload: DemoChatRequest) -> DemoChatResponse:
    try:
        source = get_demo_document_source(payload.document_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    kb_answer = _answer_kb_financial_fact_rank(payload)
    if kb_answer is not None:
        return DemoChatResponse(response=kb_answer)

    service = get_chat_qa_service()
    try:
        response = await service.ask(
            document_id=payload.document_id,
            source=source,
            session_id=payload.session_id,
            question=payload.question,
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return DemoChatResponse(response=response)


@router.post("/chat/stream")
async def demo_chat_stream(payload: DemoChatRequest) -> StreamingResponse:
    try:
        source = get_demo_document_source(payload.document_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    kb_answer = _answer_kb_financial_fact_rank(payload)
    if kb_answer is not None:
        return StreamingResponse(
            _stream_static_chat_answer(kb_answer),
            media_type="application/x-ndjson",
        )

    service = get_chat_qa_service()
    return StreamingResponse(
        _stream_demo_chat(payload=payload, source=source, service=service),
        media_type="application/x-ndjson",
    )


def _answer_kb_financial_fact_rank(payload: DemoChatRequest):
    return _kb_financial_facts_chat_service.answer_if_supported(
        document_id=payload.document_id,
        session_id=payload.session_id,
        question=payload.question,
    )


async def _stream_static_chat_answer(answer) -> AsyncIterator[str]:
    yield _encode_stream_event(
        {
            "type": "status",
            "stage": "financial_facts",
            "message": "正在查询结构化事实库...",
        }
    )
    for chunk in _iter_text_deltas(answer.answer):
        yield _encode_stream_event(
            {
                "type": "delta",
                "text": chunk,
            }
        )
        await asyncio.sleep(0.01)
    yield _encode_stream_event(
        {
            "type": "citations",
            "citations": [],
        }
    )
    yield _encode_stream_event(
        {
            "type": "done",
            "response": answer.model_dump(mode="json"),
        }
    )


async def _stream_demo_chat(
    *,
    payload: DemoChatRequest,
    source,
    service,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def publish_status(stage: str, message: str) -> None:
        await queue.put(
            {
                "type": "status",
                "stage": stage,
                "message": message,
            }
        )

    task = asyncio.create_task(
        service.ask(
            document_id=payload.document_id,
            source=source,
            session_id=payload.session_id,
            question=payload.question,
            status_callback=publish_status,
        )
    )

    try:
        while not task.done():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            yield _encode_stream_event(event)

        while not queue.empty():
            yield _encode_stream_event(await queue.get())

        try:
            answer = task.result()
        except ValueError as error:
            yield _encode_stream_event(
                {
                    "type": "error",
                    "message": str(error),
                }
            )
            return
        except Exception:
            yield _encode_stream_event(
                {
                    "type": "error",
                    "message": "问答请求失败。",
                }
            )
            return

        for chunk in _iter_text_deltas(answer.answer):
            yield _encode_stream_event(
                {
                    "type": "delta",
                    "text": chunk,
                }
            )
            await asyncio.sleep(0.01)

        yield _encode_stream_event(
            {
                "type": "citations",
                "citations": [citation.model_dump(mode="json") for citation in answer.citations],
            }
        )
        if answer.telemetry is not None:
            yield _encode_stream_event(
                {
                    "type": "telemetry",
                    "telemetry": answer.telemetry.model_dump(mode="json"),
                }
            )
        yield _encode_stream_event(
            {
                "type": "done",
                "response": answer.model_dump(mode="json"),
            }
        )
    except asyncio.CancelledError:
        task.cancel()
        raise
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


def _encode_stream_event(event: dict[str, object]) -> str:
    return f"{json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n"


def _iter_text_deltas(text: str, *, chunk_size: int = 36) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [stripped[index : index + chunk_size] for index in range(0, len(stripped), chunk_size)]


def _guess_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    return "application/octet-stream"
