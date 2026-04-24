from __future__ import annotations

import asyncio
from pathlib import Path

import filingdelta.services.demo_runs as demo_runs
from filingdelta.core.config import Settings
from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import FilingDocType, FilingSource, Market, ParserKind
from filingdelta.schemas.workflow import SingleFilingWorkflowResult
from filingdelta.services.demo_runs import DemoRunManager


class _BlockingChatService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[tuple[str, FilingSource]] = []

    async def prewarm_document(self, *, document_id: str, source: FilingSource) -> None:
        self.calls.append((document_id, source))
        self.started.set()
        await self.release.wait()


class _FailingChatService:
    async def prewarm_document(self, *, document_id: str, source: FilingSource) -> None:
        raise RuntimeError("prewarm failed")


def test_chat_index_prewarm_is_scheduled_without_waiting(monkeypatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        chat_service = _BlockingChatService()
        monkeypatch.setattr(demo_runs, "get_chat_qa_service", lambda: chat_service)

        manager = _manager(tmp_path)
        source = _source(tmp_path)

        started_at = asyncio.get_running_loop().time()
        manager._schedule_chat_index_prewarm(document_id="doc-test", source=source)
        elapsed_ms = (asyncio.get_running_loop().time() - started_at) * 1000

        assert elapsed_ms < 50
        await asyncio.wait_for(chat_service.started.wait(), timeout=1)
        assert chat_service.calls == [("doc-test", source)]
        assert manager._prewarm_tasks

        chat_service.release.set()
        await manager._drain_prewarm_tasks()
        assert not manager._prewarm_tasks

    asyncio.run(scenario())


def test_chat_index_prewarm_failure_is_fail_open(monkeypatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(demo_runs, "get_chat_qa_service", lambda: _FailingChatService())

        manager = _manager(tmp_path)
        manager._schedule_chat_index_prewarm(document_id="doc-test", source=_source(tmp_path))
        await manager._drain_prewarm_tasks()

        assert not manager._prewarm_tasks

    asyncio.run(scenario())


def test_completed_run_prewarms_api_document_id(monkeypatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        chat_service = _BlockingChatService()
        monkeypatch.setattr(demo_runs, "get_chat_qa_service", lambda: chat_service)
        monkeypatch.setattr(demo_runs, "SingleFilingWorkflow", _workflow_factory(document_id="parsed-stem"))

        manager = _manager(tmp_path)
        source = _source(tmp_path)

        await manager.create_run("doc-api-id", source)

        await asyncio.wait_for(chat_service.started.wait(), timeout=1)
        assert chat_service.calls == [("doc-api-id", source)]

        chat_service.release.set()
        await manager._drain_prewarm_tasks()

    asyncio.run(scenario())


def _manager(tmp_path: Path) -> DemoRunManager:
    return DemoRunManager(
        settings=Settings(
            OPENAI_API_KEY="test-key",
            FILINGDELTA_QDRANT_PATH=str(tmp_path / "qdrant"),
            FILINGDELTA_PARSE_PROVIDER="local",
            FILINGDELTA_USE_LLAMA_PARSE=False,
        )
    )


def _source(tmp_path: Path) -> FilingSource:
    return FilingSource(
        source_path=tmp_path / "demo.pdf",
        company_name="招商银行",
        market=Market.A_SHARE,
        doc_type=FilingDocType.ANNUAL_REPORT,
    )


def _workflow_factory(document_id: str):
    class _ImmediateWorkflow:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, *, source: FilingSource):
            return _ImmediateWorkflowHandler(_workflow_result(document_id, source))

    return _ImmediateWorkflow


class _ImmediateWorkflowHandler:
    def __init__(self, result: SingleFilingWorkflowResult) -> None:
        self._result = result

    async def stream_events(self):
        yield demo_runs.StopEvent(result=self._result)


def _workflow_result(document_id: str, source: FilingSource) -> SingleFilingWorkflowResult:
    return SingleFilingWorkflowResult(
        document_id=document_id,
        source_path=source.source_path,
        parser_kind=ParserKind.FALLBACK,
        total_pages=1,
        chunk_count=1,
        headline_metrics=HeadlineMetricFacts(
            document_id=document_id,
            source_path=source.source_path,
        ),
    )
