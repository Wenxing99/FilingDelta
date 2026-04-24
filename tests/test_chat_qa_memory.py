from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from filingdelta.core.config import Settings
from filingdelta.schemas.chat import ChatSessionState, ConversationSummary
from filingdelta.schemas.filing import FilingDocType, FilingDocument, Market, ParserKind
from filingdelta.services.chat_qa import ChatQAService, _has_conversation_context
from filingdelta.services.chat_telemetry import ChatTelemetryRecorder


@dataclass(slots=True)
class _SummaryCall:
    release: asyncio.Event
    summary: ConversationSummary


class _ControlledSummarizer:
    def __init__(self) -> None:
        self.calls: list[_SummaryCall] = []
        self._call_added = asyncio.Event()

    async def summarize(self, **_: object) -> ConversationSummary:
        call = _SummaryCall(
            release=asyncio.Event(),
            summary=ConversationSummary(summary_text=f"summary-{len(self.calls) + 1}"),
        )
        self.calls.append(call)
        self._call_added.set()
        await call.release.wait()
        return call.summary

    async def wait_for_call_count(self, expected_count: int) -> None:
        while len(self.calls) < expected_count:
            self._call_added.clear()
            await asyncio.wait_for(self._call_added.wait(), timeout=1)


def test_record_conversation_turn_schedules_memory_summary_without_waiting(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = _chat_service(tmp_path)
        summarizer = _ControlledSummarizer()
        service._memory_summarizer = summarizer
        telemetry = ChatTelemetryRecorder()

        started = perf_counter()
        await service._record_conversation_turn(
            session_state=ChatSessionState(document_id="doc-test", session_id="session-test"),
            document=_document(),
            user_question="招商银行如何管控房地产风险？",
            assistant_answer="招商银行披露了相关风险管控措施。",
        )
        elapsed_ms = (perf_counter() - started) * 1000

        assert elapsed_ms < 100
        assert telemetry.build(route_type="document_only", succeeded=True).steps.memory_summarizer_ms is None
        await summarizer.wait_for_call_count(1)

        state = await service._memory.get_or_create(document_id="doc-test", session_id="session-test")
        assert state.conversation_summary.summary_text == ""
        assert [message.role for message in state.recent_messages] == ["user", "assistant"]

        summarizer.calls[0].release.set()
        await service._drain_background_tasks()

        state = await service._memory.get_or_create(document_id="doc-test", session_id="session-test")
        assert state.conversation_summary.summary_text == "summary-1"

    asyncio.run(scenario())


def test_background_memory_summary_does_not_let_stale_task_overwrite_newer_summary(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        service = _chat_service(tmp_path)
        summarizer = _ControlledSummarizer()
        service._memory_summarizer = summarizer
        session_state = ChatSessionState(document_id="doc-test", session_id="session-test")

        await service._record_conversation_turn(
            session_state=session_state,
            document=_document(),
            user_question="第一问",
            assistant_answer="第一答",
        )
        await summarizer.wait_for_call_count(1)

        await service._record_conversation_turn(
            session_state=session_state,
            document=_document(),
            user_question="第二问",
            assistant_answer="第二答",
        )
        await summarizer.wait_for_call_count(2)

        summarizer.calls[1].release.set()
        await _wait_for_summary(service, expected_summary="summary-2")

        summarizer.calls[0].release.set()
        await service._drain_background_tasks()

        state = await service._memory.get_or_create(document_id="doc-test", session_id="session-test")
        assert state.conversation_summary.summary_text == "summary-2"

    asyncio.run(scenario())


def test_has_conversation_context_is_false_for_empty_session() -> None:
    session_state = ChatSessionState(document_id="doc-test", session_id="session-test")

    assert not _has_conversation_context(session_state)


def test_has_conversation_context_is_true_when_recent_messages_exist() -> None:
    async def scenario() -> None:
        service = _chat_service(Path("/tmp/filingdelta-chat-memory-test"))
        state = await service._memory.append_turn(
            document_id="doc-test",
            session_id="session-test",
            user_message="腾讯2025年资本开支是多少？",
            assistant_message="腾讯披露了资本开支金额。",
        )

        assert _has_conversation_context(state)

    asyncio.run(scenario())


def test_has_conversation_context_is_true_when_summary_exists() -> None:
    session_state = ChatSessionState(
        document_id="doc-test",
        session_id="session-test",
        conversation_summary=ConversationSummary(summary_text="用户之前问过房地产风险。"),
    )

    assert _has_conversation_context(session_state)


def _chat_service(tmp_path: Path) -> ChatQAService:
    return ChatQAService(
        settings=Settings(
            OPENAI_API_KEY="test-key",
            FILINGDELTA_QDRANT_PATH=str(tmp_path / "qdrant"),
            FILINGDELTA_PARSE_PROVIDER="local",
            FILINGDELTA_USE_LLAMA_PARSE=False,
        )
    )


def _document() -> FilingDocument:
    return FilingDocument(
        document_id="doc-test",
        company_name="招商银行",
        market=Market.A_SHARE,
        doc_type=FilingDocType.ANNUAL_REPORT,
        source_path=Path("data/raw/招商银行2025年度报告.pdf"),
        parser_kind=ParserKind.PYMUPDF,
        total_pages=1,
    )


async def _wait_for_summary(service: ChatQAService, *, expected_summary: str) -> None:
    for _ in range(100):
        state = await service._memory.get_or_create(document_id="doc-test", session_id="session-test")
        if state.conversation_summary.summary_text == expected_summary:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Timed out waiting for summary {expected_summary!r}")
