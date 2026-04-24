from __future__ import annotations

import asyncio
import json

from filingdelta.api.routes.demo import _encode_stream_event, _iter_text_deltas, _stream_demo_chat
from filingdelta.schemas.chat import ChatAnswer
from filingdelta.schemas.demo import DemoChatRequest


def test_encode_stream_event_uses_ndjson_without_ascii_escaping() -> None:
    encoded = _encode_stream_event(
        {
            "type": "status",
            "stage": "router",
            "message": "正在判断问题类型...",
        }
    )

    assert encoded.endswith("\n")
    assert "\\u" not in encoded
    assert json.loads(encoded)["message"] == "正在判断问题类型..."


def test_iter_text_deltas_preserves_text_order() -> None:
    text = "招商银行通过政策对齐、项目选择和贷后管理来管控房地产风险。"

    chunks = _iter_text_deltas(text, chunk_size=8)

    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_iter_text_deltas_omits_empty_text() -> None:
    assert _iter_text_deltas("   ") == []


def test_stream_demo_chat_emits_status_delta_and_done() -> None:
    async def scenario() -> None:
        payload = DemoChatRequest(
            document_id="doc-test",
            session_id="session-test",
            question="招商银行客户存款有什么变化？",
        )
        raw_events = [
            json.loads(line)
            async for line in _stream_demo_chat(
                payload=payload,
                source=object(),
                service=_FakeChatService(),
            )
        ]

        event_types = [event["type"] for event in raw_events]
        assert event_types[0] == "status"
        assert "delta" in event_types
        assert "citations" in event_types
        assert event_types[-1] == "done"
        assert raw_events[-1]["response"]["answer"] == "客户存款规模增长，活期占比下降。"

    asyncio.run(scenario())


def test_stream_demo_chat_cancels_service_task_when_closed() -> None:
    async def scenario() -> None:
        payload = DemoChatRequest(
            document_id="doc-test",
            session_id="session-test",
            question="招商银行客户存款有什么变化？",
        )
        service = _SlowChatService()
        stream = _stream_demo_chat(
            payload=payload,
            source=object(),
            service=service,
        )

        first_line = await stream.__anext__()
        assert json.loads(first_line)["type"] == "status"

        await stream.aclose()
        await asyncio.wait_for(service.cancelled.wait(), timeout=1.0)

    asyncio.run(scenario())


class _FakeChatService:
    async def ask(
        self,
        *,
        document_id: str,
        source: object,
        session_id: str | None,
        question: str,
        status_callback,
    ) -> ChatAnswer:
        await status_callback("router", "正在判断问题类型...")
        return ChatAnswer(
            document_id=document_id,
            session_id=session_id or "session",
            question=question,
            answer="客户存款规模增长，活期占比下降。",
        )


class _SlowChatService:
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def ask(
        self,
        *,
        document_id: str,
        source: object,
        session_id: str | None,
        question: str,
        status_callback,
    ) -> ChatAnswer:
        await status_callback("router", "正在判断问题类型...")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
