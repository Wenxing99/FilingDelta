from __future__ import annotations

import asyncio
import json
from pathlib import Path

import filingdelta.api.routes.demo as demo_routes
from filingdelta.api.routes.demo import (
    _encode_stream_event,
    _iter_text_deltas,
    _stream_demo_chat,
    _stream_static_chat_answer,
)
from filingdelta.financial_facts import FinancialFactsQueryService, SQLiteFinancialFactStore
from filingdelta.financial_facts.schemas import FinancialFact
from filingdelta.schemas.chat import ChatAnswer
from filingdelta.schemas.demo import DemoChatRequest
from filingdelta.services.kb_financial_facts import KbFinancialFactsChatService


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


def test_demo_chat_kb_fact_answer_short_circuits_chat_service(monkeypatch) -> None:
    async def scenario() -> None:
        payload = DemoChatRequest(
            document_id="doc-test",
            session_id="session-test",
            question="2025年哪三家公司营业收入最高？",
        )
        answer = ChatAnswer(
            document_id="doc-test",
            session_id="session-test",
            question=payload.question,
            answer="结构化事实库结果",
            retrieval_mode="kb_financial_facts",
            citations=[],
        )
        monkeypatch.setattr(demo_routes, "get_demo_document_source", lambda _document_id: object())
        monkeypatch.setattr(
            demo_routes,
            "get_chat_qa_service",
            lambda: (_ for _ in ()).throw(AssertionError("ChatQA should not be instantiated")),
        )
        monkeypatch.setattr(demo_routes, "_kb_financial_facts_chat_service", _FakeKbService(answer))

        response = await demo_routes.demo_chat(payload)

        assert response.response.answer == "结构化事实库结果"
        assert response.response.retrieval_mode == "kb_financial_facts"
        assert response.response.citations == []

    asyncio.run(scenario())


def test_demo_chat_non_kb_question_uses_chat_service(monkeypatch) -> None:
    async def scenario() -> None:
        service = _FakeChatService()
        monkeypatch.setattr(demo_routes, "get_demo_document_source", lambda _document_id: object())
        monkeypatch.setattr(demo_routes, "get_chat_qa_service", lambda: service)
        monkeypatch.setattr(demo_routes, "_kb_financial_facts_chat_service", _FakeKbService(None))

        response = await demo_routes.demo_chat(
            DemoChatRequest(
                document_id="doc-test",
                session_id="session-test",
                question="招商银行营业收入是多少？",
            )
        )

        assert response.response.answer == "客户存款规模增长，活期占比下降。"

    asyncio.run(scenario())


def test_demo_chat_document_metric_highest_business_uses_chat_service(monkeypatch) -> None:
    async def scenario() -> None:
        service = _FakeChatService()
        monkeypatch.setattr(demo_routes, "get_demo_document_source", lambda _document_id: object())
        monkeypatch.setattr(demo_routes, "get_chat_qa_service", lambda: service)
        monkeypatch.setattr(
            demo_routes,
            "_kb_financial_facts_chat_service",
            KbFinancialFactsChatService(FinancialFactsQueryService("missing.sqlite")),
        )
        questions = (
            "2025年招商银行营业收入最高的业务是什么？",
            "这份报告里营业收入最高的业务是什么？",
        )

        responses = [
            (await demo_routes.demo_chat(
                DemoChatRequest(
                    document_id="doc-test",
                    session_id="session-test",
                    question=question,
                )
            )).response
            for question in questions
        ]

        assert [response.answer for response in responses] == [
            "客户存款规模增长，活期占比下降。",
            "客户存款规模增长，活期占比下降。",
        ]
        assert all(response.retrieval_mode != "kb_financial_facts" for response in responses)

    asyncio.run(scenario())


def test_demo_chat_kb_rank_forms_without_company_scope_do_not_fall_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "facts.sqlite"
        _seed_revenue_facts(db_path)
        monkeypatch.setattr(demo_routes, "get_demo_document_source", lambda _document_id: object())
        monkeypatch.setattr(
            demo_routes,
            "get_chat_qa_service",
            lambda: (_ for _ in ()).throw(AssertionError("ChatQA should not be instantiated")),
        )
        monkeypatch.setattr(
            demo_routes,
            "_kb_financial_facts_chat_service",
            KbFinancialFactsChatService(FinancialFactsQueryService(db_path)),
        )
        questions = (
            "current-KB revenue Top3",
            "2025 revenue Top3 in current KB",
            "2025年当前KB营业收入Top3",
            "2025年营业收入Top3",
            "2025年营业收入最高是哪家企业？",
        )

        responses = [
            (await demo_routes.demo_chat(
                DemoChatRequest(
                    document_id="doc-test",
                    session_id="session-test",
                    question=question,
                )
            )).response
            for question in questions
        ]

        assert all(response.retrieval_mode == "kb_financial_facts" for response in responses)
        assert responses[0].route == "unsupported"
        assert "请明确查询年份" in responses[0].answer
        assert [response.route for response in responses[1:]] == ["document_only"] * 4
        assert all("doc-a" in response.answer for response in responses[1:])

    asyncio.run(scenario())


def test_demo_chat_unsupported_exact_metric_rank_does_not_fall_back(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(demo_routes, "get_demo_document_source", lambda _document_id: object())
        monkeypatch.setattr(
            demo_routes,
            "get_chat_qa_service",
            lambda: (_ for _ in ()).throw(AssertionError("ChatQA should not be instantiated")),
        )
        monkeypatch.setattr(
            demo_routes,
            "_kb_financial_facts_chat_service",
            KbFinancialFactsChatService(FinancialFactsQueryService("missing.sqlite")),
        )

        response = (
            await demo_routes.demo_chat(
                DemoChatRequest(
                    document_id="doc-test",
                    session_id="session-test",
                    question="2025年经营现金流Top3",
                )
            )
        ).response

        assert response.route == "unsupported"
        assert response.retrieval_mode == "kb_financial_facts"
        assert "当前结构化事实库只支持" in response.answer

    asyncio.run(scenario())


def test_stream_static_chat_answer_emits_kb_done_response() -> None:
    async def scenario() -> None:
        answer = ChatAnswer(
            document_id="doc-test",
            session_id="session-test",
            question="2025年哪三家公司营业收入最高？",
            answer="结构化事实库结果",
            retrieval_mode="kb_financial_facts",
            citations=[],
        )
        events = [json.loads(line) async for line in _stream_static_chat_answer(answer)]

        assert [event["type"] for event in events] == ["status", "delta", "citations", "done"]
        assert events[0]["stage"] == "financial_facts"
        assert events[-1]["response"]["retrieval_mode"] == "kb_financial_facts"

    asyncio.run(scenario())


class _FakeKbService:
    def __init__(self, answer: ChatAnswer | None) -> None:
        self.answer = answer

    def answer_if_supported(
        self,
        *,
        document_id: str,
        session_id: str | None,
        question: str,
    ) -> ChatAnswer | None:
        return self.answer


class _FakeChatService:
    async def ask(
        self,
        *,
        document_id: str,
        source: object,
        session_id: str | None,
        question: str,
        status_callback=None,
    ) -> ChatAnswer:
        if status_callback is not None:
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


def _seed_revenue_facts(db_path: Path) -> None:
    store = SQLiteFinancialFactStore(db_path)
    store.upsert_facts(
        [
            _financial_fact("doc-a", "A", 300),
            _financial_fact("doc-b", "B", 200),
            _financial_fact("doc-c", "C", 100),
        ]
    )


def _financial_fact(document_id: str, company_name: str, normalized_value: float) -> FinancialFact:
    return FinancialFact(
        fact_id=f"{document_id}:revenue",
        document_id=document_id,
        company_name=company_name,
        source_path=Path(f"{document_id}.pdf"),
        metric_id="revenue",
        metric_label="营业收入",
        source_metric_name="revenue",
        period_type="period",
        fiscal_period="2025 annual report",
        fiscal_year=2025,
        value=normalized_value / 1_000_000,
        unit_raw="人民币百万元",
        currency="CNY",
        scale=1_000_000,
        normalized_value=normalized_value,
        normalized_unit="CNY",
        evidence_page=8,
        evidence_quote="营业收入 100",
        review_status="verified",
    )
