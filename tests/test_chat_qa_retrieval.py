from pathlib import Path

from filingdelta.core.config import Settings
from filingdelta.retrieval.page_text_hybrid import retrieve_page_text_hybrid
from filingdelta.schemas.chat import ChatAnswer, ChatCitation, ChatRouteDecision, RetrievedChunk
from filingdelta.schemas.filing import EvidenceKind
from filingdelta.services.chat_qa import (
    LEGACY_TYPED_TABLE_ROW_PRIMARY,
    PAGE_TEXT_HYBRID_NO_TABLE_PRIMARY,
    _assemble_chat_citations,
    _dedupe_retrieved_chunks,
    _prioritize_retrieved_chunks,
    _retrieve_document_evidence,
    _sanitize_user_facing_text,
    _select_document_retrieval_strategy,
)


def test_select_document_retrieval_strategy_prefers_section_text_for_narrative_questions() -> None:
    strategy = _select_document_retrieval_strategy("腾讯如何描述 AI 广告能力？")

    assert strategy.primary_chunk_kind == EvidenceKind.SECTION_TEXT.value
    assert strategy.fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.fallback_retrieval_method == "page_text_hybrid"
    assert strategy.retrieval_mode == PAGE_TEXT_HYBRID_NO_TABLE_PRIMARY


def test_select_document_retrieval_strategy_prefers_page_text_for_metric_questions() -> None:
    strategy = _select_document_retrieval_strategy("腾讯2025年资本开支是多少？")

    assert strategy.primary_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.primary_retrieval_method == "page_text_hybrid"
    assert strategy.fallback_chunk_kind is None
    assert strategy.retrieval_mode == PAGE_TEXT_HYBRID_NO_TABLE_PRIMARY


def test_select_document_retrieval_strategy_defaults_to_page_text() -> None:
    strategy = _select_document_retrieval_strategy("请总结这份文档")

    assert strategy.primary_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.primary_retrieval_method == "page_text_hybrid"
    assert strategy.fallback_chunk_kind is None


def test_select_document_retrieval_strategy_defaults_to_no_table_row_for_customer_deposits() -> None:
    strategy = _select_document_retrieval_strategy("招商银行客户存款有什么变化？")

    assert strategy.primary_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.primary_retrieval_method == "page_text_hybrid"
    assert strategy.retrieval_mode == PAGE_TEXT_HYBRID_NO_TABLE_PRIMARY


def test_select_document_retrieval_strategy_can_use_legacy_table_row_primary() -> None:
    strategy = _select_document_retrieval_strategy(
        "æ‹›å•†é“¶è¡Œå®¢æˆ·å­˜æ¬¾æœ‰ä»€ä¹ˆå˜åŒ–ï¼Ÿ",
        route_decision=ChatRouteDecision(
            route="document_only",
            document_evidence_intent="metric_value",
        ),
        chat_retrieval_strategy=LEGACY_TYPED_TABLE_ROW_PRIMARY,
    )

    assert strategy.primary_chunk_kind == EvidenceKind.TABLE_ROW.value
    assert strategy.fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.include_fallback_when_primary_found
    assert strategy.primary_top_k == 8
    assert strategy.retrieval_mode == LEGACY_TYPED_TABLE_ROW_PRIMARY


def test_select_legacy_table_row_primary_for_customer_deposit_keyword_rule() -> None:
    strategy = _select_document_retrieval_strategy(
        "\u62db\u5546\u94f6\u884c\u5ba2\u6237\u5b58\u6b3e\u6709\u4ec0\u4e48\u53d8\u5316\uff1f",
        chat_retrieval_strategy=LEGACY_TYPED_TABLE_ROW_PRIMARY,
    )

    assert strategy.primary_chunk_kind == EvidenceKind.TABLE_ROW.value
    assert strategy.fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.include_fallback_when_primary_found
    assert strategy.primary_top_k == 8
    assert strategy.retrieval_mode == LEGACY_TYPED_TABLE_ROW_PRIMARY


def test_select_document_retrieval_strategy_uses_metric_attribution_intent() -> None:
    strategy = _select_document_retrieval_strategy(
        "腾讯2025年营销服务收入增长的主要原因是什么？",
        route_decision=ChatRouteDecision(
            route="document_only",
            document_evidence_intent="metric_attribution",
        ),
    )

    assert strategy.primary_chunk_kind == EvidenceKind.SECTION_TEXT.value
    assert strategy.fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.fallback_chunk_kinds == ()
    assert strategy.fallback_retrieval_method == "page_text_hybrid"
    assert strategy.include_fallback_when_primary_found
    assert strategy.primary_top_k == 4
    assert strategy.fallback_top_k == 4


def test_select_document_retrieval_strategy_uses_metric_value_intent() -> None:
    strategy = _select_document_retrieval_strategy(
        "腾讯2025年营销服务收入是多少？",
        route_decision=ChatRouteDecision(
            route="document_only",
            document_evidence_intent="metric_value",
        ),
    )

    assert strategy.primary_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.primary_retrieval_method == "page_text_hybrid"
    assert strategy.fallback_chunk_kind is None


def test_select_document_retrieval_strategy_uses_business_narrative_intent() -> None:
    strategy = _select_document_retrieval_strategy(
        "招商银行如何管控房地产风险？",
        route_decision=ChatRouteDecision(
            route="document_only",
            document_evidence_intent="business_narrative",
        ),
    )

    assert strategy.primary_chunk_kind == EvidenceKind.SECTION_TEXT.value
    assert strategy.fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.fallback_retrieval_method == "page_text_hybrid"
    assert strategy.include_fallback_when_primary_found
    assert strategy.primary_top_k == 4
    assert strategy.fallback_top_k == 4


def test_default_settings_use_page_text_hybrid_no_table_primary(monkeypatch) -> None:
    monkeypatch.delenv("FILINGDELTA_CHAT_RETRIEVAL_STRATEGY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.filingdelta_chat_retrieval_strategy == PAGE_TEXT_HYBRID_NO_TABLE_PRIMARY


def test_chat_answer_accepts_new_retrieval_modes() -> None:
    answer = ChatAnswer(
        document_id="doc-test",
        session_id="session-test",
        question="question",
        answer="answer",
        retrieval_mode=PAGE_TEXT_HYBRID_NO_TABLE_PRIMARY,
    )
    legacy_answer = ChatAnswer(
        document_id="doc-test",
        session_id="session-test",
        question="question",
        answer="answer",
        retrieval_mode=LEGACY_TYPED_TABLE_ROW_PRIMARY,
    )

    assert answer.retrieval_mode == PAGE_TEXT_HYBRID_NO_TABLE_PRIMARY
    assert legacy_answer.retrieval_mode == LEGACY_TYPED_TABLE_ROW_PRIMARY


def test_retrieve_document_evidence_hybrid_without_page_text_corpus_returns_empty() -> None:
    strategy = _select_document_retrieval_strategy(
        "segment revenue?",
        route_decision=ChatRouteDecision(
            route="document_only",
            document_evidence_intent="metric_value",
        ),
    )
    retriever = _FakeRetriever(
        {
            EvidenceKind.PAGE_TEXT.value: [
                _chunk("semantic-page", document_id="doc-a", kind=EvidenceKind.PAGE_TEXT.value)
            ]
        }
    )

    chunks, retrieval_mode = _retrieve_document_evidence(
        retriever=retriever,
        document_id="doc-a",
        question="segment revenue?",
        callback_manager=None,
        strategy=strategy,
    )

    assert chunks == []
    assert retrieval_mode == PAGE_TEXT_HYBRID_NO_TABLE_PRIMARY
    assert retriever.calls == []


def test_metric_value_default_retrieval_does_not_query_table_row() -> None:
    strategy = _select_document_retrieval_strategy(
        "segment revenue?",
        route_decision=ChatRouteDecision(
            route="document_only",
            document_evidence_intent="metric_value",
        ),
    )
    page_text = _chunk(
        "page-text",
        document_id="doc-a",
        kind=EvidenceKind.PAGE_TEXT.value,
        text="segment revenue was 100",
    )
    retriever = _FakeRetriever({EvidenceKind.PAGE_TEXT.value: [page_text]})

    chunks, retrieval_mode = _retrieve_document_evidence(
        retriever=retriever,
        document_id="doc-a",
        question="segment revenue",
        callback_manager=None,
        strategy=strategy,
        page_text_chunks=[page_text],
    )

    assert retrieval_mode == PAGE_TEXT_HYBRID_NO_TABLE_PRIMARY
    assert [call["chunk_kind"] for call in retriever.calls] == [EvidenceKind.PAGE_TEXT.value]
    assert all(chunk.chunk_kind == EvidenceKind.PAGE_TEXT.value for chunk in chunks)


def test_metric_attribution_retrieval_keeps_section_text_with_page_text_hybrid() -> None:
    section = _chunk("section", document_id="doc-a", kind=EvidenceKind.SECTION_TEXT.value)
    page_text = _chunk(
        "page-text",
        document_id="doc-a",
        kind=EvidenceKind.PAGE_TEXT.value,
        text="management explained revenue growth",
    )
    retriever = _FakeRetriever(
        {
            EvidenceKind.SECTION_TEXT.value: [section],
            EvidenceKind.PAGE_TEXT.value: [page_text],
        }
    )
    strategy = _select_document_retrieval_strategy(
        "why did revenue grow?",
        route_decision=ChatRouteDecision(
            route="document_only",
            document_evidence_intent="metric_attribution",
        ),
    )

    chunks, _ = _retrieve_document_evidence(
        retriever=retriever,
        document_id="doc-a",
        question="revenue growth",
        callback_manager=None,
        strategy=strategy,
        page_text_chunks=[page_text],
    )

    assert [call["chunk_kind"] for call in retriever.calls] == [
        EvidenceKind.SECTION_TEXT.value,
        EvidenceKind.PAGE_TEXT.value,
    ]
    assert {chunk.chunk_kind for chunk in chunks} == {
        EvidenceKind.SECTION_TEXT.value,
        EvidenceKind.PAGE_TEXT.value,
    }


def test_business_narrative_retrieval_keeps_section_text_with_page_text_hybrid() -> None:
    section = _chunk("section", document_id="doc-a", kind=EvidenceKind.SECTION_TEXT.value)
    page_text = _chunk(
        "page-text",
        document_id="doc-a",
        kind=EvidenceKind.PAGE_TEXT.value,
        text="risk controls and response measures",
    )
    retriever = _FakeRetriever(
        {
            EvidenceKind.SECTION_TEXT.value: [section],
            EvidenceKind.PAGE_TEXT.value: [page_text],
        }
    )
    strategy = _select_document_retrieval_strategy(
        "what risks are disclosed?",
        route_decision=ChatRouteDecision(
            route="document_only",
            document_evidence_intent="business_narrative",
        ),
    )

    chunks, _ = _retrieve_document_evidence(
        retriever=retriever,
        document_id="doc-a",
        question="risk response",
        callback_manager=None,
        strategy=strategy,
        page_text_chunks=[page_text],
    )

    assert [call["chunk_kind"] for call in retriever.calls] == [
        EvidenceKind.SECTION_TEXT.value,
        EvidenceKind.PAGE_TEXT.value,
    ]
    assert {chunk.chunk_kind for chunk in chunks} == {
        EvidenceKind.SECTION_TEXT.value,
        EvidenceKind.PAGE_TEXT.value,
    }


def test_page_text_hybrid_bm25_corpus_does_not_cross_document_id() -> None:
    retriever = _FakeRetriever({EvidenceKind.PAGE_TEXT.value: []})
    doc_a_chunk = _chunk(
        "doc-a-page",
        document_id="doc-a",
        kind=EvidenceKind.PAGE_TEXT.value,
        text="ordinary disclosure",
    )
    doc_b_chunk = _chunk(
        "doc-b-page",
        document_id="doc-b",
        kind=EvidenceKind.PAGE_TEXT.value,
        text="unique cross document bm25 term",
    )

    chunks = retrieve_page_text_hybrid(
        retriever=retriever,
        document_id="doc-a",
        question="unique cross document bm25 term",
        page_text_chunks=[doc_a_chunk, doc_b_chunk],
    )

    assert chunks == []
    assert [call["document_id"] for call in retriever.calls] == ["doc-a"]


def test_sanitize_user_facing_text_removes_empty_citation_parentheses_and_preserves_lists() -> None:
    cleaned = _sanitize_user_facing_text(
        "ROE（DOC_1、WEB_2）说明： - 衡量股东权益回报。\n"
        "核心存款16日均余额同比增长。\n\n"
        "核心存款（16日均余额口径）同比增长。\n"
        "核心存款（16日均）为77,442.68亿元。\n"
        "核心存款**16日均余额77,442.68亿元**。\n"
        "16日均余额同比增长。\n"
        "16日均继续改善。\n"
        "16平均余额提升。\n"
        "- 可用于观察资本效率 (WEB_3)。 score=0.82"
    )

    assert "DOC_1" not in cleaned
    assert "WEB_2" not in cleaned
    assert "WEB_3" not in cleaned
    assert "（）" not in cleaned
    assert "(。" not in cleaned
    assert "score=" not in cleaned
    assert "16日均余额" not in cleaned
    assert "16日均" not in cleaned
    assert "16平均余额" not in cleaned
    assert "核心存款日均余额" in cleaned
    assert "核心存款（日均余额口径）" in cleaned
    assert "核心存款（日均）" in cleaned
    assert "核心存款**日均余额77,442.68亿元**" in cleaned
    assert "日均余额同比增长" in cleaned
    assert "日均继续改善" in cleaned
    assert "平均余额提升" in cleaned
    assert "说明：\n- 衡量股东权益回报" in cleaned
    assert "\n- 可用于观察资本效率" in cleaned


def test_sanitize_user_facing_text_preserves_non_footnote_day_average_numbers() -> None:
    cleaned = _sanitize_user_facing_text("前10日均余额保持稳定，30日均余额同步披露。")

    assert "前10日均余额" in cleaned
    assert "30日均余额" in cleaned


def test_sanitize_user_facing_text_formats_million_units_for_amounts() -> None:
    cleaned = _sanitize_user_facing_text("腾讯2025年的资本开支为19,632（人民币百万元）。")

    assert cleaned == "腾讯2025年的资本开支为19,632 百万元，即 196.32 亿元。"


def test_sanitize_user_facing_text_removes_typed_metadata_parentheticals() -> None:
    cleaned = _sanitize_user_facing_text(
        "文档证据\n（第18页，财务摘要表“资本开支”，期间为fy2025）"
    )

    assert "fy2025" not in cleaned
    assert "财务摘要表" not in cleaned
    assert cleaned == "文档证据"


def test_sanitize_user_facing_text_normalizes_raw_period_hints() -> None:
    cleaned = _sanitize_user_facing_text("资本开支（FY2025）为19,632 百万元，即 196.32 亿元。")

    assert "FY2025" not in cleaned
    assert "2025年" in cleaned
    assert "196.32 亿元" in cleaned


def test_assemble_chat_citations_dedupes_document_pages_and_external_urls() -> None:
    source_path = Path("data/raw/test.pdf")
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id="chunk-a",
            document_id="doc-test",
            page_number=19,
            source_path=source_path,
            text="ROE 为 13.44%。",
        ),
        RetrievedChunk(
            chunk_id="chunk-b",
            document_id="doc-test",
            page_number=19,
            source_path=source_path,
            text="ROE 同比下降。",
        ),
    ]
    external_citations = [
        ChatCitation(
            citation_id="external-a",
            source_type="external",
            url="https://example.test/roe",
            title="Return on equity",
        ),
        ChatCitation(
            citation_id="external-b",
            source_type="external",
            url="https://example.test/roe",
            title="ROE duplicate",
        ),
    ]

    citations = _assemble_chat_citations(
        used_chunk_ids=["chunk-a", "chunk-b", "chunk-a"],
        retrieved_chunks=retrieved_chunks,
        used_external_citation_ids=["external-a", "external-b"],
        external_citations=external_citations,
    )

    assert [(citation.source_type, citation.page_number, citation.url) for citation in citations] == [
        ("document", 19, None),
        ("external", None, "https://example.test/roe"),
    ]


def test_dedupe_retrieved_chunks_collapses_same_table_row_on_same_page() -> None:
    source_path = Path("data/raw/test.pdf")
    chunks = [
        RetrievedChunk(
            chunk_id="row-a",
            document_id="doc-test",
            page_number=47,
            source_path=source_path,
            text="客户存款余额98,361.30亿元。",
            chunk_kind=EvidenceKind.TABLE_ROW.value,
            row_label="客户存款",
        ),
        RetrievedChunk(
            chunk_id="row-b",
            document_id="doc-test",
            page_number=47,
            source_path=source_path,
            text="客户存款余额98,361.30亿元，较上年末增长8.13%。",
            chunk_kind=EvidenceKind.TABLE_ROW.value,
            row_label="客户存款",
        ),
        RetrievedChunk(
            chunk_id="row-c",
            document_id="doc-test",
            page_number=30,
            source_path=source_path,
            text="客户存款总额9,836,130，活期存款占比50.79%。",
            chunk_kind=EvidenceKind.TABLE_ROW.value,
            row_label="客户存款",
        ),
    ]

    deduped = _dedupe_retrieved_chunks(chunks)

    assert [chunk.chunk_id for chunk in deduped] == ["row-a", "row-c"]


def test_prioritize_retrieved_chunks_prefers_group_customer_deposit_rows() -> None:
    source_path = Path("data/raw/test.pdf")
    chunks = [
        RetrievedChunk(
            chunk_id="company-row",
            document_id="doc-test",
            page_number=47,
            source_path=source_path,
            text="本公司核心存款日均余额77,442.68亿元。",
            chunk_kind=EvidenceKind.TABLE_ROW.value,
            row_label="公司客户存款",
        ),
        RetrievedChunk(
            chunk_id="group-row",
            document_id="doc-test",
            page_number=30,
            source_path=source_path,
            text="客户存款总额 9,836,130 100.00 9,096,587 100.00。活期存款占比为50.79%。",
            chunk_kind=EvidenceKind.TABLE_ROW.value,
            row_label="客户存款",
        ),
    ]

    prioritized = _prioritize_retrieved_chunks(
        question="招商银行客户存款有什么变化？",
        chunks=chunks,
    )

    assert [chunk.chunk_id for chunk in prioritized] == ["group-row", "company-row"]


class _FakeRetriever:
    def __init__(self, responses: dict[str | None, list[RetrievedChunk]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        *,
        document_id: str,
        question: str,
        top_k: int = 6,
        chunk_kind: str | None = None,
        callback_manager=None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "document_id": document_id,
                "question": question,
                "top_k": top_k,
                "chunk_kind": chunk_kind,
                "callback_manager": callback_manager,
            }
        )
        return list(self._responses.get(chunk_kind, []))[:top_k]


def _chunk(
    chunk_id: str,
    *,
    document_id: str,
    kind: str,
    text: str = "sample disclosure text",
    page_number: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page_number=page_number,
        source_path=Path("data/raw/test.pdf"),
        text=text,
        chunk_kind=kind,
        score=1.0,
    )
