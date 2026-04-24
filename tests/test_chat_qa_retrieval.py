from pathlib import Path

from filingdelta.schemas.chat import ChatCitation, RetrievedChunk
from filingdelta.schemas.filing import EvidenceKind
from filingdelta.services.chat_qa import (
    _assemble_chat_citations,
    _dedupe_retrieved_chunks,
    _prioritize_retrieved_chunks,
    _sanitize_user_facing_text,
    _select_document_retrieval_strategy,
)


def test_select_document_retrieval_strategy_prefers_section_text_for_narrative_questions() -> None:
    strategy = _select_document_retrieval_strategy("腾讯如何描述 AI 广告能力？")

    assert strategy.primary_chunk_kind == EvidenceKind.SECTION_TEXT.value
    assert strategy.fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value


def test_select_document_retrieval_strategy_prefers_page_text_for_metric_questions() -> None:
    strategy = _select_document_retrieval_strategy("腾讯2025年资本开支是多少？")

    assert strategy.primary_chunk_kind == EvidenceKind.TABLE_ROW.value
    assert strategy.fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.include_fallback_when_primary_found
    assert strategy.primary_top_k == 8


def test_select_document_retrieval_strategy_defaults_to_page_text() -> None:
    strategy = _select_document_retrieval_strategy("请总结这份文档")

    assert strategy.primary_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.fallback_chunk_kind is None


def test_select_document_retrieval_strategy_prefers_table_row_for_customer_deposits() -> None:
    strategy = _select_document_retrieval_strategy("招商银行客户存款有什么变化？")

    assert strategy.primary_chunk_kind == EvidenceKind.TABLE_ROW.value
    assert strategy.fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.include_fallback_when_primary_found


def test_sanitize_user_facing_text_removes_empty_citation_parentheses_and_preserves_lists() -> None:
    cleaned = _sanitize_user_facing_text(
        "ROE（DOC_1、WEB_2）说明： - 衡量股东权益回报。\n"
        "核心存款16日均余额同比增长。\n\n"
        "- 可用于观察资本效率 (WEB_3)。 score=0.82"
    )

    assert "DOC_1" not in cleaned
    assert "WEB_2" not in cleaned
    assert "WEB_3" not in cleaned
    assert "（）" not in cleaned
    assert "(。" not in cleaned
    assert "score=" not in cleaned
    assert "16日均余额" not in cleaned
    assert "核心存款日均余额" in cleaned
    assert "说明：\n- 衡量股东权益回报" in cleaned
    assert "\n- 可用于观察资本效率" in cleaned


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
