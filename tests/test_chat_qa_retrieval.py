from filingdelta.schemas.filing import EvidenceKind
from filingdelta.services.chat_qa import _select_document_retrieval_strategy


def test_select_document_retrieval_strategy_prefers_section_text_for_narrative_questions() -> None:
    strategy = _select_document_retrieval_strategy("腾讯如何描述 AI 广告能力？")

    assert strategy.primary_chunk_kind == EvidenceKind.SECTION_TEXT.value
    assert strategy.fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value


def test_select_document_retrieval_strategy_prefers_page_text_for_metric_questions() -> None:
    strategy = _select_document_retrieval_strategy("腾讯2025年资本开支是多少？")

    assert strategy.primary_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.fallback_chunk_kind is None


def test_select_document_retrieval_strategy_defaults_to_page_text() -> None:
    strategy = _select_document_retrieval_strategy("请总结这份文档")

    assert strategy.primary_chunk_kind == EvidenceKind.PAGE_TEXT.value
    assert strategy.fallback_chunk_kind is None
