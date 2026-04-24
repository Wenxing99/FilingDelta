from pathlib import Path

from filingdelta.eval.table_row_retrieval import (
    TableRowRetrievalCase,
    build_table_row_query_result,
    combine_table_row_with_page_fallback,
    summarize_table_row_results,
)
from filingdelta.schemas.chat import RetrievedChunk
from filingdelta.schemas.filing import EvidenceKind


def test_build_table_row_query_result_scores_page_row_and_metric_hits() -> None:
    case = TableRowRetrievalCase(
        case_id="CASE-1",
        document_key="doc",
        query="客户存款有什么变化？",
        expected_pages=(30,),
        expected_row_labels=("客户存款",),
        expected_metric_tags=("customer_deposits",),
    )

    result = build_table_row_query_result(
        case=case,
        mode="table_row_only",
        chunks=[
            _chunk(
                page_number=30,
                row_label="客户存款",
                metric_tags=["customer_deposits", "deposits"],
            )
        ],
        retrieval_ms=12,
        top_k=6,
    )

    assert result["page_hit@6"] is True
    assert result["row_label_hit@6"] is True
    assert result["metric_tag_hit@6"] is True
    assert result["hit_pages"] == [30]
    assert result["hit_row_labels"] == ["客户存款"]
    assert result["hit_metric_tags"] == ["customer_deposits"]


def test_combine_table_row_with_page_fallback_uses_page_text_when_no_rows() -> None:
    page_chunk = _chunk(
        chunk_kind=EvidenceKind.PAGE_TEXT.value,
        page_number=47,
        row_label=None,
        metric_tags=[],
    )

    chunks, fallback_used, fallback_source = combine_table_row_with_page_fallback(
        table_row_chunks=[],
        page_text_chunks=[page_chunk],
        top_k=6,
    )

    assert chunks == [page_chunk]
    assert fallback_used is True
    assert fallback_source == "page_text"


def test_combine_table_row_with_page_fallback_keeps_rows_first() -> None:
    row_chunk = _chunk(page_number=30, row_label="客户存款", metric_tags=["customer_deposits"])
    page_chunk = _chunk(
        chunk_kind=EvidenceKind.PAGE_TEXT.value,
        page_number=30,
        row_label=None,
        metric_tags=[],
    )

    chunks, fallback_used, fallback_source = combine_table_row_with_page_fallback(
        table_row_chunks=[row_chunk],
        page_text_chunks=[page_chunk],
        top_k=6,
    )

    assert chunks == [row_chunk, page_chunk]
    assert fallback_used is False
    assert fallback_source == "table_row"


def test_summarize_table_row_results_counts_misses() -> None:
    results = [
        {
            "id": "A",
            "page_hit@6": True,
            "row_label_hit@6": True,
            "metric_tag_hit@6": True,
            "fallback_used": False,
        },
        {
            "id": "B",
            "page_hit@6": False,
            "row_label_hit@6": False,
            "metric_tag_hit@6": True,
            "fallback_used": True,
        },
    ]

    summary = summarize_table_row_results(results, top_k=6)

    assert summary["total_queries"] == 2
    assert summary["page_hit@6_count"] == 1
    assert summary["row_label_hit@6_count"] == 1
    assert summary["metric_tag_hit@6_count"] == 2
    assert summary["fallback_count"] == 1
    assert summary["page_miss_cases"] == ["B"]
    assert summary["row_label_miss_cases"] == ["B"]
    assert summary["metric_tag_miss_cases"] == []


def _chunk(
    *,
    chunk_kind: str = EvidenceKind.TABLE_ROW.value,
    page_number: int,
    row_label: str | None,
    metric_tags: list[str],
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{chunk_kind}-{page_number}-{row_label or 'page'}",
        document_id="doc-test",
        page_number=page_number,
        source_path=Path("dummy.pdf"),
        text=f"{row_label or 'page text'} example",
        chunk_kind=chunk_kind,
        row_label=row_label,
        metric_tags=metric_tags,
    )
