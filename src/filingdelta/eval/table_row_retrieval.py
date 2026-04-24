from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from filingdelta.schemas.chat import RetrievedChunk


@dataclass(frozen=True)
class TableRowRetrievalCase:
    case_id: str
    document_key: str
    query: str
    expected_pages: tuple[int, ...]
    expected_row_labels: tuple[str, ...]
    expected_metric_tags: tuple[str, ...]
    notes: str = ""


TABLE_ROW_RETRIEVAL_CASES: tuple[TableRowRetrievalCase, ...] = (
    TableRowRetrievalCase(
        case_id="CMB-DEP-01",
        document_key="cmb_2025_annual",
        query="招商银行客户存款有什么变化？",
        expected_pages=(30, 47, 222),
        expected_row_labels=("客户存款", "活期存款", "定期存款"),
        expected_metric_tags=("customer_deposits", "deposits"),
        notes="Broad deposit question should retrieve group-level deposit rows.",
    ),
    TableRowRetrievalCase(
        case_id="CMB-DEP-02",
        document_key="cmb_2025_annual",
        query="招商银行活期存款占比是多少，发生了什么变化？",
        expected_pages=(30, 47),
        expected_row_labels=("活期存款", "客户存款"),
        expected_metric_tags=("demand_deposits", "customer_deposits"),
        notes="Demand-deposit share and terming trend.",
    ),
    TableRowRetrievalCase(
        case_id="CMB-ROE-01",
        document_key="cmb_2025_annual",
        query="招商银行净资产收益率有什么变化？",
        expected_pages=(15, 19, 249),
        expected_row_labels=("净资产收益率",),
        expected_metric_tags=("roe", "profitability_ratio"),
        notes="ROE / ROAE document metric.",
    ),
    TableRowRetrievalCase(
        case_id="CMB-NPL-01",
        document_key="cmb_2025_annual",
        query="招商银行不良贷款率是多少，有什么变化？",
        expected_pages=(45, 47, 48, 49, 64),
        expected_row_labels=("不良贷款率",),
        expected_metric_tags=("npl_ratio", "asset_quality"),
        notes="Asset-quality metric row.",
    ),
    TableRowRetrievalCase(
        case_id="TCEHY-CAPEX-01",
        document_key="tcehy_2025_annual",
        query="腾讯2025年资本开支是多少？",
        expected_pages=(18,),
        expected_row_labels=("资本开支",),
        expected_metric_tags=("capital_expenditure", "capex"),
        notes="Tencent capex table row.",
    ),
    TableRowRetrievalCase(
        case_id="TCEHY-REV-01",
        document_key="tcehy_2025_annual",
        query="腾讯2025年收入是多少？",
        expected_pages=(8, 9, 18, 130, 195, 196),
        expected_row_labels=("营业收入",),
        expected_metric_tags=("revenue", "income_statement"),
        notes="Tencent revenue row.",
    ),
    TableRowRetrievalCase(
        case_id="TCEHY-NP-01",
        document_key="tcehy_2025_annual",
        query="腾讯2025年本公司权益持有人应占盈利是多少？",
        expected_pages=(5, 8, 18, 207, 208),
        expected_row_labels=("归属股东净利润",),
        expected_metric_tags=("net_profit", "profit"),
        notes="Tencent profit attributable to equity holders.",
    ),
)


def build_table_row_query_result(
    *,
    case: TableRowRetrievalCase,
    mode: str,
    chunks: list[RetrievedChunk],
    retrieval_ms: int,
    top_k: int,
    fallback_used: bool = False,
    fallback_source: str | None = None,
) -> dict[str, Any]:
    retrieved_pages = [chunk.page_number for chunk in chunks if chunk.page_number is not None]
    row_labels = [chunk.row_label for chunk in chunks if chunk.row_label]
    metric_tags = sorted({tag for chunk in chunks for tag in chunk.metric_tags})

    page_hits = sorted(set(case.expected_pages).intersection(retrieved_pages))
    row_label_hits = sorted(set(case.expected_row_labels).intersection(row_labels))
    metric_tag_hits = sorted(set(case.expected_metric_tags).intersection(metric_tags))

    return {
        "id": case.case_id,
        "document_key": case.document_key,
        "mode": mode,
        "query": case.query,
        "expected_pages": list(case.expected_pages),
        "expected_row_labels": list(case.expected_row_labels),
        "expected_metric_tags": list(case.expected_metric_tags),
        "retrieved_pages": retrieved_pages,
        "retrieved_row_labels": row_labels,
        "retrieved_metric_tags": metric_tags,
        "hit_pages": page_hits,
        "hit_row_labels": row_label_hits,
        "hit_metric_tags": metric_tag_hits,
        f"page_hit@{top_k}": bool(page_hits),
        f"row_label_hit@{top_k}": bool(row_label_hits),
        f"metric_tag_hit@{top_k}": bool(metric_tag_hits),
        "retrieval_ms": retrieval_ms,
        "fallback_used": fallback_used,
        "fallback_source": fallback_source,
        "top_results": [
            {
                "rank": index + 1,
                "chunk_id": chunk.chunk_id,
                "chunk_kind": chunk.chunk_kind,
                "page_number": chunk.page_number,
                "row_label": chunk.row_label,
                "metric_tags": chunk.metric_tags,
                "period_hint": chunk.period_hint,
                "score": chunk.score,
                "preview": _preview(chunk.text),
            }
            for index, chunk in enumerate(chunks)
        ],
        "notes": case.notes,
    }


def summarize_table_row_results(
    query_results: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    total = len(query_results)
    page_key = f"page_hit@{top_k}"
    row_key = f"row_label_hit@{top_k}"
    metric_key = f"metric_tag_hit@{top_k}"

    page_hit_count = sum(1 for result in query_results if result[page_key])
    row_hit_count = sum(1 for result in query_results if result[row_key])
    metric_hit_count = sum(1 for result in query_results if result[metric_key])
    fallback_count = sum(1 for result in query_results if result["fallback_used"])

    return {
        "total_queries": total,
        f"{page_key}_count": page_hit_count,
        f"{page_key}_rate": _safe_ratio(page_hit_count, total),
        f"{row_key}_count": row_hit_count,
        f"{row_key}_rate": _safe_ratio(row_hit_count, total),
        f"{metric_key}_count": metric_hit_count,
        f"{metric_key}_rate": _safe_ratio(metric_hit_count, total),
        "fallback_count": fallback_count,
        "fallback_rate": _safe_ratio(fallback_count, total),
        "page_miss_cases": [result["id"] for result in query_results if not result[page_key]],
        "row_label_miss_cases": [
            result["id"] for result in query_results if not result[row_key]
        ],
        "metric_tag_miss_cases": [
            result["id"] for result in query_results if not result[metric_key]
        ],
    }


def combine_table_row_with_page_fallback(
    *,
    table_row_chunks: list[RetrievedChunk],
    page_text_chunks: list[RetrievedChunk],
    top_k: int,
) -> tuple[list[RetrievedChunk], bool, str]:
    if not table_row_chunks:
        return page_text_chunks[:top_k], True, "page_text"
    return _dedupe_chunks([*table_row_chunks, *page_text_chunks])[:top_k], False, "table_row"


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    deduped: list[RetrievedChunk] = []
    seen: set[tuple[str | None, int | None, str | None, str]] = set()
    for chunk in chunks:
        key = (
            chunk.chunk_kind,
            chunk.page_number,
            chunk.row_label,
            _normalize_for_match(chunk.text)[:180] if not chunk.row_label else "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _preview(text: str, *, limit: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()
