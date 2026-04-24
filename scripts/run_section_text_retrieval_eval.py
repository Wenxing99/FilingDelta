from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.retrieval.indexer import DocumentChunkIndexer
from filingdelta.retrieval.retriever import DocumentChunkRetriever
from filingdelta.schemas.filing import EvidenceKind, FilingDocType, FilingSource, Market


DEFAULT_MANIFEST = Path("data/outputs/eval/section_text_queries_v1.json")
DEFAULT_REPORT = Path("data/outputs/eval/section_text_retrieval_eval.json")
DEFAULT_QDRANT_PATH = Path("data/outputs/eval/qdrant_section_text_eval")
MODES = ("page_text_only", "section_text_only", "section_text_first_with_page_fallback")

_GENERIC_WRAPPER_TITLES = {
    "第一章 公司简介",
    "第二章 会计数据和财务指标摘要",
    "第三章 管理层讨论与分析",
    "管理层讨论与分析",
    "管理層討論及分析",
    "企業管治報告",
    "第八章 财务报告",
}
_GENERIC_WRAPPER_SUBSTRINGS = (
    "管理层讨论与分析",
    "管理層討論及分析",
    "企業管治報告",
    "公司简介",
    "公司簡介",
    "财务报告",
    "財務報告",
)


@dataclass(frozen=True)
class EvalDocument:
    document_key: str
    source: FilingSource


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    document_key: str
    query: str
    expected_pages: set[int]
    expected_section_type: str
    notes: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline section_text retrieval comparison for FilingDelta."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    return parser


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    manifest_path = _resolve_path(args.manifest)
    output_path = _resolve_path(args.output)
    qdrant_path = _resolve_path(args.qdrant_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(manifest_path)
    top_k = int(args.top_k or manifest.get("default_top_k") or 6)
    if args.fresh and args.reuse_index:
        raise SystemExit("--fresh and --reuse-index cannot be used together.")
    if args.fresh and qdrant_path.exists():
        shutil.rmtree(qdrant_path)
    qdrant_path.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        FILINGDELTA_QDRANT_PATH=str(qdrant_path.relative_to(REPO_ROOT))
        if qdrant_path.is_relative_to(REPO_ROOT)
        else str(qdrant_path)
    )
    client = QdrantClient(path=str(qdrant_path))
    pipeline = FilingIngestionPipeline(settings=settings)
    indexer = DocumentChunkIndexer(settings=settings, client=client)
    retriever = DocumentChunkRetriever(settings=settings, client=client)

    documents = _load_documents(manifest)
    queries = _load_queries(manifest)

    document_ids: dict[str, str] = {}
    document_stats: dict[str, dict[str, Any]] = {}
    for document_key in sorted(documents):
        document = documents[document_key]
        print(f"parsing/indexing {document_key}: {document.source.source_path}")
        parse_started = time.perf_counter()
        ingestion = pipeline.run(document.source)
        parse_ms = _elapsed_ms(parse_started)
        document_id = ingestion.parsed_filing.document.document_id
        document_ids[document_key] = document_id

        index_ms = None
        if not args.reuse_index:
            index_started = time.perf_counter()
            indexer.index_document(
                document_id=document_id,
                chunks=ingestion.chunks,
                evidence_units=ingestion.evidence_units,
            )
            index_ms = _elapsed_ms(index_started)

        section_count = sum(
            1 for unit in ingestion.evidence_units if unit.metadata.chunk_kind == EvidenceKind.SECTION_TEXT
        )
        page_count = sum(
            1 for unit in ingestion.evidence_units if unit.metadata.chunk_kind == EvidenceKind.PAGE_TEXT
        )
        document_stats[document_key] = {
            "document_id": document_id,
            "source_path": str(document.source.source_path),
            "parse_ms": parse_ms,
            "index_ms": index_ms,
            "page_text_count": page_count,
            "section_text_count": section_count,
            "reused_index": args.reuse_index,
        }

    mode_results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for query in queries:
        document_id = document_ids[query.document_key]

        page_chunks, page_ms = _run_mode(
            retriever=retriever,
            document_id=document_id,
            question=query.query,
            top_k=top_k,
            chunk_kind=EvidenceKind.PAGE_TEXT.value,
        )
        mode_results["page_text_only"].append(
            _build_query_result(
                query=query,
                mode="page_text_only",
                chunks=page_chunks,
                retrieval_ms=page_ms,
                fallback_used=False,
                top_k=top_k,
            )
        )

        section_chunks, section_ms = _run_mode(
            retriever=retriever,
            document_id=document_id,
            question=query.query,
            top_k=top_k,
            chunk_kind=EvidenceKind.SECTION_TEXT.value,
        )
        mode_results["section_text_only"].append(
            _build_query_result(
                query=query,
                mode="section_text_only",
                chunks=section_chunks,
                retrieval_ms=section_ms,
                fallback_used=False,
                top_k=top_k,
            )
        )

        fallback_started = time.perf_counter()
        fallback_used = _should_fallback(section_chunks)
        if fallback_used:
            combined_chunks = page_chunks
            fallback_source = "page_text"
        else:
            combined_chunks = section_chunks
            fallback_source = "section_text"
        fallback_ms = _elapsed_ms(fallback_started) + section_ms + (page_ms if fallback_used else 0)
        mode_results["section_text_first_with_page_fallback"].append(
            _build_query_result(
                query=query,
                mode="section_text_first_with_page_fallback",
                chunks=combined_chunks,
                retrieval_ms=fallback_ms,
                fallback_used=fallback_used,
                fallback_source=fallback_source,
                top_k=top_k,
            )
        )

    report = {
        "manifest_path": str(manifest_path),
        "manifest_version": manifest.get("version"),
        "top_k": top_k,
        "documents": document_stats,
        "modes": {
            mode: {
                "summary": _build_summary(mode_results[mode], top_k=top_k),
                "queries": mode_results[mode],
            }
            for mode in MODES
        },
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("report:", output_path)
    for mode in MODES:
        summary = report["modes"][mode]["summary"]
        print(
            f"{mode}: "
            f"page_hit@{top_k}={summary[f'page_hit@{top_k}_count']}/{summary['total_queries']} "
            f"({summary[f'page_hit@{top_k}_rate']:.3f}), "
            f"section_type_hit@{top_k}={summary[f'section_type_hit@{top_k}_count']}/{summary['total_queries']} "
            f"({summary[f'section_type_hit@{top_k}_rate']:.3f}), "
            f"fallback_rate={summary['fallback_rate']:.3f}"
        )
    client.close()


def _run_mode(
    *,
    retriever: DocumentChunkRetriever,
    document_id: str,
    question: str,
    top_k: int,
    chunk_kind: str,
):
    started = time.perf_counter()
    chunks = retriever.retrieve(
        document_id=document_id,
        question=question,
        top_k=top_k,
        chunk_kind=chunk_kind,
    )
    return chunks, _elapsed_ms(started)


def _build_query_result(
    *,
    query: EvalQuery,
    mode: str,
    chunks: list[Any],
    retrieval_ms: int,
    fallback_used: bool,
    top_k: int,
    fallback_source: str | None = None,
) -> dict[str, Any]:
    retrieved_pages = [chunk.page_number for chunk in chunks if chunk.page_number is not None]
    retrieved_page_set = set(retrieved_pages)
    hit_pages = sorted(query.expected_pages & retrieved_page_set)
    section_type_hit = any(chunk.section_type == query.expected_section_type for chunk in chunks)
    top1_title_quality = _title_quality(chunks[0].section_title) if chunks and chunks[0].chunk_kind == EvidenceKind.SECTION_TEXT.value else None

    return {
        "id": query.query_id,
        "document_key": query.document_key,
        "mode": mode,
        "query": query.query,
        "expected_pages": sorted(query.expected_pages),
        "expected_section_type": query.expected_section_type,
        "retrieved_pages": retrieved_pages,
        "hit_pages": hit_pages,
        f"page_hit@{top_k}": bool(hit_pages),
        f"section_type_hit@{top_k}": section_type_hit,
        "retrieval_ms": retrieval_ms,
        "fallback_used": fallback_used,
        "fallback_source": fallback_source,
        "top1_title_quality": top1_title_quality,
        "top_results": [
            {
                "rank": index + 1,
                "chunk_id": chunk.chunk_id,
                "chunk_kind": chunk.chunk_kind,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "section_type": chunk.section_type,
                "score": chunk.score,
                "preview": _preview(chunk.text),
            }
            for index, chunk in enumerate(chunks)
        ],
        "notes": query.notes,
    }


def _should_fallback(section_chunks: list[Any]) -> bool:
    if not section_chunks:
        return True
    top_three = section_chunks[:3]
    qualities = [_title_quality(chunk.section_title) for chunk in top_three]
    if any(quality in {"strong", "acceptable"} for quality in qualities):
        return False
    return all(quality == "junk" for quality in qualities)


def _title_quality(title: str | None) -> str:
    clean = " ".join((title or "").split())
    if not clean:
        return "junk"
    if clean in _GENERIC_WRAPPER_TITLES or any(token in clean for token in _GENERIC_WRAPPER_SUBSTRINGS):
        return "weak"
    if _looks_like_noise(clean):
        return "junk"
    if len(clean) <= 4:
        return "acceptable"
    return "strong"


def _looks_like_noise(title: str) -> bool:
    normalized = _normalize_for_match(title)
    if not normalized:
        return True
    if normalized.startswith(("略愿景", "務收入加速增長")):
        return True
    if title.endswith(("…", "...")) or "..." in title:
        return True
    if re.match(r"^\d+(?:\.\d+)?[%％]$", title):
        return True
    if re.match(r"^\d{1,2}月\d{1,2}日$", title):
        return True
    if re.match(r"^\d+(?:\.\d+)?(?:港元|亿元|億元|百萬元|百万元|万元|萬元|次)$", title):
        return True
    if any(token in title for token in ("樓", "楼", "街", "路", "號", "号")):
        return True
    return False


def _build_summary(query_results: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    page_hit_key = f"page_hit@{top_k}"
    section_hit_key = f"section_type_hit@{top_k}"
    total = len(query_results)
    page_hit_count = sum(1 for result in query_results if result[page_hit_key])
    section_hit_count = sum(1 for result in query_results if result[section_hit_key])
    fallback_count = sum(1 for result in query_results if result["fallback_used"])
    title_quality_counts = Counter(
        result["top1_title_quality"]
        for result in query_results
        if result["top1_title_quality"] is not None
    )
    by_document: dict[str, dict[str, Any]] = {}
    for result in query_results:
        bucket = by_document.setdefault(
            result["document_key"],
            {
                "total_queries": 0,
                f"{page_hit_key}_count": 0,
                f"{page_hit_key}_rate": 0.0,
                f"{section_hit_key}_count": 0,
                f"{section_hit_key}_rate": 0.0,
            },
        )
        bucket["total_queries"] += 1
        if result[page_hit_key]:
            bucket[f"{page_hit_key}_count"] += 1
        if result[section_hit_key]:
            bucket[f"{section_hit_key}_count"] += 1

    for bucket in by_document.values():
        total_queries = bucket["total_queries"]
        bucket[f"{page_hit_key}_rate"] = (
            bucket[f"{page_hit_key}_count"] / total_queries if total_queries else 0.0
        )
        bucket[f"{section_hit_key}_rate"] = (
            bucket[f"{section_hit_key}_count"] / total_queries if total_queries else 0.0
        )

    return {
        "total_queries": total,
        f"{page_hit_key}_count": page_hit_count,
        f"{page_hit_key}_rate": page_hit_count / total if total else 0.0,
        f"{section_hit_key}_count": section_hit_count,
        f"{section_hit_key}_rate": section_hit_count / total if total else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / total if total else 0.0,
        "top1_title_quality_counts": dict(title_quality_counts),
        "miss_cases": [result["id"] for result in query_results if not result[page_hit_key]],
        "section_type_miss_cases": [result["id"] for result in query_results if not result[section_hit_key]],
        "by_document": by_document,
    }


def _load_documents(manifest: dict[str, Any]) -> dict[str, EvalDocument]:
    documents: dict[str, EvalDocument] = {}
    for payload in manifest.get("documents", []):
        source_path = _resolve_path(Path(payload["source_path"]))
        source = FilingSource(
            source_path=source_path,
            company_name=payload["company_name"],
            ticker=payload.get("ticker"),
            market=Market(payload.get("market") or Market.OTHER.value),
            doc_type=FilingDocType(payload.get("doc_type") or FilingDocType.OTHER.value),
            fiscal_period=payload.get("fiscal_period"),
            language=payload.get("language") or "zh",
        )
        document_key = str(payload["document_key"])
        documents[document_key] = EvalDocument(document_key=document_key, source=source)
    return documents


def _load_queries(manifest: dict[str, Any]) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    for payload in manifest.get("queries", []):
        queries.append(
            EvalQuery(
                query_id=str(payload["id"]),
                document_key=str(payload["document_key"]),
                query=str(payload["query"]),
                expected_pages={int(page) for page in payload.get("expected_pages", [])},
                expected_section_type=str(payload["expected_section_type"]),
                notes=str(payload.get("notes") or ""),
            )
        )
    return queries


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _preview(text: str, *, limit: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _normalize_for_match(text: str) -> str:
    return "".join(text.lower().split())


if __name__ == "__main__":
    main()
