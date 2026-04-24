from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.eval.table_row_retrieval import (
    TABLE_ROW_RETRIEVAL_CASES,
    TableRowRetrievalCase,
    build_table_row_query_result,
    combine_table_row_with_page_fallback,
    summarize_table_row_results,
)
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.retrieval.indexer import DocumentChunkIndexer
from filingdelta.retrieval.retriever import DocumentChunkRetriever
from filingdelta.schemas.filing import EvidenceKind, FilingDocType, FilingSource, Market


DEFAULT_OUTPUT = Path("data/outputs/eval/table_row_retrieval_eval.json")
DEFAULT_QDRANT_PATH = Path("data/outputs/eval/qdrant_table_row_eval")
MODES = ("page_text_only", "table_row_only", "table_row_first_with_page_fallback")


@dataclass(frozen=True)
class EvalDocument:
    document_key: str
    source: FilingSource


DOCUMENTS: dict[str, EvalDocument] = {
    "cmb_2025_annual": EvalDocument(
        document_key="cmb_2025_annual",
        source=FilingSource(
            source_path=REPO_ROOT / "data" / "raw" / "招商银行2025年度报告.pdf",
            company_name="招商银行",
            market=Market.A_SHARE,
            doc_type=FilingDocType.ANNUAL_REPORT,
            fiscal_period="2025年度报告",
            language="zh",
        ),
    ),
    "tcehy_2025_annual": EvalDocument(
        document_key="tcehy_2025_annual",
        source=FilingSource(
            source_path=REPO_ROOT / "data" / "raw" / "腾讯控股2025年度报告.pdf",
            company_name="腾讯控股",
            market=Market.H_SHARE,
            doc_type=FilingDocType.ANNUAL_REPORT,
            fiscal_period="2025年度报告",
            language="zh",
        ),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run table_row retrieval-only eval for FilingDelta."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        help="Run a specific case id. Can be passed multiple times.",
    )
    parser.add_argument("--list-cases", action="store_true")
    return parser


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    cases = _select_cases(args.case_ids)
    if args.list_cases:
        _print_cases(cases)
        return

    if args.fresh and args.reuse_index:
        raise SystemExit("--fresh and --reuse-index cannot be used together.")

    output_path = _resolve_path(args.output)
    qdrant_path = _resolve_path(args.qdrant_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    used_document_keys = sorted({case.document_key for case in cases})
    document_ids: dict[str, str] = {}
    document_stats: dict[str, dict[str, Any]] = {}

    for document_key in used_document_keys:
        document = DOCUMENTS[document_key]
        print(f"parsing/indexing {document_key}: {document.source.source_path}", flush=True)
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

        document_stats[document_key] = {
            "document_id": document_id,
            "source_path": str(document.source.source_path),
            "parse_ms": parse_ms,
            "index_ms": index_ms,
            "page_text_count": sum(
                1
                for unit in ingestion.evidence_units
                if unit.metadata.chunk_kind == EvidenceKind.PAGE_TEXT
            ),
            "section_text_count": sum(
                1
                for unit in ingestion.evidence_units
                if unit.metadata.chunk_kind == EvidenceKind.SECTION_TEXT
            ),
            "table_row_count": sum(
                1
                for unit in ingestion.evidence_units
                if unit.metadata.chunk_kind == EvidenceKind.TABLE_ROW
            ),
            "reused_index": args.reuse_index,
        }

    mode_results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for case in cases:
        document_id = document_ids[case.document_key]
        print(f"running {case.case_id}: {case.query}", flush=True)

        page_chunks, page_ms = _run_mode(
            retriever=retriever,
            document_id=document_id,
            question=case.query,
            top_k=args.top_k,
            chunk_kind=EvidenceKind.PAGE_TEXT.value,
        )
        table_chunks, table_ms = _run_mode(
            retriever=retriever,
            document_id=document_id,
            question=case.query,
            top_k=max(args.top_k, 8),
            chunk_kind=EvidenceKind.TABLE_ROW.value,
        )

        mode_results["page_text_only"].append(
            build_table_row_query_result(
                case=case,
                mode="page_text_only",
                chunks=page_chunks[: args.top_k],
                retrieval_ms=page_ms,
                top_k=args.top_k,
            )
        )
        mode_results["table_row_only"].append(
            build_table_row_query_result(
                case=case,
                mode="table_row_only",
                chunks=table_chunks[: args.top_k],
                retrieval_ms=table_ms,
                top_k=args.top_k,
            )
        )

        combined_chunks, fallback_used, fallback_source = combine_table_row_with_page_fallback(
            table_row_chunks=table_chunks,
            page_text_chunks=page_chunks,
            top_k=args.top_k,
        )
        mode_results["table_row_first_with_page_fallback"].append(
            build_table_row_query_result(
                case=case,
                mode="table_row_first_with_page_fallback",
                chunks=combined_chunks,
                retrieval_ms=table_ms + (page_ms if fallback_used else 0),
                top_k=args.top_k,
                fallback_used=fallback_used,
                fallback_source=fallback_source,
            )
        )

    report = {
        "version": "table_row_retrieval_eval_v1",
        "top_k": args.top_k,
        "documents": document_stats,
        "modes": {
            mode: {
                "summary": summarize_table_row_results(mode_results[mode], top_k=args.top_k),
                "queries": mode_results[mode],
            }
            for mode in MODES
        },
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("report:", output_path, flush=True)
    for mode in MODES:
        summary = report["modes"][mode]["summary"]
        print(
            f"{mode}: "
            f"page_hit@{args.top_k}={summary[f'page_hit@{args.top_k}_count']}/{summary['total_queries']} "
            f"({summary[f'page_hit@{args.top_k}_rate']:.3f}), "
            f"row_label_hit@{args.top_k}={summary[f'row_label_hit@{args.top_k}_count']}/{summary['total_queries']} "
            f"({summary[f'row_label_hit@{args.top_k}_rate']:.3f}), "
            f"metric_tag_hit@{args.top_k}={summary[f'metric_tag_hit@{args.top_k}_count']}/{summary['total_queries']} "
            f"({summary[f'metric_tag_hit@{args.top_k}_rate']:.3f}), "
            f"fallback_rate={summary['fallback_rate']:.3f}",
            flush=True,
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


def _select_cases(case_ids: list[str]) -> list[TableRowRetrievalCase]:
    selected = list(TABLE_ROW_RETRIEVAL_CASES)
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in selected})
        if missing:
            raise SystemExit(f"Unknown case id(s): {', '.join(missing)}")
    return selected


def _print_cases(cases: list[TableRowRetrievalCase]) -> None:
    for case in cases:
        print(f"{case.case_id}\t{case.document_key}\t{case.query}", flush=True)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


if __name__ == "__main__":
    main()
