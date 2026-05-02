from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.eval.retrieval_diagnosis import (
    BM25Index,
    DIAGNOSIS_MODES,
    RetrievalCandidate,
    build_mode_result,
    build_live_pilot_context_from_query,
    build_original_failed_rescue,
    combine_strategy_candidates,
    diagnosis_strategy_for_case,
    enrich_case_result_with_diagnosis_context,
    evidence_units_to_retrieved_chunks,
    rank_semantic_chunks,
    reciprocal_rank_fusion,
    render_diagnosis_markdown,
    unknown_live_pilot_context,
)
from filingdelta.eval.smoke_v2 import (
    SMOKE_V2_TIER,
    SmokeV2Case,
    SmokeV2Manifest,
    load_smoke_v2_manifest,
    select_smoke_v2_cases,
)
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.retrieval.indexer import DocumentChunkIndexer
from filingdelta.retrieval.retriever import DocumentChunkRetriever
from filingdelta.schemas.chat import RetrievedChunk
from filingdelta.schemas.filing import EvidenceKind


DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_PILOT_REPORT = Path("data/outputs/eval/golden_queries_v2_smoke_pilot_report.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/golden_queries_v2_retrieval_diagnosis.json")
DEFAULT_MARKDOWN = Path("docs/golden_queries_v2_retrieval_diagnosis.md")
DEFAULT_QDRANT_ROOT = Path("tmp/smoke-v2-retrieval-diagnosis-qdrant")
DEFAULT_CANDIDATE_TOP_K = 20
DEFAULT_FINAL_TOP_K = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run smoke_v2 retrieval diagnosis across semantic, BM25, and hybrid modes."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pilot-report", type=Path, default=DEFAULT_PILOT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--qdrant-path", type=Path, default=None)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--case", dest="case_ids", action="append", default=[])
    parser.add_argument("--company", dest="companies", action="append", default=[])
    parser.add_argument("--industry", dest="industries", action="append", default=[])
    parser.add_argument("--intent", dest="intents", action="append", default=[])
    parser.add_argument("--list-cases", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    _configure_stdio()
    args = build_parser().parse_args(argv)

    manifest = load_smoke_v2_manifest(_resolve_path(args.manifest), base_dir=REPO_ROOT)
    cases = select_smoke_v2_cases(
        manifest.queries,
        case_ids=set(args.case_ids) or None,
        tiers={SMOKE_V2_TIER},
        companies=set(args.companies) or None,
        industries=set(args.industries) or None,
        intents=set(args.intents) or None,
    )
    if args.list_cases:
        _print_cases(cases)
        return None
    if not cases:
        raise SystemExit("No smoke_v2 cases selected.")
    if args.candidate_top_k < 1 or args.final_top_k < 1:
        raise SystemExit("--candidate-top-k and --final-top-k must be positive.")

    live_pilot_by_case_id, live_pilot_report = _load_live_pilot_context(
        _resolve_path(args.pilot_report)
    )
    report = run_retrieval_diagnosis(
        manifest=manifest,
        cases=cases,
        candidate_top_k=args.candidate_top_k,
        final_top_k=args.final_top_k,
        qdrant_path=_resolve_diagnosis_qdrant_path(args.qdrant_path),
        live_pilot_by_case_id=live_pilot_by_case_id,
        live_pilot_report=live_pilot_report,
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = _resolve_path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_diagnosis_markdown(report), encoding="utf-8")

    print("report:", output_path, flush=True)
    print("markdown:", markdown_path, flush=True)
    for mode, mode_summary in report["summary"]["mode_summary"].items():
        print(
            f"{mode}: page_hit={mode_summary['hit_count']}/"
            f"{mode_summary['total_cases']}",
            flush=True,
        )
    return report


def run_retrieval_diagnosis(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    candidate_top_k: int,
    final_top_k: int,
    qdrant_path: Path,
    live_pilot_by_case_id: dict[str, dict[str, Any]] | None = None,
    live_pilot_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    qdrant_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(FILINGDELTA_QDRANT_PATH=_settings_path_value(qdrant_path))
    client = QdrantClient(path=str(qdrant_path))
    pipeline = FilingIngestionPipeline(settings=settings)
    indexer = DocumentChunkIndexer(settings=settings, client=client)
    retriever = DocumentChunkRetriever(settings=settings, client=client)

    try:
        document_contexts = _prepare_documents(
            manifest=manifest,
            cases=cases,
            pipeline=pipeline,
            indexer=indexer,
        )
        case_results = [
            _run_case(
                case=case,
                context=document_contexts[case.document_key],
                retriever=retriever,
                candidate_top_k=candidate_top_k,
                final_top_k=final_top_k,
                live_pilot=(
                    live_pilot_by_case_id or {}
                ).get(case.case_id, unknown_live_pilot_context()),
            )
            for case in cases
        ]
    finally:
        client.close()

    summary = _build_summary(case_results)
    return {
        "version": "golden_queries_v2_retrieval_diagnosis.v1",
        "manifest_path": str(manifest.source_path),
        "manifest_version": manifest.version,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "rrf_k": 60,
        "modes": list(DIAGNOSIS_MODES),
        "diagnosis_scope": {
            "id": "expected_intent_diagnosis/page_hit_only",
            "strategy_source": "manifest_expected_intent",
            "scoring": "page_hit_only",
            "full_live_pilot_rescue_claimed": False,
        },
        "live_pilot_report": live_pilot_report
        or {"path": None, "status": "unknown", "loaded_cases": 0},
        "qdrant_path": str(qdrant_path),
        "total_wall_ms": _elapsed_ms(started_at),
        "summary": summary,
        "documents": {
            key: context["stats"]
            for key, context in sorted(document_contexts.items())
        },
        "original_failed_case_rescue": build_original_failed_rescue(case_results),
        "cases": case_results,
    }


def _prepare_documents(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    pipeline: FilingIngestionPipeline,
    indexer: DocumentChunkIndexer,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for document_key in sorted({case.document_key for case in cases}):
        document = manifest.documents.require(document_key)
        if not document.exists:
            raise SystemExit(f"source document not found: {document.source_path}")

        print(f"parsing/indexing {document_key}", flush=True)
        parse_started = time.perf_counter()
        ingestion = pipeline.run(document.source)
        parse_ms = _elapsed_ms(parse_started)
        document_id = ingestion.parsed_filing.document.document_id

        index_started = time.perf_counter()
        indexer.index_document(
            document_id=document_id,
            chunks=ingestion.chunks,
            evidence_units=ingestion.evidence_units,
        )
        index_ms = _elapsed_ms(index_started)

        chunks = evidence_units_to_retrieved_chunks(
            document_id=document_id,
            evidence_units=ingestion.evidence_units,
        )
        chunks_by_kind: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            if chunk.chunk_kind is None:
                continue
            chunks_by_kind.setdefault(chunk.chunk_kind, []).append(chunk)

        contexts[document_key] = {
            "document_id": document_id,
            "chunks_by_kind": chunks_by_kind,
            "bm25_by_kind": {
                kind: BM25Index(kind_chunks)
                for kind, kind_chunks in chunks_by_kind.items()
            },
            "stats": {
                "document_id": document_id,
                "source_path": str(document.source_path),
                "parse_ms": parse_ms,
                "index_ms": index_ms,
                "page_text_count": len(chunks_by_kind.get(EvidenceKind.PAGE_TEXT.value, [])),
                "section_text_count": len(chunks_by_kind.get(EvidenceKind.SECTION_TEXT.value, [])),
                "table_row_count": len(chunks_by_kind.get(EvidenceKind.TABLE_ROW.value, [])),
            },
        }
    return contexts


def _run_case(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    candidate_top_k: int,
    final_top_k: int,
    live_pilot: dict[str, Any] | None,
) -> dict[str, Any]:
    print(f"running {case.case_id}", flush=True)
    strategy = diagnosis_strategy_for_case(case)
    mode_results: dict[str, dict[str, Any]] = {}
    for mode in DIAGNOSIS_MODES:
        started_at = time.perf_counter()
        candidates = _retrieve_strategy_candidates(
            mode=mode,
            strategy=strategy,
            case=case,
            context=context,
            retriever=retriever,
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
        )
        mode_results[mode] = build_mode_result(
            case=case,
            mode=mode,
            candidates=candidates,
            final_top_k=final_top_k,
            retrieval_ms=_elapsed_ms(started_at),
        )

    case_result = {
        "id": case.case_id,
        "query_id": case.case_id.rsplit("::", 1)[-1],
        "tier": case.tier,
        "document_key": case.document_key,
        "company": case.company,
        "industry": case.industry,
        "query": case.query,
        "expected_route": case.expected_route,
        "expected_document_evidence_intent": case.expected_document_evidence_intent,
        "primary_evidence_kind": case.primary_evidence_kind,
        "secondary_evidence_kinds": list(case.secondary_evidence_kinds),
        "expected_pages": list(case.expected_pages),
        "strategy": {
            "primary_chunk_kind": strategy.primary_chunk_kind,
            "fallback_chunk_kinds": list(strategy.fallback_chunk_kinds),
            "include_fallback_when_primary_found": strategy.include_fallback_when_primary_found,
            "primary_top_k": strategy.primary_top_k,
            "fallback_top_k": strategy.fallback_top_k,
        },
        "modes": mode_results,
        "notes": case.notes,
    }
    return enrich_case_result_with_diagnosis_context(case_result, live_pilot=live_pilot)


def _retrieve_strategy_candidates(
    *,
    mode: str,
    strategy,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    candidate_top_k: int,
    final_top_k: int,
) -> list[RetrievalCandidate]:
    primary_top_k = _kind_candidate_top_k(
        mode=mode,
        strategy_top_k=strategy.primary_top_k,
        candidate_top_k=candidate_top_k,
    )
    fallback_top_k = _kind_candidate_top_k(
        mode=mode,
        strategy_top_k=strategy.fallback_top_k,
        candidate_top_k=candidate_top_k,
    )
    primary = _retrieve_kind_candidates(
        mode=mode,
        chunk_kind=strategy.primary_chunk_kind,
        case=case,
        context=context,
        retriever=retriever,
        candidate_top_k=primary_top_k,
    )
    fallback: list[RetrievalCandidate] = []
    for fallback_kind in strategy.fallback_chunk_kinds:
        fallback.extend(
            _retrieve_kind_candidates(
                mode=mode,
                chunk_kind=fallback_kind,
                case=case,
                context=context,
                retriever=retriever,
                candidate_top_k=fallback_top_k,
            )
        )
    return combine_strategy_candidates(
        primary_candidates=primary,
        fallback_candidates=fallback,
        strategy=strategy,
        final_top_k=final_top_k,
    )


def _kind_candidate_top_k(*, mode: str, strategy_top_k: int, candidate_top_k: int) -> int:
    if mode == "semantic_only":
        return strategy_top_k
    return candidate_top_k


def _retrieve_kind_candidates(
    *,
    mode: str,
    chunk_kind: str,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    candidate_top_k: int,
) -> list[RetrievalCandidate]:
    if mode == "semantic_only":
        return _semantic_candidates(
            retriever=retriever,
            document_id=context["document_id"],
            case=case,
            chunk_kind=chunk_kind,
            candidate_top_k=candidate_top_k,
        )
    if mode == "bm25_only":
        return _bm25_candidates(
            context=context,
            case=case,
            chunk_kind=chunk_kind,
            candidate_top_k=candidate_top_k,
        )
    semantic = _semantic_candidates(
        retriever=retriever,
        document_id=context["document_id"],
        case=case,
        chunk_kind=chunk_kind,
        candidate_top_k=candidate_top_k,
    )
    bm25 = _bm25_candidates(
        context=context,
        case=case,
        chunk_kind=chunk_kind,
        candidate_top_k=candidate_top_k,
    )
    return reciprocal_rank_fusion(semantic, bm25)


def _semantic_candidates(
    *,
    retriever: DocumentChunkRetriever,
    document_id: str,
    case: SmokeV2Case,
    chunk_kind: str,
    candidate_top_k: int,
) -> list[RetrievalCandidate]:
    chunks = retriever.retrieve(
        document_id=document_id,
        question=case.query,
        top_k=candidate_top_k,
        chunk_kind=chunk_kind,
    )
    return rank_semantic_chunks(chunks, top_k=candidate_top_k)


def _bm25_candidates(
    *,
    context: dict[str, Any],
    case: SmokeV2Case,
    chunk_kind: str,
    candidate_top_k: int,
) -> list[RetrievalCandidate]:
    index = context["bm25_by_kind"].get(chunk_kind)
    if index is None:
        return []
    return index.search(case.query, top_k=candidate_top_k)


def _build_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    mode_summary: dict[str, dict[str, Any]] = {}
    for mode in DIAGNOSIS_MODES:
        total = len(case_results)
        hit_count = sum(1 for result in case_results if result["modes"][mode]["hit"])
        mode_summary[mode] = {
            "total_cases": total,
            "hit_count": hit_count,
            "miss_count": total - hit_count,
            "hit_rate": hit_count / total if total else 0.0,
            "miss_cases": [
                result["id"] for result in case_results if not result["modes"][mode]["hit"]
            ],
        }
    return {
        "total_cases": len(case_results),
        "mode_summary": mode_summary,
        "mode_hits": {
            mode: f"{payload['hit_count']}/{payload['total_cases']}"
            for mode, payload in mode_summary.items()
        },
    }


def _print_cases(cases: list[SmokeV2Case]) -> None:
    for case in cases:
        print(
            f"{case.case_id}\t{case.tier}\t{case.document_key}\t"
            f"{case.expected_document_evidence_intent}\t{case.query}",
            flush=True,
        )


def _load_live_pilot_context(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return {}, {"path": str(path), "status": "missing", "loaded_cases": 0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, {
            "path": str(path),
            "status": "invalid_json",
            "loaded_cases": 0,
            "error": str(exc),
        }

    contexts: dict[str, dict[str, Any]] = {}
    for query_result in payload.get("queries") or []:
        case_id = query_result.get("id")
        if not case_id:
            continue
        contexts[case_id] = build_live_pilot_context_from_query(query_result)
    return contexts, {"path": str(path), "status": "loaded", "loaded_cases": len(contexts)}


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_diagnosis_qdrant_path(path: Path | None) -> Path:
    if path is not None:
        return _resolve_path(path)
    suffix = time.strftime("%Y%m%d-%H%M%S")
    return _resolve_path(DEFAULT_QDRANT_ROOT.with_name(f"{DEFAULT_QDRANT_ROOT.name}-{suffix}"))


def _settings_path_value(qdrant_path: Path) -> str:
    if qdrant_path.is_relative_to(REPO_ROOT):
        return str(qdrant_path.relative_to(REPO_ROOT))
    return str(qdrant_path)


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
