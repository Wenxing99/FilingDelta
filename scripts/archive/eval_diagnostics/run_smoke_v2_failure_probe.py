from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.eval.failure_probe import (
    FAILURE_PROBE_TARGET_QUERY_IDS,
    build_false_positive_summaries,
    build_gold_page_coverage,
    classify_failure_category,
    rank_expected_pages,
    render_failure_probe_markdown,
)
from filingdelta.eval.retrieval_diagnosis import (
    BM25Index,
    DIAGNOSIS_MODES,
    RetrievalCandidate,
    build_live_pilot_context_from_query,
    combine_strategy_candidates,
    diagnosis_strategy_for_case,
    evidence_units_to_retrieved_chunks,
    rank_semantic_chunks,
    reciprocal_rank_fusion,
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


DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_DIAGNOSIS_REPORT = Path("data/outputs/eval/golden_queries_v2_retrieval_diagnosis.json")
DEFAULT_PILOT_REPORT = Path("data/outputs/eval/golden_queries_v2_smoke_pilot_report.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/golden_queries_v2_failure_probe.json")
DEFAULT_MARKDOWN = Path("docs/golden_queries_v2_failure_probe.md")
DEFAULT_QDRANT_ROOT = Path("tmp/smoke-v2-failure-probe-qdrant")
DEFAULT_CANDIDATE_TOP_K = 20
DEFAULT_FINAL_TOP_K = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run failure attribution probe for unresolved smoke_v2 retrieval cases."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--diagnosis-report", type=Path, default=DEFAULT_DIAGNOSIS_REPORT)
    parser.add_argument("--pilot-report", type=Path, default=DEFAULT_PILOT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--qdrant-path", type=Path, default=None)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_CANDIDATE_TOP_K)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--case", dest="query_ids", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    if args.candidate_top_k < 1 or args.final_top_k < 1:
        raise SystemExit("--candidate-top-k and --final-top-k must be positive.")

    manifest = load_smoke_v2_manifest(_resolve_path(args.manifest), base_dir=REPO_ROOT)
    selected_query_ids = tuple(args.query_ids) if args.query_ids else FAILURE_PROBE_TARGET_QUERY_IDS
    cases = _select_probe_cases(manifest=manifest, query_ids=selected_query_ids)
    if not cases:
        raise SystemExit("No failure probe cases selected.")

    live_pilot_by_case_id, pilot_report_status = _load_live_pilot_context(
        _resolve_path(args.pilot_report)
    )
    diagnosis_report_status = _load_diagnosis_report_status(_resolve_path(args.diagnosis_report))
    report = run_failure_probe(
        manifest=manifest,
        cases=cases,
        candidate_top_k=args.candidate_top_k,
        final_top_k=args.final_top_k,
        qdrant_path=_resolve_probe_qdrant_path(args.qdrant_path),
        diagnosis_report_path=_resolve_path(args.diagnosis_report),
        diagnosis_report_status=diagnosis_report_status,
        pilot_report_path=_resolve_path(args.pilot_report),
        pilot_report_status=pilot_report_status,
        live_pilot_by_case_id=live_pilot_by_case_id,
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = _resolve_path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_failure_probe_markdown(report), encoding="utf-8")

    print("report:", output_path, flush=True)
    print("markdown:", markdown_path, flush=True)
    for case in report["cases"]:
        print(
            f"{case['query_id']}: {case['failure_category']} -> "
            f"{case['recommended_next_fix']}",
            flush=True,
        )
    return report


def run_failure_probe(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    candidate_top_k: int,
    final_top_k: int,
    qdrant_path: Path,
    diagnosis_report_path: Path,
    diagnosis_report_status: dict[str, Any],
    pilot_report_path: Path,
    pilot_report_status: dict[str, Any],
    live_pilot_by_case_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    started_at = time.perf_counter()
    qdrant_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(FILINGDELTA_QDRANT_PATH=_settings_path_value(qdrant_path))
    client = QdrantClient(path=str(qdrant_path))
    pipeline = FilingIngestionPipeline(settings=settings)
    indexer = DocumentChunkIndexer(settings=settings, client=client)
    retriever = DocumentChunkRetriever(settings=settings, client=client)

    try:
        contexts = _prepare_documents(
            manifest=manifest,
            cases=cases,
            pipeline=pipeline,
            indexer=indexer,
        )
        case_results = [
            _run_failure_case(
                case=case,
                context=contexts[case.document_key],
                retriever=retriever,
                candidate_top_k=candidate_top_k,
                final_top_k=final_top_k,
                live_pilot=live_pilot_by_case_id.get(
                    case.case_id,
                    unknown_live_pilot_context(),
                ),
            )
            for case in cases
        ]
    finally:
        client.close()

    return {
        "version": "golden_queries_v2_failure_probe.v1",
        "manifest_path": str(manifest.source_path),
        "retrieval_diagnosis_path": str(diagnosis_report_path),
        "retrieval_diagnosis_status": diagnosis_report_status,
        "pilot_report_path": str(pilot_report_path),
        "pilot_report_status": pilot_report_status,
        "target_query_ids": _selected_query_ids_from_cases(cases),
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "modes": list(DIAGNOSIS_MODES),
        "qdrant_path": str(qdrant_path),
        "total_wall_ms": _elapsed_ms(started_at),
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
            "parsed_filing": ingestion.parsed_filing,
            "evidence_units": ingestion.evidence_units,
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
                "total_pages": ingestion.parsed_filing.document.total_pages,
                "evidence_units": len(ingestion.evidence_units),
            },
        }
    return contexts


def _run_failure_case(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    candidate_top_k: int,
    final_top_k: int,
    live_pilot: dict[str, Any],
) -> dict[str, Any]:
    print(f"probing {case.case_id}", flush=True)
    candidates_by_mode = {
        mode: _retrieve_strategy_candidates(
            mode=mode,
            strategy=diagnosis_strategy_for_case(case),
            case=case,
            context=context,
            retriever=retriever,
            candidate_top_k=candidate_top_k,
        )
        for mode in DIAGNOSIS_MODES
    }
    gold_page_coverage = build_gold_page_coverage(
        parsed_filing=context["parsed_filing"],
        evidence_units=context["evidence_units"],
        expected_pages=case.expected_pages,
    )
    mode_rankings = {
        mode: rank_expected_pages(
            candidates=candidates,
            expected_pages=case.expected_pages,
            final_top_k=final_top_k,
        )
        for mode, candidates in candidates_by_mode.items()
    }
    top_false_positives = {
        mode: build_false_positive_summaries(
            query=case.query,
            candidates=candidates,
            expected_pages=case.expected_pages,
            final_top_k=final_top_k,
        )
        for mode, candidates in candidates_by_mode.items()
    }
    classification = classify_failure_category(
        expected_intent=case.expected_document_evidence_intent,
        live_observed_intent=live_pilot.get("pilot_observed_intent"),
        gold_page_coverage=gold_page_coverage,
        mode_rankings=mode_rankings,
        top_false_positives=top_false_positives,
    )

    return {
        "id": case.case_id,
        "query_id": case.case_id.rsplit("::", 1)[-1],
        "company": case.company,
        "industry": case.industry,
        "query": case.query,
        "expected_pages": list(case.expected_pages),
        "expected_intent": case.expected_document_evidence_intent,
        "expected_primary_evidence_kind": case.primary_evidence_kind,
        "expected_secondary_evidence_kinds": list(case.secondary_evidence_kinds),
        "pilot_status": live_pilot.get("pilot_status", "unknown"),
        "live_observed_intent": live_pilot.get("pilot_observed_intent"),
        "pilot_intent_hit": live_pilot.get("pilot_intent_hit"),
        "pilot_page_hit_at_6": live_pilot.get("pilot_page_hit_at_6"),
        "gold_page_coverage": gold_page_coverage,
        "mode_rankings": mode_rankings,
        "top_false_positive_pages": top_false_positives,
        **classification,
        "notes": case.notes,
    }


def _retrieve_strategy_candidates(
    *,
    mode: str,
    strategy,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    candidate_top_k: int,
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
        final_top_k=candidate_top_k,
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


def _select_probe_cases(
    *,
    manifest: SmokeV2Manifest,
    query_ids: tuple[str, ...],
) -> list[SmokeV2Case]:
    all_smoke_cases = select_smoke_v2_cases(manifest.queries, tiers={SMOKE_V2_TIER})
    by_query_id = {case.case_id.rsplit("::", 1)[-1]: case for case in all_smoke_cases}
    return [by_query_id[query_id] for query_id in query_ids if query_id in by_query_id]


def _selected_query_ids_from_cases(cases: list[SmokeV2Case]) -> list[str]:
    return [case.case_id.rsplit("::", 1)[-1] for case in cases]


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


def _load_diagnosis_report_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(path), "status": "invalid_json", "error": str(exc)}
    return {
        "path": str(path),
        "status": "loaded",
        "version": payload.get("version"),
        "diagnosis_scope": payload.get("diagnosis_scope"),
    }


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_probe_qdrant_path(path: Path | None) -> Path:
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
