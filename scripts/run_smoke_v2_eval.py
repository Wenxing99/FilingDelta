from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from filingdelta.agents.chat_router import ChatRouterAgent
from filingdelta.core.config import REPO_ROOT
from filingdelta.core.config import Settings
from filingdelta.eval.smoke_v2 import (
    SMOKE_V2_TIER,
    SmokeV2Manifest,
    SmokeV2Observation,
    build_builtin_placeholder_manifest_payload,
    build_smoke_v2_report,
    load_smoke_v2_manifest,
    load_smoke_v2_manifest_from_payload,
    render_smoke_v2_markdown_summary,
    select_smoke_v2_cases,
)
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.retrieval.indexer import DocumentChunkIndexer
from filingdelta.retrieval.page_text_hybrid import evidence_units_to_page_text_chunks
from filingdelta.retrieval.retriever import DocumentChunkRetriever
from filingdelta.schemas.filing import EvidenceKind
from filingdelta.services.chat_qa import (
    _retrieve_document_evidence,
    _select_document_retrieval_strategy,
)


DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/smoke_v2_report.json")
DEFAULT_QDRANT_ROOT = Path("tmp/smoke-v2-live-qdrant")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, dry-run, or run live retrieval for the smoke_v2 eval manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-summary", type=Path, default=None)
    parser.add_argument("--qdrant-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--case", dest="case_ids", action="append", default=[])
    parser.add_argument("--company", dest="companies", action="append", default=[])
    parser.add_argument("--industry", dest="industries", action="append", default=[])
    parser.add_argument("--intent", dest="intents", action="append", default=[])
    parser.add_argument("--tier", dest="tiers", action="append", default=[])
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--validate-only", action="store_true")
    mode_group.add_argument("--dry-run", action="store_true")
    mode_group.add_argument("--live-retrieval", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--use-built-in-placeholders",
        action="store_true",
        help="Use two CMB/Tencent placeholder cases derived from existing eval cases.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    _configure_stdio()

    args = build_parser().parse_args(argv)
    mode = _resolve_mode(args)
    manifest = _load_manifest(args)
    cases = select_smoke_v2_cases(
        manifest.queries,
        case_ids=set(args.case_ids) or None,
        tiers=set(args.tiers) if args.tiers else {SMOKE_V2_TIER},
        companies=set(args.companies) or None,
        industries=set(args.industries) or None,
        intents=set(args.intents) or None,
    )
    if args.list_cases:
        _print_cases(cases)
        return None
    if not cases:
        raise SystemExit("No smoke_v2 cases selected.")

    observations: dict[str, SmokeV2Observation] | None = None
    live_execution: dict[str, Any] | None = None
    if mode == "live_retrieval":
        observations, live_execution = asyncio.run(
            _build_live_retrieval_observations(
                manifest=manifest,
                cases=cases,
                top_k=args.top_k or manifest.default_top_k,
                qdrant_path=_resolve_live_qdrant_path(args.qdrant_path),
            )
        )

    report = build_smoke_v2_report(
        manifest=manifest,
        cases=cases,
        mode=mode,
        top_k=args.top_k,
        observations=observations,
    )
    if live_execution is not None:
        report["live_execution"] = live_execution
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown_summary is not None:
        markdown_path = _resolve_path(args.markdown_summary)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_smoke_v2_markdown_summary(report),
            encoding="utf-8",
        )

    summary = report["summary"]
    missing = report["documents"]["missing_document_keys"]
    print("report:", output_path, flush=True)
    if args.markdown_summary is not None:
        print("markdown_summary:", _resolve_path(args.markdown_summary), flush=True)
    print(
        f"mode={report['mode']} total_cases={summary['total_queries']} "
        f"missing_documents={len(missing)}",
        flush=True,
    )
    print("status_counts:", json.dumps(summary["status_counts"], ensure_ascii=False), flush=True)
    if missing:
        print("missing_document_keys:", ", ".join(missing), flush=True)
    return report


async def _build_live_retrieval_observations(
    *,
    manifest: SmokeV2Manifest,
    cases,
    top_k: int,
    qdrant_path: Path,
) -> tuple[dict[str, SmokeV2Observation], dict[str, Any]]:
    qdrant_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(FILINGDELTA_QDRANT_PATH=_settings_path_value(qdrant_path))
    client = QdrantClient(path=str(qdrant_path))
    pipeline = FilingIngestionPipeline(settings=settings)
    indexer = DocumentChunkIndexer(settings=settings, client=client)
    retriever = DocumentChunkRetriever(settings=settings, client=client)
    router = ChatRouterAgent(settings=settings)

    observations: dict[str, SmokeV2Observation] = {}
    document_ids: dict[str, str] = {}
    documents: dict[str, Any] = {}
    document_stats: dict[str, dict[str, Any]] = {}
    page_text_chunks_by_document_key: dict[str, list[Any]] = {}
    used_document_keys = sorted({case.document_key for case in cases})
    started_at = time.perf_counter()
    try:
        for document_key in used_document_keys:
            document = manifest.documents.require(document_key)
            if not document.exists:
                document_stats[document_key] = {
                    "source_path": str(document.source_path),
                    "exists": False,
                    "error": "source document not found",
                }
                continue

            print(f"parsing/indexing {document_key}", flush=True)
            parse_started = time.perf_counter()
            ingestion = pipeline.run(document.source)
            parse_ms = _elapsed_ms(parse_started)
            document_id = ingestion.parsed_filing.document.document_id
            document_ids[document_key] = document_id
            documents[document_key] = ingestion.parsed_filing.document
            page_text_chunks_by_document_key[document_key] = evidence_units_to_page_text_chunks(
                document_id=document_id,
                evidence_units=ingestion.evidence_units,
            )

            index_started = time.perf_counter()
            indexer.index_document(
                document_id=document_id,
                chunks=ingestion.chunks,
                evidence_units=ingestion.evidence_units,
            )
            index_ms = _elapsed_ms(index_started)
            document_stats[document_key] = {
                "document_id": document_id,
                "source_path": str(document.source_path),
                "exists": True,
                "parse_ms": parse_ms,
                "index_ms": index_ms,
                "page_text_count": _count_evidence_units(
                    ingestion.evidence_units,
                    EvidenceKind.PAGE_TEXT.value,
                ),
                "section_text_count": _count_evidence_units(
                    ingestion.evidence_units,
                    EvidenceKind.SECTION_TEXT.value,
                ),
                "table_row_count": _count_evidence_units(
                    ingestion.evidence_units,
                    EvidenceKind.TABLE_ROW.value,
                ),
            }

        for case in cases:
            print(f"running {case.case_id}", flush=True)
            case_started = time.perf_counter()
            document_id = document_ids.get(case.document_key)
            document = documents.get(case.document_key)
            if document_id is None or document is None:
                observations[case.case_id] = SmokeV2Observation(
                    executed=True,
                    error="source document was not indexed",
                    latency_ms=_elapsed_ms(case_started),
                )
                continue

            try:
                route_decision = await router.route(question=case.query, document=document)
                retrieved_chunks = []
                retrieval_mode = None
                if route_decision.route in {"document_only", "mixed"}:
                    strategy = _select_document_retrieval_strategy(
                        case.query,
                        route_decision=route_decision,
                        chat_retrieval_strategy=settings.filingdelta_chat_retrieval_strategy,
                    )
                    retrieved_chunks, retrieval_mode = _retrieve_document_evidence(
                        retriever=retriever,
                        document_id=document_id,
                        question=case.query,
                        callback_manager=None,
                        strategy=strategy,
                        page_text_chunks=page_text_chunks_by_document_key.get(case.document_key, []),
                    )
                observations[case.case_id] = _observation_from_live_result(
                    route=route_decision.route,
                    document_evidence_intent=route_decision.document_evidence_intent,
                    retrieval_mode=retrieval_mode,
                    chunks=retrieved_chunks,
                    latency_ms=_elapsed_ms(case_started),
                )
            except Exception as error:  # noqa: BLE001 - eval report should capture case failures.
                observations[case.case_id] = SmokeV2Observation(
                    executed=True,
                    error=f"{type(error).__name__}: {error}",
                    latency_ms=_elapsed_ms(case_started),
                )
    finally:
        client.close()

    return observations, {
        "qdrant_path": str(qdrant_path),
        "total_wall_ms": _elapsed_ms(started_at),
        "documents": document_stats,
    }


def _load_manifest(args: argparse.Namespace):
    if args.use_built_in_placeholders:
        payload = build_builtin_placeholder_manifest_payload()
        return load_smoke_v2_manifest_from_payload(payload, base_dir=REPO_ROOT)
    return load_smoke_v2_manifest(_resolve_path(args.manifest), base_dir=REPO_ROOT)


def _print_cases(cases) -> None:
    for case in cases:
        print(
            f"{case.case_id}\t{case.tier}\t{case.document_key}\t"
            f"{case.expected_document_evidence_intent}\t{case.query}",
            flush=True,
        )


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_live_qdrant_path(path: Path | None) -> Path:
    if path is not None:
        return _resolve_path(path)
    suffix = time.strftime("%Y%m%d-%H%M%S")
    return _resolve_path(DEFAULT_QDRANT_ROOT.with_name(f"{DEFAULT_QDRANT_ROOT.name}-{suffix}"))


def _resolve_mode(args: argparse.Namespace) -> str:
    if args.validate_only:
        return "validate_only"
    if args.dry_run:
        return "dry_run"
    return "live_retrieval"


def _settings_path_value(qdrant_path: Path) -> str:
    if qdrant_path.is_relative_to(REPO_ROOT):
        return str(qdrant_path.relative_to(REPO_ROOT))
    return str(qdrant_path)


def _observation_from_live_result(
    *,
    route: str,
    document_evidence_intent: str,
    retrieval_mode: str | None,
    chunks: list[Any],
    latency_ms: int,
) -> SmokeV2Observation:
    return SmokeV2Observation(
        executed=True,
        route=route,
        document_evidence_intent=document_evidence_intent,
        retrieval_mode=retrieval_mode,
        retrieved_evidence_kinds=tuple(
            chunk.chunk_kind for chunk in chunks if chunk.chunk_kind
        ),
        citation_pages=tuple(chunk.page_number for chunk in chunks if chunk.page_number),
        retrieved_row_labels=tuple(chunk.row_label for chunk in chunks if chunk.row_label),
        retrieved_metric_tags=tuple(
            sorted({tag for chunk in chunks for tag in chunk.metric_tags})
        ),
        retrieved_section_types=tuple(
            chunk.section_type for chunk in chunks if chunk.section_type
        ),
        latency_ms=latency_ms,
    )


def _count_evidence_units(evidence_units: list[Any], chunk_kind: str) -> int:
    return sum(1 for unit in evidence_units if unit.metadata.chunk_kind.value == chunk_kind)


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
