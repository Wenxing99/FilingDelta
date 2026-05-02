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
from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.eval.retrieval_diagnosis import (
    BM25Index,
    RankSource,
    RetrievalCandidate,
    candidate_to_json,
    dedupe_candidates,
    dedupe_key,
    evidence_units_to_retrieved_chunks,
    rank_semantic_chunks,
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
from filingdelta.schemas.chat import ChatRouteDecision, RetrievedChunk
from filingdelta.schemas.filing import EvidenceKind
from filingdelta.services.chat_qa import (
    _prioritize_retrieved_chunks,
    _retrieve_document_evidence,
    _select_document_retrieval_strategy,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_smoke_v2_page_text_hybrid_grid import (  # noqa: E402
    weighted_reciprocal_rank_fusion,
)


VARIANTS = (
    "current_live_strategy",
    "page_text_hybrid_fallback",
    "page_text_hybrid_override",
)
DOCUMENT_RETRIEVAL_ROUTES = {"document_only", "mixed"}
DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/golden_queries_v2_hybrid_strategy_alignment.json")
DEFAULT_MARKDOWN = Path("data/outputs/eval/golden_queries_v2_hybrid_strategy_alignment.md")
DEFAULT_QDRANT_ROOT = Path("tmp/smoke-v2-hybrid-strategy-alignment-qdrant")
DEFAULT_CANDIDATE_TOP_K = 50
DEFAULT_FINAL_TOP_K = 6
DEFAULT_WEIGHTED_SEMANTIC_TOP_N = 5
DEFAULT_WEIGHTED_BM25_TOP_N = 5
DEFAULT_WEIGHTED_RRF_K = 20
DEFAULT_WEIGHTED_ALPHA_SEMANTIC = 0.4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an offline smoke_v2 alignment diagnosis for typed retrieval strategy "
            "vs page_text hybrid fallback/override."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
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
    included_cases, skipped_cases = split_document_only_cases(cases)
    if args.list_cases:
        _print_cases(included_cases, skipped_cases)
        return None
    if args.final_top_k < 1:
        raise SystemExit("--final-top-k must be positive.")
    if args.candidate_top_k < args.final_top_k:
        raise SystemExit("--candidate-top-k must be >= --final-top-k.")

    report = asyncio.run(
        run_hybrid_strategy_alignment(
            manifest=manifest,
            cases=included_cases,
            skipped_cases=skipped_cases,
            candidate_top_k=args.candidate_top_k,
            final_top_k=args.final_top_k,
            qdrant_path=_resolve_alignment_qdrant_path(args.qdrant_path),
        )
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = _resolve_path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_alignment_markdown(report), encoding="utf-8")

    print("report:", output_path, flush=True)
    print("markdown:", markdown_path, flush=True)
    summary = report["summary"]
    print(
        f"included_document_only={summary['included_case_count']} "
        f"skipped={summary['skipped_case_count']}",
        flush=True,
    )
    for variant in VARIANTS:
        payload = summary["variants"][variant]
        print(
            f"{variant}: page_hit@{args.final_top_k}="
            f"{payload['page_hit_count']}/{payload['total_cases']} "
            f"primary_evidence_kind_hit={payload['primary_evidence_kind_hit_count']} "
            f"route_hit={payload['route_hit_count']} "
            f"intent_hit={payload['intent_hit_count']}",
            flush=True,
        )
    print(
        "fallback_rescues:",
        ", ".join(summary["page_text_fallback_rescue_query_ids"]) or "-",
        flush=True,
    )
    print(
        "override_only_rescues:",
        ", ".join(summary["override_only_rescue_query_ids"]) or "-",
        flush=True,
    )
    return report


async def run_hybrid_strategy_alignment(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    skipped_cases: list[dict[str, Any]],
    candidate_top_k: int,
    final_top_k: int,
    qdrant_path: Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    qdrant_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(FILINGDELTA_QDRANT_PATH=_settings_path_value(qdrant_path))
    client = QdrantClient(path=str(qdrant_path))
    pipeline = FilingIngestionPipeline(settings=settings)
    indexer = DocumentChunkIndexer(settings=settings, client=client)
    retriever = DocumentChunkRetriever(settings=settings, client=client)
    router = ChatRouterAgent(settings=settings)

    try:
        print(
            f"[1/3] preparing documents for {len(cases)} document_only cases "
            f"({len({case.document_key for case in cases})} documents)",
            flush=True,
        )
        document_contexts = _prepare_documents(
            manifest=manifest,
            cases=cases,
            pipeline=pipeline,
            indexer=indexer,
        )
        print("[2/3] running router + strategy variants", flush=True)
        case_results = []
        for index, case in enumerate(cases, start=1):
            print(f"[2/3] running {index}/{len(cases)} {case.case_id}", flush=True)
            case_results.append(
                await _run_case_alignment(
                    case=case,
                    context=document_contexts.get(case.document_key),
                    router=router,
                    retriever=retriever,
                    candidate_top_k=candidate_top_k,
                    final_top_k=final_top_k,
                )
            )
    finally:
        client.close()

    print("[3/3] writing reports", flush=True)
    return {
        "version": "golden_queries_v2_hybrid_strategy_alignment.v1",
        "scope": {
            "included_expected_route": "document_only",
            "skipped_non_document_only_manifest_cases": True,
            "answer_synthesis_run": False,
            "production_retrieval_modified": False,
            "variants": list(VARIANTS),
        },
        "manifest_path": str(manifest.source_path),
        "manifest_version": manifest.version,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "page_text_hybrid_config": {
            "semantic_top_n": DEFAULT_WEIGHTED_SEMANTIC_TOP_N,
            "bm25_top_n": DEFAULT_WEIGHTED_BM25_TOP_N,
            "rrf_k": DEFAULT_WEIGHTED_RRF_K,
            "alpha_semantic": DEFAULT_WEIGHTED_ALPHA_SEMANTIC,
        },
        "qdrant_path": str(qdrant_path),
        "documents": {
            key: context.get("stats", {}) for key, context in document_contexts.items()
        },
        "skipped_cases": skipped_cases,
        "cases": case_results,
        "summary": build_alignment_summary(
            case_results=case_results,
            skipped_cases=skipped_cases,
            final_top_k=final_top_k,
        ),
        "total_wall_ms": _elapsed_ms(started_at),
    }


def split_document_only_cases(
    cases: list[SmokeV2Case],
) -> tuple[list[SmokeV2Case], list[dict[str, Any]]]:
    included: list[SmokeV2Case] = []
    skipped: list[dict[str, Any]] = []
    for case in cases:
        if case.expected_route == "document_only":
            included.append(case)
            continue
        skipped.append(
            {
                "id": case.case_id,
                "query_id": case.case_id.rsplit("::", 1)[-1],
                "document_key": case.document_key,
                "company": case.company,
                "query": case.query,
                "expected_route": case.expected_route,
                "skip_reason": "manifest expected_route is not document_only",
            }
        )
    return included, skipped


async def _run_case_alignment(
    *,
    case: SmokeV2Case,
    context: dict[str, Any] | None,
    router: ChatRouterAgent,
    retriever: DocumentChunkRetriever,
    candidate_top_k: int,
    final_top_k: int,
) -> dict[str, Any]:
    if context is None or "error" in context:
        route_decision = None
        router_error = context.get("error") if context else "document context not prepared"
        variants = _error_variants(
            case=case,
            route_decision=route_decision,
            final_top_k=final_top_k,
            error=router_error,
        )
    else:
        route_started = time.perf_counter()
        try:
            route_decision = await router.route(
                question=case.query,
                document=context["document"],
            )
            route_ms = _elapsed_ms(route_started)
            router_error = None
        except Exception as error:  # noqa: BLE001 - diagnosis report should capture failures.
            route_decision = None
            route_ms = _elapsed_ms(route_started)
            router_error = f"{type(error).__name__}: {error}"

        variants = {}
        if route_decision is None:
            variants["current_live_strategy"] = build_variant_result_from_candidates(
                case=case,
                variant="current_live_strategy",
                route_decision=None,
                candidates=[],
                final_top_k=final_top_k,
                retrieval_ms=route_ms,
                retrieval_mode="router_error",
                error=router_error,
            )
            variants["page_text_hybrid_fallback"] = build_variant_result_from_candidates(
                case=case,
                variant="page_text_hybrid_fallback",
                route_decision=None,
                candidates=[],
                final_top_k=final_top_k,
                retrieval_ms=route_ms,
                retrieval_mode="router_error",
                error=router_error,
            )
        else:
            variants["current_live_strategy"] = _run_current_live_strategy(
                case=case,
                context=context,
                retriever=retriever,
                route_decision=route_decision,
                final_top_k=final_top_k,
            )
            variants["page_text_hybrid_fallback"] = _run_hybrid_fallback_strategy(
                case=case,
                context=context,
                retriever=retriever,
                route_decision=route_decision,
                candidate_top_k=candidate_top_k,
                final_top_k=final_top_k,
            )

        variants["page_text_hybrid_override"] = _run_page_text_override(
            case=case,
            context=context,
            retriever=retriever,
            route_decision=route_decision,
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
        )

    rescue = classify_case_rescue(variants)
    strategy = _strategy_payload(case=case, route_decision=route_decision)
    return {
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
        "supporting_pages": list(case.supporting_pages),
        "observed_route": route_decision.route if route_decision else None,
        "observed_document_evidence_intent": (
            route_decision.document_evidence_intent if route_decision else None
        ),
        "router_error": router_error,
        "strategy": strategy,
        "variants": variants,
        "rescue": rescue,
        "notes": case.notes,
    }


def _run_current_live_strategy(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    route_decision: ChatRouteDecision,
    final_top_k: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if route_decision.route not in DOCUMENT_RETRIEVAL_ROUTES:
        return build_variant_result_from_candidates(
            case=case,
            variant="current_live_strategy",
            route_decision=route_decision,
            candidates=[],
            final_top_k=final_top_k,
            retrieval_ms=_elapsed_ms(started_at),
            retrieval_mode="no_document_retrieval_for_route",
        )
    strategy = _select_document_retrieval_strategy(case.query, route_decision=route_decision)
    chunks, retrieval_mode = _retrieve_document_evidence(
        retriever=retriever,
        document_id=context["document_id"],
        question=case.query,
        callback_manager=None,
        strategy=strategy,
        page_text_chunks=context.get("page_text_chunks", []),
    )
    return build_variant_result_from_chunks(
        case=case,
        variant="current_live_strategy",
        route_decision=route_decision,
        chunks=chunks,
        final_top_k=final_top_k,
        retrieval_ms=_elapsed_ms(started_at),
        retrieval_mode=retrieval_mode,
        rank_source="current_semantic_strategy",
    )


def _run_hybrid_fallback_strategy(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    route_decision: ChatRouteDecision,
    candidate_top_k: int,
    final_top_k: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if route_decision.route not in DOCUMENT_RETRIEVAL_ROUTES:
        return build_variant_result_from_candidates(
            case=case,
            variant="page_text_hybrid_fallback",
            route_decision=route_decision,
            candidates=[],
            final_top_k=final_top_k,
            retrieval_ms=_elapsed_ms(started_at),
            retrieval_mode="no_document_retrieval_for_route",
        )
    strategy = _select_document_retrieval_strategy(case.query, route_decision=route_decision)
    candidates = _retrieve_strategy_candidates_with_hybrid_page_fallback(
        case=case,
        context=context,
        retriever=retriever,
        strategy=strategy,
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
    )
    return build_variant_result_from_candidates(
        case=case,
        variant="page_text_hybrid_fallback",
        route_decision=route_decision,
        candidates=candidates,
        final_top_k=final_top_k,
        retrieval_ms=_elapsed_ms(started_at),
        retrieval_mode="typed_strategy_with_page_text_hybrid_fallback",
    )


def _run_page_text_override(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    route_decision: ChatRouteDecision | None,
    candidate_top_k: int,
    final_top_k: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    candidates = _page_text_hybrid_candidates(
        case=case,
        context=context,
        retriever=retriever,
        candidate_top_k=candidate_top_k,
        top_k=final_top_k,
    )
    return build_variant_result_from_candidates(
        case=case,
        variant="page_text_hybrid_override",
        route_decision=route_decision,
        candidates=candidates,
        final_top_k=final_top_k,
        retrieval_ms=_elapsed_ms(started_at),
        retrieval_mode="page_text_hybrid_override",
    )


def _retrieve_strategy_candidates_with_hybrid_page_fallback(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    strategy: Any,
    candidate_top_k: int,
    final_top_k: int,
) -> list[RetrievalCandidate]:
    primary_chunks = retriever.retrieve(
        document_id=context["document_id"],
        question=case.query,
        top_k=strategy.primary_top_k,
        chunk_kind=strategy.primary_chunk_kind,
    )
    primary_candidates = _chunks_to_candidates(
        primary_chunks,
        rank_source=f"semantic:{strategy.primary_chunk_kind}",
    )
    fallback_chunk_kinds = strategy.fallback_chunk_kinds
    if not fallback_chunk_kinds and strategy.fallback_chunk_kind is not None:
        fallback_chunk_kinds = (strategy.fallback_chunk_kind,)

    fallback_candidates: list[RetrievalCandidate] = []
    for fallback_chunk_kind in fallback_chunk_kinds:
        if fallback_chunk_kind == EvidenceKind.PAGE_TEXT.value:
            fallback_candidates.extend(
                _page_text_hybrid_candidates(
                    case=case,
                    context=context,
                    retriever=retriever,
                    candidate_top_k=candidate_top_k,
                    top_k=strategy.fallback_top_k,
                )
            )
            continue
        fallback_chunks = retriever.retrieve(
            document_id=context["document_id"],
            question=case.query,
            top_k=strategy.fallback_top_k,
            chunk_kind=fallback_chunk_kind,
        )
        fallback_candidates.extend(
            _chunks_to_candidates(
                fallback_chunks,
                rank_source=f"semantic:{fallback_chunk_kind}",
            )
        )

    return _combine_strategy_candidates_like_chat(
        question=case.query,
        primary_candidates=primary_candidates,
        fallback_candidates=fallback_candidates,
        include_fallback_when_primary_found=strategy.include_fallback_when_primary_found,
        final_top_k=final_top_k,
    )


def _page_text_hybrid_candidates(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    candidate_top_k: int,
    top_k: int,
) -> list[RetrievalCandidate]:
    semantic = _semantic_candidates(
        retriever=retriever,
        document_id=context["document_id"],
        case=case,
        chunk_kind=EvidenceKind.PAGE_TEXT.value,
        top_k=min(candidate_top_k, DEFAULT_WEIGHTED_SEMANTIC_TOP_N),
    )
    bm25 = context["bm25_page_text"].search(
        case.query,
        top_k=min(candidate_top_k, DEFAULT_WEIGHTED_BM25_TOP_N),
    )
    return weighted_reciprocal_rank_fusion(
        semantic,
        bm25,
        alpha_semantic=DEFAULT_WEIGHTED_ALPHA_SEMANTIC,
        rrf_k=DEFAULT_WEIGHTED_RRF_K,
    )[:top_k]


def _semantic_candidates(
    *,
    retriever: DocumentChunkRetriever,
    document_id: str,
    case: SmokeV2Case,
    chunk_kind: str,
    top_k: int,
) -> list[RetrievalCandidate]:
    chunks = retriever.retrieve(
        document_id=document_id,
        question=case.query,
        top_k=top_k,
        chunk_kind=chunk_kind,
    )
    return rank_semantic_chunks(chunks, top_k=top_k)


def _combine_strategy_candidates_like_chat(
    *,
    question: str,
    primary_candidates: list[RetrievalCandidate],
    fallback_candidates: list[RetrievalCandidate],
    include_fallback_when_primary_found: bool,
    final_top_k: int,
) -> list[RetrievalCandidate]:
    if not fallback_candidates:
        candidates = dedupe_candidates(primary_candidates)
    elif primary_candidates and not include_fallback_when_primary_found:
        candidates = dedupe_candidates(primary_candidates)
    else:
        candidates = dedupe_candidates([*primary_candidates, *fallback_candidates])

    by_key = {dedupe_key(candidate.chunk): candidate for candidate in candidates}
    prioritized_chunks = _prioritize_retrieved_chunks(
        question=question,
        chunks=[candidate.chunk for candidate in candidates],
    )
    prioritized_candidates = [
        by_key[key]
        for chunk in prioritized_chunks
        if (key := dedupe_key(chunk)) in by_key
    ]
    return prioritized_candidates[:final_top_k]


def build_variant_result_from_chunks(
    *,
    case: SmokeV2Case,
    variant: str,
    route_decision: ChatRouteDecision | None,
    chunks: list[RetrievedChunk],
    final_top_k: int,
    retrieval_ms: int,
    retrieval_mode: str,
    rank_source: str,
    error: str | None = None,
) -> dict[str, Any]:
    candidates = _chunks_to_candidates(chunks, rank_source=rank_source)
    return build_variant_result_from_candidates(
        case=case,
        variant=variant,
        route_decision=route_decision,
        candidates=candidates,
        final_top_k=final_top_k,
        retrieval_ms=retrieval_ms,
        retrieval_mode=retrieval_mode,
        error=error,
    )


def build_variant_result_from_candidates(
    *,
    case: SmokeV2Case,
    variant: str,
    route_decision: ChatRouteDecision | None,
    candidates: list[RetrievalCandidate],
    final_top_k: int,
    retrieval_ms: int,
    retrieval_mode: str,
    error: str | None = None,
) -> dict[str, Any]:
    final_candidates = candidates[:final_top_k]
    retrieved_pages = [
        candidate.chunk.page_number
        for candidate in final_candidates
        if candidate.chunk.page_number is not None
    ]
    retrieved_evidence_kinds = [
        candidate.chunk.chunk_kind
        for candidate in final_candidates
        if candidate.chunk.chunk_kind
    ]
    hit_pages = sorted(set(case.expected_pages).intersection(retrieved_pages))
    supporting_hit_pages = sorted(set(case.supporting_pages).intersection(retrieved_pages))
    page_hit = bool(hit_pages)
    supporting_hit = bool(supporting_hit_pages)
    return {
        "variant": variant,
        "retrieval_mode": retrieval_mode,
        "route_hit": (
            None if route_decision is None else route_decision.route == case.expected_route
        ),
        "intent_hit": (
            None
            if route_decision is None
            else route_decision.document_evidence_intent
            == case.expected_document_evidence_intent
        ),
        "primary_evidence_kind_hit": case.primary_evidence_kind in retrieved_evidence_kinds,
        "page_hit": page_hit,
        "page_match_status": _page_match_status(
            primary_hit=page_hit,
            supporting_hit=supporting_hit,
        ),
        "supporting_hit": supporting_hit,
        "expected_pages": list(case.expected_pages),
        "supporting_pages": list(case.supporting_pages),
        "retrieved_pages": retrieved_pages,
        "hit_pages": hit_pages,
        "supporting_hit_pages": supporting_hit_pages,
        "retrieved_evidence_kinds": retrieved_evidence_kinds,
        "retrieved_row_labels": [
            candidate.chunk.row_label for candidate in final_candidates if candidate.chunk.row_label
        ],
        "retrieved_section_types": [
            candidate.chunk.section_type
            for candidate in final_candidates
            if candidate.chunk.section_type
        ],
        "retrieval_ms": retrieval_ms,
        "error": error,
        "top_chunks": [
            candidate_to_json(candidate, rank=rank)
            for rank, candidate in enumerate(final_candidates, start=1)
        ],
    }


def classify_case_rescue(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current_hit = bool(variants["current_live_strategy"].get("page_hit"))
    fallback_hit = bool(variants["page_text_hybrid_fallback"].get("page_hit"))
    override_hit = bool(variants["page_text_hybrid_override"].get("page_hit"))
    fallback_rescue = (not current_hit) and fallback_hit
    override_only_rescue = (not current_hit) and (not fallback_hit) and override_hit
    return {
        "page_text_fallback_rescue": fallback_rescue,
        "override_only_rescue": override_only_rescue,
        "override_hit_but_typed_strategy_miss": override_only_rescue,
        "notes": _rescue_note(
            current_hit=current_hit,
            fallback_hit=fallback_hit,
            override_hit=override_hit,
        ),
    }


def build_alignment_summary(
    *,
    case_results: list[dict[str, Any]],
    skipped_cases: list[dict[str, Any]],
    final_top_k: int,
) -> dict[str, Any]:
    variant_summaries = {
        variant: _summarize_variant(case_results, variant=variant)
        for variant in VARIANTS
    }
    return {
        "final_top_k": final_top_k,
        "included_case_count": len(case_results),
        "skipped_case_count": len(skipped_cases),
        "skipped_query_ids": [case["query_id"] for case in skipped_cases],
        "variants": variant_summaries,
        "route_hit_count": _count_bool(
            result["variants"]["current_live_strategy"].get("route_hit")
            for result in case_results
        ),
        "intent_hit_count": _count_bool(
            result["variants"]["current_live_strategy"].get("intent_hit")
            for result in case_results
        ),
        "page_text_fallback_rescue_count": sum(
            1 for result in case_results if result["rescue"]["page_text_fallback_rescue"]
        ),
        "page_text_fallback_rescue_query_ids": [
            result["query_id"]
            for result in case_results
            if result["rescue"]["page_text_fallback_rescue"]
        ],
        "override_only_rescue_count": sum(
            1 for result in case_results if result["rescue"]["override_only_rescue"]
        ),
        "override_only_rescue_query_ids": [
            result["query_id"]
            for result in case_results
            if result["rescue"]["override_only_rescue"]
        ],
    }


def render_alignment_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Golden Queries v2 Hybrid Strategy Alignment",
        "",
        "## 摘要",
        "",
        f"- Manifest：`{report['manifest_path']}`",
        f"- Included document_only cases：`{summary['included_case_count']}`",
        f"- Skipped non-document_only cases：`{summary['skipped_case_count']}`",
        f"- final top_k：`{summary['final_top_k']}`",
        (
            "- page_text hybrid config："
            f"`semantic_top={report['page_text_hybrid_config']['semantic_top_n']}`, "
            f"`bm25_top={report['page_text_hybrid_config']['bm25_top_n']}`, "
            f"`rrf_k={report['page_text_hybrid_config']['rrf_k']}`, "
            f"`alpha={report['page_text_hybrid_config']['alpha_semantic']}`"
        ),
        "- 口径：离线诊断；不运行 answer synthesis；不修改 production retrieval。",
        "- `page_text_hybrid_fallback` 只替换 typed strategy 中的 page_text fallback。",
        "- `page_text_hybrid_override` 强制全 page_text hybrid，仅代表理论上限，不是上线策略。",
        "",
        "## Variant 对照",
        "",
        (
            "| Variant | route_hit | intent_hit | primary_evidence_kind_hit | "
            "page_hit | supporting_hit | partial_support_only |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        payload = summary["variants"][variant]
        lines.append(
            "| "
            f"`{variant}` | "
            f"{payload['route_hit_count']}/{payload['total_cases']} | "
            f"{payload['intent_hit_count']}/{payload['total_cases']} | "
            f"{payload['primary_evidence_kind_hit_count']}/{payload['total_cases']} | "
            f"{payload['page_hit_count']}/{payload['total_cases']} | "
            f"{payload['supporting_hit_count']} | "
            f"{_join_text(payload['partial_support_only_query_ids'])} |"
        )

    lines.extend(
        [
            "",
            "## Rescue 归类",
            "",
            (
                "- page_text fallback rescue："
                f"`{_join_text(summary['page_text_fallback_rescue_query_ids'])}`"
            ),
            (
                "- override-only rescue："
                f"`{_join_text(summary['override_only_rescue_query_ids'])}`"
            ),
            "",
        ]
    )
    if report["skipped_cases"]:
        lines.extend(
            [
                "## Skipped Cases",
                "",
                "| Case | expected_route | reason |",
                "|---|---|---|",
            ]
        )
        for skipped in report["skipped_cases"]:
            lines.append(
                f"| `{skipped['id']}` | `{skipped['expected_route']}` | "
                f"{_esc(skipped['skip_reason'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Case Detail",
            "",
            (
                "| Case | query | expected pages | observed route/intent | strategy | "
                "current pages | fallback pages | override pages | rescue |"
            ),
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for result in report["cases"]:
        variants = result["variants"]
        lines.append(
            "| "
            f"`{result['id']}` | "
            f"{_esc(result['query'])} | "
            f"{_join_pages(result['expected_pages'])} | "
            f"`{result.get('observed_route') or 'unknown'}`/"
            f"`{result.get('observed_document_evidence_intent') or 'unknown'}` | "
            f"{_strategy_cell(result.get('strategy') or {})} | "
            f"{_variant_cell(variants['current_live_strategy'])} | "
            f"{_variant_cell(variants['page_text_hybrid_fallback'])} | "
            f"{_variant_cell(variants['page_text_hybrid_override'])} | "
            f"{_rescue_cell(result['rescue'])} |"
        )
    return "\n".join(lines) + "\n"


def _prepare_documents(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    pipeline: FilingIngestionPipeline,
    indexer: DocumentChunkIndexer,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for index, document_key in enumerate(
        sorted({case.document_key for case in cases}),
        start=1,
    ):
        document = manifest.documents.require(document_key)
        if not document.exists:
            contexts[document_key] = {
                "error": "source document not found",
                "stats": {
                    "source_path": str(document.source_path),
                    "exists": False,
                },
            }
            continue

        print(f"[1/3] preparing documents: {index} {document_key}", flush=True)
        parse_started = time.perf_counter()
        ingestion = pipeline.run(document.source)
        parse_ms = _elapsed_ms(parse_started)
        parsed_document = ingestion.parsed_filing.document
        document_id = parsed_document.document_id

        index_started = time.perf_counter()
        indexer.index_document(
            document_id=document_id,
            chunks=ingestion.chunks,
            evidence_units=ingestion.evidence_units,
        )
        index_ms = _elapsed_ms(index_started)

        all_chunks = evidence_units_to_retrieved_chunks(
            document_id=document_id,
            evidence_units=ingestion.evidence_units,
        )
        page_text_chunks = [
            chunk for chunk in all_chunks if chunk.chunk_kind == EvidenceKind.PAGE_TEXT.value
        ]
        contexts[document_key] = {
            "document_id": document_id,
            "document": parsed_document,
            "page_text_chunks": page_text_chunks,
            "bm25_page_text": BM25Index(page_text_chunks),
            "stats": {
                "document_id": document_id,
                "source_path": str(document.source_path),
                "exists": True,
                "parse_ms": parse_ms,
                "index_ms": index_ms,
                "page_text_count": len(page_text_chunks),
                "section_text_count": sum(
                    1
                    for chunk in all_chunks
                    if chunk.chunk_kind == EvidenceKind.SECTION_TEXT.value
                ),
                "table_row_count": sum(
                    1
                    for chunk in all_chunks
                    if chunk.chunk_kind == EvidenceKind.TABLE_ROW.value
                ),
            },
        }
    return contexts


def _summarize_variant(
    case_results: list[dict[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    variant_results = [result["variants"][variant] for result in case_results]
    total = len(variant_results)
    return {
        "total_cases": total,
        "route_hit_count": _count_bool(result.get("route_hit") for result in variant_results),
        "intent_hit_count": _count_bool(result.get("intent_hit") for result in variant_results),
        "primary_evidence_kind_hit_count": _count_bool(
            result.get("primary_evidence_kind_hit") for result in variant_results
        ),
        "page_hit_count": _count_bool(result.get("page_hit") for result in variant_results),
        "supporting_hit_count": _count_bool(
            result.get("supporting_hit") for result in variant_results
        ),
        "partial_support_only_query_ids": [
            case_result["query_id"]
            for case_result in case_results
            if case_result["variants"][variant].get("page_match_status")
            == "partial_support_only"
        ],
        "miss_query_ids": [
            case_result["query_id"]
            for case_result in case_results
            if not case_result["variants"][variant].get("page_hit")
        ],
    }


def _strategy_payload(
    *,
    case: SmokeV2Case,
    route_decision: ChatRouteDecision | None,
) -> dict[str, Any] | None:
    if route_decision is None or route_decision.route not in DOCUMENT_RETRIEVAL_ROUTES:
        return None
    strategy = _select_document_retrieval_strategy(case.query, route_decision=route_decision)
    fallback_chunk_kinds = strategy.fallback_chunk_kinds
    if not fallback_chunk_kinds and strategy.fallback_chunk_kind is not None:
        fallback_chunk_kinds = (strategy.fallback_chunk_kind,)
    return {
        "primary_chunk_kind": strategy.primary_chunk_kind,
        "fallback_chunk_kinds": list(fallback_chunk_kinds),
        "include_fallback_when_primary_found": strategy.include_fallback_when_primary_found,
        "primary_top_k": strategy.primary_top_k,
        "fallback_top_k": strategy.fallback_top_k,
        "retrieval_mode": strategy.retrieval_mode,
    }


def _error_variants(
    *,
    case: SmokeV2Case,
    route_decision: ChatRouteDecision | None,
    final_top_k: int,
    error: str,
) -> dict[str, dict[str, Any]]:
    return {
        variant: build_variant_result_from_candidates(
            case=case,
            variant=variant,
            route_decision=route_decision,
            candidates=[],
            final_top_k=final_top_k,
            retrieval_ms=0,
            retrieval_mode="error",
            error=error,
        )
        for variant in VARIANTS
    }


def _chunks_to_candidates(
    chunks: list[RetrievedChunk],
    *,
    rank_source: str,
) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            chunk=chunk,
            score=chunk.score or 0.0,
            rank_sources=(
                RankSource(source=rank_source, rank=rank, score=chunk.score),
            ),
        )
        for rank, chunk in enumerate(chunks, start=1)
    ]


def _page_match_status(*, primary_hit: bool, supporting_hit: bool) -> str:
    if primary_hit:
        return "primary_hit"
    if supporting_hit:
        return "partial_support_only"
    return "miss"


def _rescue_note(*, current_hit: bool, fallback_hit: bool, override_hit: bool) -> str:
    if not current_hit and fallback_hit:
        return "page_text hybrid fallback changes the typed-strategy page-hit result."
    if not current_hit and not fallback_hit and override_hit:
        return "page_text hybrid can find a page only under diagnostic override."
    if current_hit:
        return "current typed strategy already hits a primary page."
    return "not rescued by page_text hybrid fallback or override."


def _count_bool(values: Any) -> int:
    return sum(1 for value in values if value is True)


def _variant_cell(variant: dict[str, Any]) -> str:
    pages = _join_pages(variant.get("retrieved_pages", []))
    return f"`{variant.get('page_match_status', 'miss')}` {pages}"


def _strategy_cell(strategy: dict[str, Any]) -> str:
    if not strategy:
        return "`-`"
    fallback = ", ".join(strategy.get("fallback_chunk_kinds", [])) or "-"
    return (
        f"`{strategy.get('primary_chunk_kind')}` -> "
        f"`{fallback}`"
    )


def _rescue_cell(rescue: dict[str, Any]) -> str:
    if rescue.get("page_text_fallback_rescue"):
        return "`fallback_rescue`"
    if rescue.get("override_only_rescue"):
        return "`override_only_rescue`"
    return "`-`"


def _join_pages(pages: list[int] | tuple[int, ...]) -> str:
    return ", ".join(str(page) for page in pages) or "-"


def _join_text(values: list[str]) -> str:
    return ", ".join(values) or "-"


def _esc(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _print_cases(
    included_cases: list[SmokeV2Case],
    skipped_cases: list[dict[str, Any]],
) -> None:
    for case in included_cases:
        print(f"{case.case_id}\t{case.company}\t{case.query}", flush=True)
    for case in skipped_cases:
        print(
            f"SKIPPED {case['id']}\t{case['company']}\t{case['skip_reason']}",
            flush=True,
        )


def _resolve_alignment_qdrant_path(path: Path | None) -> Path:
    if path is not None:
        return _resolve_path(path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return _resolve_path(DEFAULT_QDRANT_ROOT / timestamp)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _settings_path_value(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
