from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from filingdelta.agents.chat_router import ChatRouterAgent
from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.eval.retrieval_diagnosis import RetrievalCandidate, rank_semantic_chunks
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
from filingdelta.schemas.chat import ChatRouteDecision
from filingdelta.schemas.filing import EvidenceKind
from filingdelta.services.chat_qa import (
    LEGACY_TYPED_TABLE_ROW_PRIMARY,
    _retrieve_document_evidence,
    _select_document_retrieval_strategy,
)


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_smoke_v2_hybrid_strategy_alignment import (  # noqa: E402
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_WEIGHTED_ALPHA_SEMANTIC,
    DEFAULT_WEIGHTED_BM25_TOP_N,
    DEFAULT_WEIGHTED_RRF_K,
    DEFAULT_WEIGHTED_SEMANTIC_TOP_N,
    DOCUMENT_RETRIEVAL_ROUTES,
    build_variant_result_from_candidates,
    build_variant_result_from_chunks,
    split_document_only_cases,
    _combine_strategy_candidates_like_chat,
    _configure_stdio,
    _elapsed_ms,
    _esc,
    _join_pages,
    _page_text_hybrid_candidates,
    _prepare_documents,
    _resolve_path,
    _settings_path_value,
)


VARIANTS = (
    "current_typed_strategy",
    "no_table_row_strategy",
    "page_text_hybrid_override",
)
DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/golden_queries_v2_no_table_row_strategy.json")
DEFAULT_MARKDOWN = Path("data/outputs/eval/golden_queries_v2_no_table_row_strategy.md")
DEFAULT_QDRANT_ROOT = Path("tmp/smoke-v2-no-table-row-strategy-qdrant")


@dataclass(frozen=True)
class NoTableRowPlan:
    intent: str
    primary_chunk_kind: str | None
    page_text_hybrid_top_k: int
    include_page_text_when_primary_found: bool = True
    primary_top_k: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "primary_chunk_kind": self.primary_chunk_kind,
            "fallback_chunk_kinds": [EvidenceKind.PAGE_TEXT.value],
            "page_text_hybrid_top_k": self.page_text_hybrid_top_k,
            "include_page_text_when_primary_found": self.include_page_text_when_primary_found,
            "primary_top_k": self.primary_top_k,
            "uses_table_row": False,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an offline no-table-row typed retrieval strategy diagnosis."
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
        run_no_table_row_strategy(
            manifest=manifest,
            cases=included_cases,
            skipped_cases=skipped_cases,
            candidate_top_k=args.candidate_top_k,
            final_top_k=args.final_top_k,
            qdrant_path=_resolve_no_table_qdrant_path(args.qdrant_path),
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
    markdown_path.write_text(render_no_table_row_markdown(report), encoding="utf-8")

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
            f"regressions={payload['previous_passed_regression_count']} "
            f"rescues={payload['rescue_count']}",
            flush=True,
        )
    return report


async def run_no_table_row_strategy(
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
        print("[2/3] running router + no-table-row variants", flush=True)
        case_results = []
        for index, case in enumerate(cases, start=1):
            print(f"[2/3] running {index}/{len(cases)} {case.case_id}", flush=True)
            case_results.append(
                await _run_case_no_table_row(
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
        "version": "golden_queries_v2_no_table_row_strategy.v1",
        "scope": {
            "included_expected_route": "document_only",
            "skipped_non_document_only_manifest_cases": True,
            "answer_synthesis_run": False,
            "production_retrieval_modified": False,
            "table_row_schema_or_builder_removed": False,
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
        "table_row_proof_point_risks": table_row_proof_point_risks(),
        "qdrant_path": str(qdrant_path),
        "documents": {
            key: context.get("stats", {}) for key, context in document_contexts.items()
        },
        "skipped_cases": skipped_cases,
        "cases": case_results,
        "summary": build_no_table_row_summary(
            case_results=case_results,
            skipped_cases=skipped_cases,
            final_top_k=final_top_k,
        ),
        "total_wall_ms": _elapsed_ms(started_at),
    }


def no_table_row_plan_for_intent(intent: str, *, final_top_k: int) -> NoTableRowPlan:
    if intent == "metric_value":
        return NoTableRowPlan(
            intent=intent,
            primary_chunk_kind=None,
            page_text_hybrid_top_k=final_top_k,
            primary_top_k=0,
        )
    if intent in {"metric_attribution", "business_narrative"}:
        return NoTableRowPlan(
            intent=intent,
            primary_chunk_kind=EvidenceKind.SECTION_TEXT.value,
            page_text_hybrid_top_k=4,
            primary_top_k=4,
        )
    return NoTableRowPlan(
        intent=intent,
        primary_chunk_kind=None,
        page_text_hybrid_top_k=final_top_k,
        primary_top_k=0,
    )


def build_no_table_row_summary(
    *,
    case_results: list[dict[str, Any]],
    skipped_cases: list[dict[str, Any]],
    final_top_k: int,
) -> dict[str, Any]:
    return {
        "final_top_k": final_top_k,
        "included_case_count": len(case_results),
        "skipped_case_count": len(skipped_cases),
        "skipped_query_ids": [case["query_id"] for case in skipped_cases],
        "variants": {
            variant: _summarize_variant(case_results, variant=variant)
            for variant in VARIANTS
        },
    }


def render_no_table_row_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Golden Queries v2 No-Table-Row Strategy",
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
        "- no_table_row_strategy 不把 `table_row` 放进 primary retrieval，也不作为 metric_attribution fallback。",
        "- page_text_hybrid_override 只代表理论上限，不是上线策略。",
        "",
        "## Variant 对照",
        "",
        (
            "| Variant | route_hit | intent_hit | primary_evidence_kind_hit | page_hit | "
            "regressions | rescues | miss cases |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|",
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
            f"{payload['previous_passed_regression_count']} | "
            f"{payload['rescue_count']} | "
            f"{_join_text(payload['miss_query_ids'])} |"
        )

    lines.extend(
        [
            "",
            "## Table-row Proof Point Risks",
            "",
            "| Risk | why it matters |",
            "|---|---|",
        ]
    )
    for risk in report["table_row_proof_point_risks"]:
        lines.append(f"| `{risk['id']}` | {_esc(risk['risk_note'])} |")

    if report["skipped_cases"]:
        lines.extend(
            [
                "",
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

    lines.extend(
        [
            "",
            "## Case Detail",
            "",
            (
                "| Case | query | expected pages | observed route/intent | no-table plan | "
                "current pages | no-table pages | override pages | classification |"
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
            f"{_plan_cell(result.get('no_table_row_plan') or {})} | "
            f"{_variant_cell(variants['current_typed_strategy'])} | "
            f"{_variant_cell(variants['no_table_row_strategy'])} | "
            f"{_variant_cell(variants['page_text_hybrid_override'])} | "
            f"{_classification_cell(result['classification'])} |"
        )
    return "\n".join(lines) + "\n"


async def _run_case_no_table_row(
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
        variants = _error_variants(case=case, final_top_k=final_top_k, error=router_error)
        plan = None
    else:
        try:
            route_decision = await router.route(
                question=case.query,
                document=context["document"],
            )
            router_error = None
        except Exception as error:  # noqa: BLE001 - diagnosis should capture router failures.
            route_decision = None
            router_error = f"{type(error).__name__}: {error}"

        if route_decision is None:
            variants = _error_variants(case=case, final_top_k=final_top_k, error=router_error)
            plan = None
        else:
            current = _run_current_typed_strategy(
                case=case,
                context=context,
                retriever=retriever,
                route_decision=route_decision,
                final_top_k=final_top_k,
            )
            no_table, plan = _run_no_table_row_variant(
                case=case,
                context=context,
                retriever=retriever,
                route_decision=route_decision,
                candidate_top_k=candidate_top_k,
                final_top_k=final_top_k,
            )
            override = _run_page_text_override(
                case=case,
                context=context,
                retriever=retriever,
                route_decision=route_decision,
                candidate_top_k=candidate_top_k,
                final_top_k=final_top_k,
            )
            variants = {
                "current_typed_strategy": current,
                "no_table_row_strategy": no_table,
                "page_text_hybrid_override": override,
            }

    classification = classify_no_table_case(variants)
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
        "no_table_row_plan": plan.to_json() if plan else None,
        "variants": variants,
        "classification": classification,
        "notes": case.notes,
    }


def _run_current_typed_strategy(
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
            variant="current_typed_strategy",
            route_decision=route_decision,
            candidates=[],
            final_top_k=final_top_k,
            retrieval_ms=_elapsed_ms(started_at),
            retrieval_mode="no_document_retrieval_for_route",
        )
    strategy = _select_document_retrieval_strategy(
        case.query,
        route_decision=route_decision,
        chat_retrieval_strategy=LEGACY_TYPED_TABLE_ROW_PRIMARY,
    )
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
        variant="current_typed_strategy",
        route_decision=route_decision,
        chunks=chunks,
        final_top_k=final_top_k,
        retrieval_ms=_elapsed_ms(started_at),
        retrieval_mode=retrieval_mode,
        rank_source="current_typed_semantic_strategy",
    )


def _run_no_table_row_variant(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    route_decision: ChatRouteDecision,
    candidate_top_k: int,
    final_top_k: int,
) -> tuple[dict[str, Any], NoTableRowPlan]:
    started_at = time.perf_counter()
    intent = route_decision.document_evidence_intent
    plan = no_table_row_plan_for_intent(intent, final_top_k=final_top_k)
    if route_decision.route not in DOCUMENT_RETRIEVAL_ROUTES:
        return (
            build_variant_result_from_candidates(
                case=case,
                variant="no_table_row_strategy",
                route_decision=route_decision,
                candidates=[],
                final_top_k=final_top_k,
                retrieval_ms=_elapsed_ms(started_at),
                retrieval_mode="no_document_retrieval_for_route",
            ),
            plan,
        )

    candidates = _no_table_row_candidates(
        case=case,
        context=context,
        retriever=retriever,
        plan=plan,
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
    )
    return (
        build_variant_result_from_candidates(
            case=case,
            variant="no_table_row_strategy",
            route_decision=route_decision,
            candidates=candidates,
            final_top_k=final_top_k,
            retrieval_ms=_elapsed_ms(started_at),
            retrieval_mode="no_table_row_offline_strategy",
        ),
        plan,
    )


def _run_page_text_override(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    route_decision: ChatRouteDecision,
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


def _no_table_row_candidates(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    plan: NoTableRowPlan,
    candidate_top_k: int,
    final_top_k: int,
) -> list[RetrievalCandidate]:
    primary_candidates: list[RetrievalCandidate] = []
    if plan.primary_chunk_kind:
        chunks = retriever.retrieve(
            document_id=context["document_id"],
            question=case.query,
            top_k=plan.primary_top_k,
            chunk_kind=plan.primary_chunk_kind,
        )
        primary_candidates = rank_semantic_chunks(chunks, top_k=plan.primary_top_k)

    page_text_candidates = _page_text_hybrid_candidates(
        case=case,
        context=context,
        retriever=retriever,
        candidate_top_k=candidate_top_k,
        top_k=plan.page_text_hybrid_top_k,
    )
    return _combine_strategy_candidates_like_chat(
        question=case.query,
        primary_candidates=primary_candidates,
        fallback_candidates=page_text_candidates,
        include_fallback_when_primary_found=plan.include_page_text_when_primary_found,
        final_top_k=final_top_k,
    )


def classify_no_table_case(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current_hit = bool(variants["current_typed_strategy"].get("page_hit"))
    no_table_hit = bool(variants["no_table_row_strategy"].get("page_hit"))
    override_hit = bool(variants["page_text_hybrid_override"].get("page_hit"))
    return {
        "previous_passed_regression": current_hit and not no_table_hit,
        "rescue": (not current_hit) and no_table_hit,
        "override_only_rescue": (not current_hit) and (not no_table_hit) and override_hit,
        "unchanged_hit": current_hit and no_table_hit,
        "unchanged_miss": (not current_hit) and (not no_table_hit) and (not override_hit),
    }


def table_row_proof_point_risks() -> list[dict[str, str]]:
    return [
        {
            "id": "customer_deposits",
            "risk_note": (
                "客户存款、活期/定期存款等行级指标曾依赖 table_row 做精确 row_label 绑定；"
                "降级前需要单独验证客户存款类 chat quality smoke。"
            ),
        },
        {
            "id": "roe_roae",
            "risk_note": (
                "ROE/ROAE、归母净利润等 headline metric 可能仍需要行级结构化证据来稳定单位、"
                "期间和口径。"
            ),
        },
        {
            "id": "capex_and_numeric_rows",
            "risk_note": (
                "资本开支、研发投入、经营 KPI 等数值行可能被 page_text 找到页，但答案抽取仍可能"
                "失去 table_row 的字段级约束。"
            ),
        },
    ]


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
        "previous_passed_regression_count": sum(
            1 for result in case_results if result["classification"]["previous_passed_regression"]
        )
        if variant == "no_table_row_strategy"
        else 0,
        "previous_passed_regression_query_ids": [
            result["query_id"]
            for result in case_results
            if result["classification"]["previous_passed_regression"]
        ]
        if variant == "no_table_row_strategy"
        else [],
        "rescue_count": sum(
            1 for result in case_results if result["classification"]["rescue"]
        )
        if variant == "no_table_row_strategy"
        else 0,
        "rescue_query_ids": [
            result["query_id"] for result in case_results if result["classification"]["rescue"]
        ]
        if variant == "no_table_row_strategy"
        else [],
        "override_only_rescue_count": sum(
            1 for result in case_results if result["classification"]["override_only_rescue"]
        )
        if variant == "page_text_hybrid_override"
        else 0,
        "override_only_rescue_query_ids": [
            result["query_id"]
            for result in case_results
            if result["classification"]["override_only_rescue"]
        ]
        if variant == "page_text_hybrid_override"
        else [],
        "miss_query_ids": [
            case_result["query_id"]
            for case_result in case_results
            if not case_result["variants"][variant].get("page_hit")
        ],
    }


def _error_variants(
    *,
    case: SmokeV2Case,
    final_top_k: int,
    error: str | None,
) -> dict[str, dict[str, Any]]:
    return {
        variant: build_variant_result_from_candidates(
            case=case,
            variant=variant,
            route_decision=None,
            candidates=[],
            final_top_k=final_top_k,
            retrieval_ms=0,
            retrieval_mode="error",
            error=error,
        )
        for variant in VARIANTS
    }


def _count_bool(values: Any) -> int:
    return sum(1 for value in values if value is True)


def _variant_cell(variant: dict[str, Any]) -> str:
    return f"`{variant.get('page_match_status', 'miss')}` {_join_pages(variant.get('retrieved_pages', []))}"


def _plan_cell(plan: dict[str, Any]) -> str:
    if not plan:
        return "`-`"
    primary = plan.get("primary_chunk_kind") or "none"
    return f"`{primary}` + `page_text_hybrid`"


def _classification_cell(classification: dict[str, Any]) -> str:
    if classification.get("previous_passed_regression"):
        return "`regression`"
    if classification.get("rescue"):
        return "`rescue`"
    if classification.get("override_only_rescue"):
        return "`override_only`"
    if classification.get("unchanged_hit"):
        return "`unchanged_hit`"
    return "`unchanged_miss`"


def _join_text(values: list[str]) -> str:
    return ", ".join(values) or "-"


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


def _resolve_no_table_qdrant_path(path: Path | None) -> Path:
    if path is not None:
        return _resolve_path(path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return _resolve_path(DEFAULT_QDRANT_ROOT / timestamp)


if __name__ == "__main__":
    main()
