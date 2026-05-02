from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from filingdelta.core.config import REPO_ROOT, Settings
from filingdelta.eval.retrieval_diagnosis import RankSource, RetrievalCandidate, dedupe_key
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
from filingdelta.schemas.filing import EvidenceKind


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_smoke_v2_page_text_hybrid_grid import (  # noqa: E402
    GridConfig,
    _assert_page_text_candidates,
    _elapsed_ms,
    _prepare_documents,
    _retrieve_max_candidates,
    _settings_path_value,
    build_grid_case_result,
    load_baseline_report_context,
    weighted_reciprocal_rank_fusion,
)


DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_BASELINE_REPORT = Path("data/outputs/eval/golden_queries_v2_live_pilot_20_report.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/golden_queries_v2_page_text_hybrid_guard.json")
DEFAULT_MARKDOWN = Path("data/outputs/eval/golden_queries_v2_page_text_hybrid_guard.md")
DEFAULT_QDRANT_ROOT = Path("tmp/smoke-v2-page-text-hybrid-guard-qdrant")
DEFAULT_FINAL_TOP_K = 6
DEFAULT_MAX_SEMANTIC_TOP_N = 15
DEFAULT_MAX_BM25_TOP_N = 15
BEST_WEIGHTED_CONFIG = GridConfig(
    semantic_top_n=5,
    bm25_top_n=5,
    rrf_k=20,
    alpha_semantic=0.4,
)
WATCHLIST_QUERY_IDS = ("INS-01", "U-02")
U02_QUERY_ID = "U-02"


@dataclass(frozen=True)
class GuardVariant:
    id: str
    family: str
    description: str
    semantic_top_n: int = BEST_WEIGHTED_CONFIG.semantic_top_n
    bm25_top_n: int = BEST_WEIGHTED_CONFIG.bm25_top_n
    rrf_k: int = BEST_WEIGHTED_CONFIG.rrf_k
    alpha_semantic: float = BEST_WEIGHTED_CONFIG.alpha_semantic
    semantic_floor: int = 0
    bm25_only_cap: int | None = None
    overlap_boost: float = 0.0
    intent_alpha: bool = False

    def alpha_for_case(self, *, case: SmokeV2Case, baseline: dict[str, Any]) -> float:
        if not self.intent_alpha:
            return self.alpha_semantic
        intent = baseline.get("observed_intent") or case.expected_document_evidence_intent
        if intent == "business_narrative":
            return 0.75
        if intent == "metric_attribution":
            return 0.55
        if intent == "metric_value":
            return 0.25
        return self.alpha_semantic

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "description": self.description,
            "semantic_top_n": self.semantic_top_n,
            "bm25_top_n": self.bm25_top_n,
            "rrf_k": self.rrf_k,
            "alpha_semantic": self.alpha_semantic,
            "semantic_floor": self.semantic_floor,
            "bm25_only_cap": self.bm25_only_cap,
            "overlap_boost": self.overlap_boost,
            "intent_alpha": self.intent_alpha,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run page_text-only hybrid guard variants for smoke_v2 cases."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--qdrant-path", type=Path, default=None)
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
    if args.final_top_k < 1:
        raise SystemExit("--final-top-k must be positive.")

    baseline_by_case_id, baseline_report = load_baseline_report_context(
        _resolve_path(args.baseline_report)
    )
    report = run_page_text_hybrid_guard(
        manifest=manifest,
        cases=cases,
        baseline_by_case_id=baseline_by_case_id,
        baseline_report=baseline_report,
        final_top_k=args.final_top_k,
        qdrant_path=_resolve_guard_qdrant_path(args.qdrant_path),
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = _resolve_path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_page_text_hybrid_guard_markdown(report), encoding="utf-8")

    best = report["summary"]["best_guard_variant"]
    print("report:", output_path, flush=True)
    print("markdown:", markdown_path, flush=True)
    print(
        "best_guard_variant:",
        best["variant"]["id"],
        f"primary_page_hit@{args.final_top_k}={best['page_hit_count']}/{best['total_cases']}",
        f"supporting_hit={best['supporting_page_hit_count']}",
        f"regressions={best['baseline_passed_regression_count']}",
        f"rescues={best['baseline_page_miss_rescue_count']}",
        flush=True,
    )
    return report


def run_page_text_hybrid_guard(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    baseline_by_case_id: dict[str, dict[str, Any]],
    baseline_report: dict[str, Any],
    final_top_k: int,
    qdrant_path: Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    variants = build_guard_variants()
    print(
        f"[1/4] preparing documents for {len(cases)} cases "
        f"({len({case.document_key for case in cases})} documents)",
        flush=True,
    )

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
        print(
            f"[2/4] retrieving max page_text candidates: semantic_top={DEFAULT_MAX_SEMANTIC_TOP_N}, "
            f"bm25_top={DEFAULT_MAX_BM25_TOP_N}",
            flush=True,
        )
        max_candidates = _retrieve_max_candidates(
            cases=cases,
            document_contexts=document_contexts,
            retriever=retriever,
            baseline_by_case_id=baseline_by_case_id,
            semantic_top_n=DEFAULT_MAX_SEMANTIC_TOP_N,
            bm25_top_n=DEFAULT_MAX_BM25_TOP_N,
        )
    finally:
        client.close()

    print(f"[3/4] evaluating guard variants: {len(variants)} variants", flush=True)
    evaluations = [
        evaluate_guard_variant(
            variant=variant,
            max_candidates=max_candidates,
            final_top_k=final_top_k,
        )
        for variant in variants
    ]
    print("[4/4] writing reports", flush=True)
    summary = build_guard_summary(evaluations=evaluations, final_top_k=final_top_k)
    best_variant_id = summary["best_guard_variant"]["variant"]["id"]
    best_cases = next(
        evaluation["cases"]
        for evaluation in evaluations
        if evaluation["variant"]["id"] == best_variant_id
    )
    return {
        "version": "golden_queries_v2_page_text_hybrid_guard.v1",
        "manifest_path": str(manifest.source_path),
        "manifest_version": manifest.version,
        "baseline_report": baseline_report,
        "final_top_k": final_top_k,
        "max_semantic_top_n": DEFAULT_MAX_SEMANTIC_TOP_N,
        "max_bm25_top_n": DEFAULT_MAX_BM25_TOP_N,
        "shadow_scope": {
            "chunk_kind": EvidenceKind.PAGE_TEXT.value,
            "answer_synthesis_run": False,
            "formal_chat_retrieval_modified": False,
            "guard_uses_cached_max_candidates": True,
            "guard_repeats_embedding_calls": False,
        },
        "qdrant_path": str(qdrant_path),
        "total_wall_ms": _elapsed_ms(started_at),
        "summary": summary,
        "documents": {
            key: context["stats"]
            for key, context in sorted(document_contexts.items())
        },
        "best_cases": best_cases,
        "u02_gold_refresh_check": build_u02_gold_refresh_check(
            max_candidates=max_candidates,
            evaluations=evaluations,
            best_variant_id=best_variant_id,
            final_top_k=final_top_k,
        ),
        "variant_results": [
            {key: value for key, value in evaluation.items() if key != "cases"}
            for evaluation in evaluations
        ],
    }


def build_guard_variants() -> list[GuardVariant]:
    return [
        GuardVariant(
            id="weighted_rrf_best",
            family="weighted_rrf_best",
            description="Current best coarse-grid weighted RRF config.",
        ),
        GuardVariant(
            id="semantic_floor_top1",
            family="semantic_floor",
            description="Weighted RRF with semantic top1 page retained in top6.",
            semantic_floor=1,
        ),
        GuardVariant(
            id="semantic_floor_top2",
            family="semantic_floor",
            description="Weighted RRF with semantic top2 unique pages retained in top6.",
            semantic_floor=2,
        ),
        GuardVariant(
            id="bm25_only_cap_2",
            family="bm25_only_cap",
            description="Weighted RRF with at most 2 BM25-only pages in top6.",
            bm25_only_cap=2,
        ),
        GuardVariant(
            id="bm25_only_cap_3",
            family="bm25_only_cap",
            description="Weighted RRF with at most 3 BM25-only pages in top6.",
            bm25_only_cap=3,
        ),
        GuardVariant(
            id="overlap_boost",
            family="overlap_boost",
            description="Weighted RRF with a small boost for pages appearing in both semantic and BM25 candidates.",
            overlap_boost=0.10,
        ),
        GuardVariant(
            id="intent_alpha",
            family="intent_alpha",
            description="Intent-aware alpha: metric_value BM25-leaning, business_narrative semantic-leaning.",
            intent_alpha=True,
        ),
    ]


def evaluate_guard_variant(
    *,
    variant: GuardVariant,
    max_candidates: list[dict[str, Any]],
    final_top_k: int,
) -> dict[str, Any]:
    cases = []
    for payload in max_candidates:
        case = payload["case"]
        baseline = payload["baseline"]
        semantic = payload["semantic_candidates"][: variant.semantic_top_n]
        bm25 = payload["bm25_candidates"][: variant.bm25_top_n]
        candidates = build_guard_candidates(
            variant=variant,
            case=case,
            baseline=baseline,
            semantic_candidates=semantic,
            bm25_candidates=bm25,
            final_top_k=final_top_k,
        )
        result = build_grid_case_result(
            case=case,
            baseline=baseline,
            candidates=candidates,
            final_top_k=final_top_k,
            config_id=variant.id,
        )
        cases.append(result)
    return _build_variant_evaluation(variant=variant, cases=cases)


def build_guard_candidates(
    *,
    variant: GuardVariant,
    case: SmokeV2Case,
    baseline: dict[str, Any],
    semantic_candidates: list[RetrievalCandidate],
    bm25_candidates: list[RetrievalCandidate],
    final_top_k: int,
) -> list[RetrievalCandidate]:
    _assert_page_text_candidates([*semantic_candidates, *bm25_candidates])
    candidates = weighted_reciprocal_rank_fusion(
        semantic_candidates,
        bm25_candidates,
        alpha_semantic=variant.alpha_for_case(case=case, baseline=baseline),
        rrf_k=variant.rrf_k,
    )
    if variant.overlap_boost > 0:
        candidates = apply_overlap_boost(
            candidates=candidates,
            semantic_candidates=semantic_candidates,
            bm25_candidates=bm25_candidates,
            boost_weight=variant.overlap_boost,
            rrf_k=variant.rrf_k,
        )
    if variant.semantic_floor:
        candidates = apply_semantic_floor(
            ranked_candidates=candidates,
            semantic_candidates=semantic_candidates,
            floor_count=variant.semantic_floor,
            final_top_k=final_top_k,
        )
    if variant.bm25_only_cap is not None:
        candidates = apply_bm25_only_cap(
            ranked_candidates=candidates,
            semantic_candidates=semantic_candidates,
            cap=variant.bm25_only_cap,
            final_top_k=final_top_k,
        )
    _assert_page_text_candidates(candidates)
    return candidates


def apply_semantic_floor(
    *,
    ranked_candidates: list[RetrievalCandidate],
    semantic_candidates: list[RetrievalCandidate],
    floor_count: int,
    final_top_k: int,
) -> list[RetrievalCandidate]:
    _assert_page_text_candidates([*ranked_candidates, *semantic_candidates])
    floor_candidates = _first_unique_page_candidates(semantic_candidates, limit=floor_count)
    floor_pages = _candidate_page_set(floor_candidates)
    merged = [
        *floor_candidates,
        *[
            candidate
            for candidate in ranked_candidates
            if candidate.chunk.page_number not in floor_pages
        ],
    ]
    return _dedupe_candidates(merged)[:final_top_k]


def apply_bm25_only_cap(
    *,
    ranked_candidates: list[RetrievalCandidate],
    semantic_candidates: list[RetrievalCandidate],
    cap: int,
    final_top_k: int,
) -> list[RetrievalCandidate]:
    _assert_page_text_candidates([*ranked_candidates, *semantic_candidates])
    semantic_pages = _candidate_page_set(semantic_candidates)
    selected: list[RetrievalCandidate] = []
    bm25_only_count = 0
    seen: set[tuple[object, ...]] = set()
    for candidate in ranked_candidates:
        key = dedupe_key(candidate.chunk)
        if key in seen:
            continue
        is_bm25_only_page = candidate.chunk.page_number not in semantic_pages
        if is_bm25_only_page and bm25_only_count >= cap:
            continue
        selected.append(candidate)
        seen.add(key)
        if is_bm25_only_page:
            bm25_only_count += 1
        if len(selected) >= final_top_k:
            break
    return selected


def apply_overlap_boost(
    *,
    candidates: list[RetrievalCandidate],
    semantic_candidates: list[RetrievalCandidate],
    bm25_candidates: list[RetrievalCandidate],
    boost_weight: float,
    rrf_k: int,
) -> list[RetrievalCandidate]:
    _assert_page_text_candidates([*candidates, *semantic_candidates, *bm25_candidates])
    overlap_pages = _candidate_page_set(semantic_candidates).intersection(
        _candidate_page_set(bm25_candidates)
    )
    if not overlap_pages:
        return candidates

    boost = boost_weight / (rrf_k + 1)
    boosted: list[tuple[int, RetrievalCandidate]] = []
    for index, candidate in enumerate(candidates):
        if candidate.chunk.page_number in overlap_pages:
            candidate = RetrievalCandidate(
                chunk=candidate.chunk,
                score=candidate.score + boost,
                rank_sources=(
                    *candidate.rank_sources,
                    RankSource(source="overlap_boost", rank=1, score=boost),
                ),
            )
        boosted.append((index, candidate))
    boosted.sort(key=lambda item: (-item[1].score, item[0]))
    return [candidate for _, candidate in boosted]


def build_guard_summary(
    *,
    evaluations: list[dict[str, Any]],
    final_top_k: int,
) -> dict[str, Any]:
    ranked_guard = sorted(evaluations, key=_guard_sort_key)
    ranked_page_hit = sorted(evaluations, key=_page_hit_sort_key)
    return {
        "final_top_k": final_top_k,
        "variant_count": len(evaluations),
        "best_guard_variant": _variant_summary(ranked_guard[0]),
        "best_page_hit_variant": _variant_summary(ranked_page_hit[0]),
        "variant_summaries": [_variant_summary(evaluation) for evaluation in ranked_page_hit],
    }


def build_u02_gold_refresh_check(
    *,
    max_candidates: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    best_variant_id: str,
    final_top_k: int,
) -> dict[str, Any]:
    payload = next(
        (row for row in max_candidates if row["case"].case_id.rsplit("::", 1)[-1] == U02_QUERY_ID),
        None,
    )
    if payload is None:
        return {"query_id": U02_QUERY_ID, "status": "not_in_selected_cases"}

    case = payload["case"]
    expected_pages = list(case.expected_pages)
    semantic_top_pages = _candidate_pages(payload["semantic_candidates"])
    bm25_top_pages = _candidate_pages(payload["bm25_candidates"])
    variant_pages: dict[str, list[int]] = {}
    variant_hits: dict[str, bool] = {}
    for evaluation in evaluations:
        case_result = next(
            row for row in evaluation["cases"] if row["query_id"] == U02_QUERY_ID
        )
        variant_id = evaluation["variant"]["id"]
        variant_pages[variant_id] = list(case_result["retrieved_pages"])
        variant_hits[variant_id] = bool(case_result["hit"])

    weighted_pages = variant_pages.get("weighted_rrf_best", [])
    best_pages = variant_pages.get(best_variant_id, [])
    return {
        "query_id": U02_QUERY_ID,
        "case_id": case.case_id,
        "query": case.query,
        "expected_pages": expected_pages,
        "baseline": payload["baseline"],
        "semantic_top_pages": semantic_top_pages,
        "bm25_top_pages": bm25_top_pages,
        "variant_pages": variant_pages,
        "variant_hits": variant_hits,
        "best_variant_id": best_variant_id,
        "expected_page_in_semantic_top6": bool(
            set(expected_pages).intersection(semantic_top_pages[:final_top_k])
        ),
        "expected_page_in_bm25_top6": bool(
            set(expected_pages).intersection(bm25_top_pages[:final_top_k])
        ),
        "expected_page_in_weighted_rrf_top6": bool(
            set(expected_pages).intersection(weighted_pages[:final_top_k])
        ),
        "expected_page_in_best_guard_top6": bool(
            set(expected_pages).intersection(best_pages[:final_top_k])
        ),
        "check_note": _u02_gold_refresh_check_text(
            expected_pages=expected_pages,
            semantic_top_pages=semantic_top_pages,
            bm25_top_pages=bm25_top_pages,
            weighted_pages=weighted_pages,
            best_pages=best_pages,
            best_variant_id=best_variant_id,
        ),
    }


def render_page_text_hybrid_guard_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    best_guard = summary["best_guard_variant"]
    best_page_hit = summary["best_page_hit_variant"]
    u02 = report["u02_gold_refresh_check"]
    lines = [
        "# Golden Queries v2 PageText Hybrid Guard",
        "",
        "## 摘要",
        "",
        f"- Manifest：`{report['manifest_path']}`",
        f"- Baseline report：`{report['baseline_report'].get('path', '-')}`",
        f"- Case 总数：`{best_guard['total_cases']}`",
        f"- Variant 总数：`{summary['variant_count']}`",
        f"- final top_k：`{summary['final_top_k']}`",
        "- 口径：只使用 `page_text` candidates；不运行 answer synthesis；不修改正式 retrieval。",
        f"- Best guard：`{best_guard['variant']['id']}` "
        f"primary_page_hit@6={best_guard['page_hit_count']}/{best_guard['total_cases']} "
        f"supporting_hit={best_guard['supporting_page_hit_count']} "
        f"partial_support_only={','.join(best_guard['partial_support_only_query_ids']) or '-'} "
        f"regressions={best_guard['baseline_passed_regression_count']} "
        f"rescues={best_guard['baseline_page_miss_rescue_count']}",
        f"- Best page-hit：`{best_page_hit['variant']['id']}` "
        f"primary_page_hit@6={best_page_hit['page_hit_count']}/{best_page_hit['total_cases']} "
        f"regressions={best_page_hit['baseline_passed_regression_count']}",
        "",
        "## Variant 对照",
        "",
        "| Variant | family | primary_page_hit@6 | supporting_hit | partial_support_only | regressions | rescues | regressed | rescued | watchlist |",
        "|---|---|---:|---:|---|---:|---:|---|---|---|",
    ]
    for variant in summary["variant_summaries"]:
        lines.append(_variant_table_row(variant))

    lines.extend(
        [
            "",
            "## U-02 Gold Refresh Check",
            "",
        ]
    )
    if u02.get("status") == "not_in_selected_cases":
        lines.append("- U-02 not in selected cases.")
    else:
        lines.extend(
            [
                f"- Case：`{u02['case_id']}`",
                f"- Query：{_esc(u02['query'])}",
                f"- Expected pages：`{', '.join(str(page) for page in u02['expected_pages'])}`",
                f"- Baseline：`status={u02['baseline'].get('status')} "
                f"page_hit={_tri_state_label(u02['baseline'].get('page_hit_at_k'))}`",
                f"- Semantic top pages：`{', '.join(str(page) for page in u02['semantic_top_pages'][:10])}`",
                f"- BM25 top pages：`{', '.join(str(page) for page in u02['bm25_top_pages'][:10])}`",
                f"- Weighted RRF `weighted_rrf_best` top pages：`{', '.join(str(page) for page in u02['variant_pages'].get('weighted_rrf_best', []))}`",
                f"- Best guard `{u02['best_variant_id']}` top pages：`{', '.join(str(page) for page in u02['variant_pages'].get(u02['best_variant_id'], []))}`",
                f"- Gold page in weighted top6：`{u02['expected_page_in_weighted_rrf_top6']}`",
                f"- Gold page in best guard top6：`{u02['expected_page_in_best_guard_top6']}`",
                f"- Check：{_esc(u02['check_note'])}",
            ]
        )

    lines.extend(
        [
            "",
            "## Best Guard Case Detail",
            "",
            "| Case | query | expected_pages | supporting_pages | retrieved_pages | page_match_status | baseline |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for case in report["best_cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_esc(case['id'])}`",
                    _esc(case["query"]),
                    ", ".join(str(page) for page in case["expected_pages"]) or "-",
                    ", ".join(str(page) for page in case.get("supporting_pages", [])) or "-",
                    ", ".join(str(page) for page in case["retrieved_pages"]) or "-",
                    _hit_label(case.get("page_match_status")),
                    _baseline_cell(case["baseline"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _build_variant_evaluation(*, variant: GuardVariant, cases: list[dict[str, Any]]) -> dict[str, Any]:
    hit_count = sum(1 for case in cases if case["hit"])
    supporting_hit_count = sum(1 for case in cases if case.get("supporting_hit"))
    partial_support_only = [
        case["query_id"]
        for case in cases
        if case.get("page_match_status") == "partial_support_only"
    ]
    return {
        "variant": variant.to_json(),
        "page_hit_count": hit_count,
        "primary_page_hit_count": hit_count,
        "supporting_page_hit_count": supporting_hit_count,
        "partial_support_only_count": len(partial_support_only),
        "partial_support_only_query_ids": partial_support_only,
        "total_cases": len(cases),
        "page_hit_rate": _safe_ratio(hit_count, len(cases)),
        "baseline_passed_regression_count": sum(
            1
            for case in cases
            if case["baseline"].get("status") == "passed" and not case["hit"]
        ),
        "baseline_page_miss_rescue_count": sum(
            1
            for case in cases
            if case["baseline"].get("page_hit_at_k") is False and case["hit"]
        ),
        "regressed_query_ids": [
            case["query_id"]
            for case in cases
            if case["baseline"].get("status") == "passed" and not case["hit"]
        ],
        "rescued_query_ids": [
            case["query_id"]
            for case in cases
            if case["baseline"].get("page_hit_at_k") is False and case["hit"]
        ],
        "watchlist": _watchlist_status(cases),
        "cases": cases,
    }


def _variant_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": evaluation["variant"],
        "page_hit_count": evaluation["page_hit_count"],
        "primary_page_hit_count": evaluation["primary_page_hit_count"],
        "supporting_page_hit_count": evaluation["supporting_page_hit_count"],
        "partial_support_only_count": evaluation["partial_support_only_count"],
        "partial_support_only_query_ids": evaluation["partial_support_only_query_ids"],
        "total_cases": evaluation["total_cases"],
        "page_hit_rate": evaluation["page_hit_rate"],
        "baseline_passed_regression_count": evaluation["baseline_passed_regression_count"],
        "baseline_page_miss_rescue_count": evaluation["baseline_page_miss_rescue_count"],
        "regressed_query_ids": evaluation["regressed_query_ids"],
        "rescued_query_ids": evaluation["rescued_query_ids"],
        "watchlist": evaluation["watchlist"],
    }


def _guard_sort_key(evaluation: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        evaluation["baseline_passed_regression_count"],
        -evaluation["page_hit_count"],
        -evaluation["baseline_page_miss_rescue_count"],
        _variant_priority(evaluation["variant"]["id"]),
    )


def _page_hit_sort_key(evaluation: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        -evaluation["page_hit_count"],
        evaluation["baseline_passed_regression_count"],
        -evaluation["baseline_page_miss_rescue_count"],
        _variant_priority(evaluation["variant"]["id"]),
    )


def _variant_priority(variant_id: str) -> int:
    order = {variant.id: index for index, variant in enumerate(build_guard_variants())}
    return order.get(variant_id, 999)


def _watchlist_status(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_query_id = {case["query_id"]: case for case in cases}
    return {
        query_id: {
            "hit": bool(case["hit"]) if case else None,
            "baseline_status": (case or {}).get("baseline", {}).get("status"),
            "retrieved_pages": (case or {}).get("retrieved_pages", []),
        }
        for query_id in WATCHLIST_QUERY_IDS
        for case in [by_query_id.get(query_id)]
    }


def _variant_table_row(variant: dict[str, Any]) -> str:
    return (
        "| "
        + " | ".join(
            [
                f"`{variant['variant']['id']}`",
                f"`{variant['variant']['family']}`",
                f"{variant['page_hit_count']}/{variant['total_cases']}",
                str(variant["supporting_page_hit_count"]),
                _esc(", ".join(variant["partial_support_only_query_ids"]) or "-"),
                str(variant["baseline_passed_regression_count"]),
                str(variant["baseline_page_miss_rescue_count"]),
                _esc(", ".join(variant["regressed_query_ids"]) or "-"),
                _esc(", ".join(variant["rescued_query_ids"]) or "-"),
                _watchlist_cell(variant),
            ]
        )
        + " |"
    )


def _watchlist_cell(payload: dict[str, Any]) -> str:
    watchlist = payload.get("watchlist") or {}
    return ", ".join(
        f"{query_id}={_hit_label((watchlist.get(query_id) or {}).get('hit'))}"
        if (watchlist.get(query_id) or {}).get("hit") is not None
        else f"{query_id}=unknown"
        for query_id in WATCHLIST_QUERY_IDS
    )


def _first_unique_page_candidates(
    candidates: list[RetrievalCandidate],
    *,
    limit: int,
) -> list[RetrievalCandidate]:
    selected: list[RetrievalCandidate] = []
    seen_pages: set[int] = set()
    for candidate in candidates:
        page = candidate.chunk.page_number
        if page is None or page in seen_pages:
            continue
        selected.append(candidate)
        seen_pages.add(page)
        if len(selected) >= limit:
            break
    return selected


def _dedupe_candidates(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    deduped: list[RetrievalCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        key = dedupe_key(candidate.chunk)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _candidate_pages(candidates: list[RetrievalCandidate]) -> list[int]:
    return [candidate.chunk.page_number for candidate in candidates if candidate.chunk.page_number]


def _candidate_page_set(candidates: list[RetrievalCandidate]) -> set[int]:
    return set(_candidate_pages(candidates))


def _u02_gold_refresh_check_text(
    *,
    expected_pages: list[int],
    semantic_top_pages: list[int],
    bm25_top_pages: list[int],
    weighted_pages: list[int],
    best_pages: list[int],
    best_variant_id: str,
) -> str:
    expected = set(expected_pages)
    in_semantic = bool(expected.intersection(semantic_top_pages[:DEFAULT_FINAL_TOP_K]))
    in_bm25 = bool(expected.intersection(bm25_top_pages[:DEFAULT_FINAL_TOP_K]))
    in_weighted = bool(expected.intersection(weighted_pages[:DEFAULT_FINAL_TOP_K]))
    in_best = bool(expected.intersection(best_pages[:DEFAULT_FINAL_TOP_K]))
    if in_weighted:
        return (
            "After the gold refresh, U-02 is hit by weighted_rrf_best; this is no longer "
            "a regression and should not be attributed to guard logic."
        )
    if in_best:
        return (
            f"{best_variant_id} keeps a gold page in top6, but weighted_rrf_best does not; "
            "treat this as a guard watchlist observation, not a gold-refresh regression fix."
        )
    if in_semantic and not in_weighted:
        return (
            "Gold page appears in semantic top6 but is pushed out by BM25-heavy weighted RRF; "
            f"{best_variant_id} still does not keep it in top6."
        )
    if not in_semantic and not in_bm25:
        return "Gold page is absent from both semantic and BM25 top6 page_text candidate pools."
    return "Gold page remains outside top6 after guard ranking; needs follow-up retrieval analysis."


def _baseline_cell(baseline: dict[str, Any]) -> str:
    return (
        f"status={baseline.get('status', 'unknown')} "
        f"page_hit={_tri_state_label(baseline.get('page_hit_at_k'))}"
    )


def _hit_label(hit: object) -> str:
    if hit is True or hit == "primary_hit":
        return "primary_hit"
    if hit == "partial_support_only":
        return "partial_support_only"
    return "miss"


def _tri_state_label(value: object) -> str:
    if value is True:
        return "hit"
    if value is False:
        return "miss"
    return "unknown"


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _esc(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_guard_qdrant_path(path: Path | None) -> Path:
    if path is not None:
        return _resolve_path(path)
    suffix = time.strftime("%Y%m%d-%H%M%S")
    return _resolve_path(DEFAULT_QDRANT_ROOT.with_name(f"{DEFAULT_QDRANT_ROOT.name}-{suffix}"))


def _print_cases(cases: list[SmokeV2Case]) -> None:
    for case in cases:
        print(
            f"{case.case_id}\t{case.tier}\t{case.document_key}\t"
            f"{case.expected_document_evidence_intent}\t{case.query}",
            flush=True,
        )


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
