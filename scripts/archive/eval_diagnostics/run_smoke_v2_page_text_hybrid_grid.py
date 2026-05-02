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
from filingdelta.eval.retrieval_diagnosis import (
    BM25Index,
    RankSource,
    RetrievalCandidate,
    build_mode_result,
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
from filingdelta.schemas.filing import EvidenceKind


DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_BASELINE_REPORT = Path("data/outputs/eval/golden_queries_v2_live_pilot_20_report.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/golden_queries_v2_page_text_hybrid_grid.json")
DEFAULT_MARKDOWN = Path("data/outputs/eval/golden_queries_v2_page_text_hybrid_grid.md")
DEFAULT_QDRANT_ROOT = Path("tmp/smoke-v2-page-text-hybrid-grid-qdrant")
DEFAULT_FINAL_TOP_K = 6
SEMANTIC_TOP_N_VALUES = (5, 8, 10, 15)
BM25_TOP_N_VALUES = (5, 8, 10, 15)
RRF_K_VALUES = (20, 60)
ALPHA_SEMANTIC_VALUES = (0.25, 0.4, 0.5, 0.6, 0.75)
GRID_MODE = "weighted_rrf_page_text"


@dataclass(frozen=True)
class GridConfig:
    semantic_top_n: int
    bm25_top_n: int
    rrf_k: int
    alpha_semantic: float

    @property
    def id(self) -> str:
        alpha = str(self.alpha_semantic).replace(".", "p")
        return (
            f"s{self.semantic_top_n}_b{self.bm25_top_n}_"
            f"k{self.rrf_k}_a{alpha}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "semantic_top_n": self.semantic_top_n,
            "bm25_top_n": self.bm25_top_n,
            "rrf_k": self.rrf_k,
            "alpha_semantic": self.alpha_semantic,
            "alpha_bm25": round(1 - self.alpha_semantic, 6),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a page_text-only weighted RRF grid search for smoke_v2 cases."
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
    report = run_page_text_hybrid_grid(
        manifest=manifest,
        cases=cases,
        baseline_by_case_id=baseline_by_case_id,
        baseline_report=baseline_report,
        final_top_k=args.final_top_k,
        qdrant_path=_resolve_grid_qdrant_path(args.qdrant_path),
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = _resolve_path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_page_text_hybrid_grid_markdown(report), encoding="utf-8")

    print("report:", output_path, flush=True)
    print("markdown:", markdown_path, flush=True)
    best = report["summary"]["best_overall"]
    print(
        "best_overall:",
        best["config"]["id"],
        f"primary_page_hit@{args.final_top_k}={best['page_hit_count']}/{best['total_cases']}",
        f"supporting_hit={best['supporting_page_hit_count']}",
        f"regressions={best['baseline_passed_regression_count']}",
        flush=True,
    )
    return report


def run_page_text_hybrid_grid(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    baseline_by_case_id: dict[str, dict[str, Any]],
    baseline_report: dict[str, Any],
    final_top_k: int,
    qdrant_path: Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    max_semantic_top_n = max(SEMANTIC_TOP_N_VALUES)
    max_bm25_top_n = max(BM25_TOP_N_VALUES)
    configs = build_grid_configs()
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
            f"[2/4] retrieving max candidates: semantic_top={max_semantic_top_n}, "
            f"bm25_top={max_bm25_top_n}",
            flush=True,
        )
        max_candidates = _retrieve_max_candidates(
            cases=cases,
            document_contexts=document_contexts,
            retriever=retriever,
            baseline_by_case_id=baseline_by_case_id,
            semantic_top_n=max_semantic_top_n,
            bm25_top_n=max_bm25_top_n,
        )
    finally:
        client.close()

    print(f"[3/4] evaluating grid: {len(configs)} configs", flush=True)
    evaluations: list[dict[str, Any]] = []
    best_seen: dict[str, Any] | None = None
    for index, config in enumerate(configs, start=1):
        evaluation = evaluate_grid_config(
            config=config,
            max_candidates=max_candidates,
            final_top_k=final_top_k,
        )
        evaluations.append(evaluation)
        best_seen = _better_config(evaluation, best_seen)
        if index == 1 or index == len(configs) or index % 25 == 0:
            assert best_seen is not None
            print(
                "[3/4] evaluating grid: "
                f"{index}/{len(configs)} configs, "
                f"current_best={best_seen['page_hit_count']}/"
                f"{best_seen['total_cases']} "
                f"regressions={best_seen['baseline_passed_regression_count']}",
                flush=True,
            )

    print("[4/4] writing reports", flush=True)
    summary = build_grid_summary(evaluations=evaluations, final_top_k=final_top_k)
    best_config_id = summary["best_overall"]["config"]["id"]
    best_cases = next(
        evaluation["cases"]
        for evaluation in evaluations
        if evaluation["config"]["id"] == best_config_id
    )
    return {
        "version": "golden_queries_v2_page_text_hybrid_grid.v1",
        "manifest_path": str(manifest.source_path),
        "manifest_version": manifest.version,
        "baseline_report": baseline_report,
        "final_top_k": final_top_k,
        "max_semantic_top_n": max_semantic_top_n,
        "max_bm25_top_n": max_bm25_top_n,
        "grid": {
            "semantic_top_n_values": list(SEMANTIC_TOP_N_VALUES),
            "bm25_top_n_values": list(BM25_TOP_N_VALUES),
            "rrf_k_values": list(RRF_K_VALUES),
            "alpha_semantic_values": list(ALPHA_SEMANTIC_VALUES),
            "config_count": len(configs),
        },
        "shadow_scope": {
            "chunk_kind": EvidenceKind.PAGE_TEXT.value,
            "answer_synthesis_run": False,
            "formal_chat_retrieval_modified": False,
            "grid_uses_cached_max_candidates": True,
            "grid_repeats_embedding_calls": False,
        },
        "qdrant_path": str(qdrant_path),
        "total_wall_ms": _elapsed_ms(started_at),
        "summary": summary,
        "documents": {
            key: context["stats"]
            for key, context in sorted(document_contexts.items())
        },
        "best_cases": best_cases,
        "config_results": [
            {key: value for key, value in evaluation.items() if key != "cases"}
            for evaluation in evaluations
        ],
    }


def build_grid_configs() -> list[GridConfig]:
    return [
        GridConfig(
            semantic_top_n=semantic_top_n,
            bm25_top_n=bm25_top_n,
            rrf_k=rrf_k,
            alpha_semantic=alpha_semantic,
        )
        for semantic_top_n in SEMANTIC_TOP_N_VALUES
        for bm25_top_n in BM25_TOP_N_VALUES
        for rrf_k in RRF_K_VALUES
        for alpha_semantic in ALPHA_SEMANTIC_VALUES
    ]


def weighted_reciprocal_rank_fusion(
    semantic_candidates: list[RetrievalCandidate],
    bm25_candidates: list[RetrievalCandidate],
    *,
    alpha_semantic: float,
    rrf_k: int,
) -> list[RetrievalCandidate]:
    if not 0 <= alpha_semantic <= 1:
        raise ValueError("alpha_semantic must be between 0 and 1.")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive.")
    _assert_page_text_candidates([*semantic_candidates, *bm25_candidates])

    weighted_sources = (
        ("semantic", alpha_semantic, semantic_candidates),
        ("bm25", 1 - alpha_semantic, bm25_candidates),
    )
    by_key: dict[tuple[object, ...], dict[str, Any]] = {}
    insertion_order: dict[tuple[object, ...], int] = {}
    next_order = 0

    for source_name, weight, candidates in weighted_sources:
        if weight <= 0:
            continue
        seen_in_source: set[tuple[object, ...]] = set()
        for rank, candidate in enumerate(candidates, start=1):
            key = dedupe_key(candidate.chunk)
            if key in seen_in_source:
                continue
            seen_in_source.add(key)
            if key not in by_key:
                by_key[key] = {
                    "chunk": candidate.chunk,
                    "score": 0.0,
                    "rank_sources": {},
                }
                insertion_order[key] = next_order
                next_order += 1
            by_key[key]["score"] += weight / (rrf_k + rank)
            by_key[key]["rank_sources"][source_name] = RankSource(
                source=source_name,
                rank=rank,
                score=candidate.score,
            )

    fused = [
        RetrievalCandidate(
            chunk=payload["chunk"],
            score=payload["score"],
            rank_sources=tuple(
                payload["rank_sources"][source]
                for source in ("semantic", "bm25")
                if source in payload["rank_sources"]
            ),
        )
        for payload in by_key.values()
    ]
    fused.sort(key=lambda candidate: (-candidate.score, insertion_order[dedupe_key(candidate.chunk)]))
    return fused


def evaluate_grid_config(
    *,
    config: GridConfig,
    max_candidates: list[dict[str, Any]],
    final_top_k: int,
) -> dict[str, Any]:
    cases = []
    for payload in max_candidates:
        semantic = payload["semantic_candidates"][: config.semantic_top_n]
        bm25 = payload["bm25_candidates"][: config.bm25_top_n]
        fused = weighted_reciprocal_rank_fusion(
            semantic,
            bm25,
            alpha_semantic=config.alpha_semantic,
            rrf_k=config.rrf_k,
        )
        result = build_grid_case_result(
            case=payload["case"],
            baseline=payload["baseline"],
            candidates=fused,
            final_top_k=final_top_k,
            config_id=config.id,
        )
        cases.append(result)

    hit_count = sum(1 for case in cases if case["hit"])
    supporting_hit_count = sum(1 for case in cases if case.get("supporting_hit"))
    partial_support_only = [
        case["query_id"]
        for case in cases
        if case.get("page_match_status") == "partial_support_only"
    ]
    return {
        "config": config.to_json(),
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


def build_grid_case_result(
    *,
    case: SmokeV2Case,
    baseline: dict[str, Any],
    candidates: list[RetrievalCandidate],
    final_top_k: int,
    config_id: str,
) -> dict[str, Any]:
    mode_result = build_mode_result(
        case=case,
        mode=GRID_MODE,
        candidates=candidates,
        final_top_k=final_top_k,
        retrieval_ms=0,
    )
    return {
        "id": case.case_id,
        "query_id": case.case_id.rsplit("::", 1)[-1],
        "company": case.company,
        "industry": case.industry,
        "query": case.query,
        "expected_document_evidence_intent": case.expected_document_evidence_intent,
        "primary_evidence_kind": case.primary_evidence_kind,
        "expected_pages": list(case.expected_pages),
        "supporting_pages": list(case.supporting_pages),
        "baseline": baseline,
        "config_id": config_id,
        "hit": mode_result["hit"],
        "supporting_hit": mode_result.get("supporting_hit", False),
        "page_match_status": mode_result.get("page_match_status", "miss"),
        "retrieved_pages": mode_result["retrieved_pages"],
        "hit_pages": mode_result["hit_pages"],
        "supporting_hit_pages": mode_result.get("supporting_hit_pages", []),
        "top_results": mode_result["top_results"],
    }


def build_grid_summary(
    *,
    evaluations: list[dict[str, Any]],
    final_top_k: int,
) -> dict[str, Any]:
    ranked = sorted(evaluations, key=_config_sort_key)
    best_overall = ranked[0]
    no_regression = [
        evaluation
        for evaluation in ranked
        if evaluation["baseline_passed_regression_count"] == 0
    ]
    low_regression = [
        evaluation
        for evaluation in ranked
        if evaluation["baseline_passed_regression_count"] <= 1
    ]
    return {
        "final_top_k": final_top_k,
        "config_count": len(evaluations),
        "best_overall": _config_summary(best_overall),
        "best_no_baseline_regression": _config_summary(no_regression[0])
        if no_regression else None,
        "best_low_regression": _config_summary(low_regression[0])
        if low_regression else None,
        "top_configs": [_config_summary(evaluation) for evaluation in ranked[:20]],
        "boundary_configs": [
            _config_summary(evaluation)
            for evaluation in ranked
            if _is_boundary_config(evaluation["config"])
        ][:20],
    }


def render_page_text_hybrid_grid_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Golden Queries v2 PageText Hybrid Grid",
        "",
        "## 摘要",
        "",
        f"- Manifest：`{report['manifest_path']}`",
        f"- Baseline report：`{report['baseline_report'].get('path', '-')}`",
        f"- Case 总数：`{summary['best_overall']['total_cases']}`",
        f"- Config 总数：`{summary['config_count']}`",
        f"- final top_k：`{summary['final_top_k']}`",
        "- 口径：只使用 `page_text` candidates；不运行 answer synthesis；不修改正式 retrieval。",
        f"- Best overall：primary_page_hit@6="
        f"`{summary['best_overall']['page_hit_count']}/{summary['best_overall']['total_cases']}`；"
        f"supporting_hit=`{summary['best_overall']['supporting_page_hit_count']}`；"
        f"partial_support_only="
        f"`{', '.join(summary['best_overall']['partial_support_only_query_ids']) or '-'}`。",
        "",
        "## Best Configs",
        "",
        "| Bucket | config | primary_page_hit@6 | supporting_hit | partial_support_only | regressions | rescues | watchlist |",
        "|---|---|---:|---:|---|---:|---:|---|",
        _summary_row("best_overall", summary["best_overall"]),
        _summary_row("best_no_baseline_regression", summary["best_no_baseline_regression"]),
        _summary_row("best_low_regression", summary["best_low_regression"]),
        "",
        "## Top Configs",
        "",
        "| Rank | config | semantic_top | bm25_top | rrf_k | alpha | primary_page_hit@6 | supporting_hit | partial_support_only | regressions | rescues | regressed cases |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for rank, config in enumerate(summary["top_configs"], start=1):
        lines.append(_config_table_row(rank, config))

    lines.extend(
        [
            "",
            "## Boundary Configs",
            "",
            "| Rank | config | semantic_top | bm25_top | rrf_k | alpha | primary_page_hit@6 | supporting_hit | partial_support_only | regressions | rescues | regressed cases |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for rank, config in enumerate(summary["boundary_configs"], start=1):
        lines.append(_config_table_row(rank, config))

    lines.extend(
        [
            "",
            "## Best Overall Case Detail",
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

    lines.append("")
    for case in report["best_cases"]:
        lines.append(f"### `{case['id']}`")
        lines.append(
            "- Expected pages："
            f"`{', '.join(str(page) for page in case['expected_pages']) or '-'}`"
        )
        lines.append(
            "- Supporting pages："
            f"`{', '.join(str(page) for page in case.get('supporting_pages', [])) or '-'}`"
        )
        lines.append(
            "- Retrieved pages："
            f"`{', '.join(str(page) for page in case['retrieved_pages']) or '-'}`"
        )
        for top in case["top_results"][:3]:
            source_bits = ", ".join(
                f"{source['source']}#{source['rank']}"
                for source in top["rank_sources"]
            )
            lines.append(
                "  - "
                f"rank={top['rank']} page={top['page_number']} "
                f"source={source_bits}: {_esc(top['preview'])}"
            )
        lines.append("")

    return "\n".join(lines)


def load_baseline_report_context(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
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
        contexts[case_id] = build_baseline_context_from_query(query_result)
    return contexts, {
        "path": str(path),
        "status": "loaded",
        "loaded_cases": len(contexts),
        "mode": payload.get("mode"),
        "top_k": payload.get("top_k"),
    }


def build_baseline_context_from_query(query_result: dict[str, Any]) -> dict[str, Any]:
    scores = query_result.get("scores") or {}
    observed = query_result.get("observed") or {}
    expected = query_result.get("expected") or {}
    return {
        "status": query_result.get("status") or "unknown",
        "page_hit_at_k": _first_score(scores, "page_hit@"),
        "intent_hit": scores.get("intent_hit"),
        "route_hit": scores.get("route_hit"),
        "observed_intent": observed.get("document_evidence_intent"),
        "observed_route": observed.get("route"),
        "observed_pages": list(observed.get("citation_pages") or []),
        "expected_pages": list(expected.get("pages") or []),
        "failure_reasons": list(query_result.get("failure_reasons") or []),
    }


def unknown_baseline_context() -> dict[str, Any]:
    return {
        "status": "unknown",
        "page_hit_at_k": None,
        "intent_hit": None,
        "route_hit": None,
        "observed_intent": None,
        "observed_route": None,
        "observed_pages": [],
        "expected_pages": [],
        "failure_reasons": [],
    }


def _prepare_documents(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    pipeline: FilingIngestionPipeline,
    indexer: DocumentChunkIndexer,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    document_keys = sorted({case.document_key for case in cases})
    for index, document_key in enumerate(document_keys, start=1):
        document = manifest.documents.require(document_key)
        if not document.exists:
            raise SystemExit(f"source document not found: {document.source_path}")

        print(f"[1/4] preparing documents: {index}/{len(document_keys)} {document_key}", flush=True)
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

        all_chunks = evidence_units_to_retrieved_chunks(
            document_id=document_id,
            evidence_units=ingestion.evidence_units,
        )
        page_text_chunks = [
            chunk for chunk in all_chunks if chunk.chunk_kind == EvidenceKind.PAGE_TEXT.value
        ]
        contexts[document_key] = {
            "document_id": document_id,
            "page_text_chunks": page_text_chunks,
            "bm25_page_text": BM25Index(page_text_chunks),
            "stats": {
                "document_id": document_id,
                "source_path": str(document.source_path),
                "parse_ms": parse_ms,
                "index_ms": index_ms,
                "page_text_count": len(page_text_chunks),
                "section_text_count": sum(
                    1 for chunk in all_chunks if chunk.chunk_kind == EvidenceKind.SECTION_TEXT.value
                ),
                "table_row_count": sum(
                    1 for chunk in all_chunks if chunk.chunk_kind == EvidenceKind.TABLE_ROW.value
                ),
            },
        }
    return contexts


def _retrieve_max_candidates(
    *,
    cases: list[SmokeV2Case],
    document_contexts: dict[str, dict[str, Any]],
    retriever: DocumentChunkRetriever,
    baseline_by_case_id: dict[str, dict[str, Any]],
    semantic_top_n: int,
    bm25_top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[2/4] retrieving max candidates: {index}/{len(cases)} {case.case_id}", flush=True)
        context = document_contexts[case.document_key]
        semantic_chunks = retriever.retrieve(
            document_id=context["document_id"],
            question=case.query,
            top_k=semantic_top_n,
            chunk_kind=EvidenceKind.PAGE_TEXT.value,
        )
        semantic_candidates = rank_semantic_chunks(semantic_chunks, top_k=semantic_top_n)
        bm25_candidates = context["bm25_page_text"].search(case.query, top_k=bm25_top_n)
        _assert_page_text_candidates([*semantic_candidates, *bm25_candidates])
        rows.append(
            {
                "case": case,
                "baseline": baseline_by_case_id.get(case.case_id, unknown_baseline_context()),
                "semantic_candidates": semantic_candidates,
                "bm25_candidates": bm25_candidates,
            }
        )
    return rows


def _config_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": evaluation["config"],
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


def _config_sort_key(evaluation: dict[str, Any]) -> tuple[int, int, int, int, float]:
    config = evaluation["config"]
    return (
        -evaluation["page_hit_count"],
        evaluation["baseline_passed_regression_count"],
        -evaluation["baseline_page_miss_rescue_count"],
        config["semantic_top_n"] + config["bm25_top_n"],
        abs(config["alpha_semantic"] - 0.5),
    )


def _better_config(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
) -> dict[str, Any]:
    if current is None:
        return candidate
    return candidate if _config_sort_key(candidate) < _config_sort_key(current) else current


def _watchlist_status(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_query_id = {case["query_id"]: case for case in cases}
    return {
        query_id: {
            "hit": bool(case["hit"]) if case else None,
            "baseline_status": (case or {}).get("baseline", {}).get("status"),
            "retrieved_pages": (case or {}).get("retrieved_pages", []),
        }
        for query_id in ("INS-01", "U-02")
        for case in [by_query_id.get(query_id)]
    }


def _is_boundary_config(config: dict[str, Any]) -> bool:
    return (
        config["semantic_top_n"] == max(SEMANTIC_TOP_N_VALUES)
        or config["bm25_top_n"] == max(BM25_TOP_N_VALUES)
        or config["alpha_semantic"] in {min(ALPHA_SEMANTIC_VALUES), max(ALPHA_SEMANTIC_VALUES)}
    )


def _summary_row(bucket: str, payload: dict[str, Any] | None) -> str:
    if payload is None:
        return f"| `{bucket}` | - | - | - | - | - | - | - |"
    return (
        "| "
        + " | ".join(
            [
                f"`{bucket}`",
                f"`{payload['config']['id']}`",
                f"{payload['page_hit_count']}/{payload['total_cases']}",
                str(payload["supporting_page_hit_count"]),
                _esc(", ".join(payload["partial_support_only_query_ids"]) or "-"),
                str(payload["baseline_passed_regression_count"]),
                str(payload["baseline_page_miss_rescue_count"]),
                _watchlist_cell(payload),
            ]
        )
        + " |"
    )


def _config_table_row(rank: int, payload: dict[str, Any]) -> str:
    config = payload["config"]
    return (
        "| "
        + " | ".join(
            [
                str(rank),
                f"`{config['id']}`",
                str(config["semantic_top_n"]),
                str(config["bm25_top_n"]),
                str(config["rrf_k"]),
                str(config["alpha_semantic"]),
                f"{payload['page_hit_count']}/{payload['total_cases']}",
                str(payload["supporting_page_hit_count"]),
                _esc(", ".join(payload["partial_support_only_query_ids"]) or "-"),
                str(payload["baseline_passed_regression_count"]),
                str(payload["baseline_page_miss_rescue_count"]),
                _esc(", ".join(payload["regressed_query_ids"]) or "-"),
            ]
        )
        + " |"
    )


def _watchlist_cell(payload: dict[str, Any]) -> str:
    watchlist = payload.get("watchlist") or {}
    parts = []
    for query_id in ("INS-01", "U-02"):
        status = watchlist.get(query_id) or {}
        hit = status.get("hit")
        parts.append(f"{query_id}={_hit_label(hit) if hit is not None else 'unknown'}")
    return ", ".join(parts)


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


def _assert_page_text_candidates(candidates: list[RetrievalCandidate]) -> None:
    non_page_text = [
        candidate.chunk.chunk_kind
        for candidate in candidates
        if candidate.chunk.chunk_kind != EvidenceKind.PAGE_TEXT.value
    ]
    if non_page_text:
        raise ValueError(
            "PageText hybrid grid only accepts page_text chunks; got "
            f"{sorted(set(str(kind) for kind in non_page_text))}"
        )


def _first_score(scores: dict[str, Any], prefix: str) -> Any:
    for key, value in scores.items():
        if key.startswith(prefix):
            return value
    return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _esc(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _print_cases(cases: list[SmokeV2Case]) -> None:
    for case in cases:
        print(
            f"{case.case_id}\t{case.tier}\t{case.document_key}\t"
            f"{case.expected_document_evidence_intent}\t{case.query}",
            flush=True,
        )


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_grid_qdrant_path(path: Path | None) -> Path:
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
