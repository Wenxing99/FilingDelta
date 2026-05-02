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
    RetrievalCandidate,
    build_mode_result,
    evidence_units_to_retrieved_chunks,
    rank_semantic_chunks,
    reciprocal_rank_fusion,
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


PAGE_TEXT_SHADOW_MODES = (
    "semantic_page_text",
    "bm25_page_text",
    "hybrid_rrf_page_text",
)
DEFAULT_MANIFEST = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_BASELINE_REPORT = Path("data/outputs/eval/golden_queries_v2_live_pilot_20_report.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/golden_queries_v2_page_text_rrf_shadow.json")
DEFAULT_MARKDOWN = Path("data/outputs/eval/golden_queries_v2_page_text_rrf_shadow.md")
DEFAULT_QDRANT_ROOT = Path("tmp/smoke-v2-page-text-rrf-shadow-qdrant")
DEFAULT_CANDIDATE_TOP_K = 50
DEFAULT_FINAL_TOP_K = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a page_text-only BM25/RRF shadow diagnosis for smoke_v2 cases."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
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

    baseline_by_case_id, baseline_report = load_baseline_report_context(
        _resolve_path(args.baseline_report)
    )
    report = run_page_text_rrf_shadow(
        manifest=manifest,
        cases=cases,
        baseline_by_case_id=baseline_by_case_id,
        baseline_report=baseline_report,
        candidate_top_k=args.candidate_top_k,
        final_top_k=args.final_top_k,
        qdrant_path=_resolve_shadow_qdrant_path(args.qdrant_path),
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = _resolve_path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_page_text_shadow_markdown(report), encoding="utf-8")

    print("report:", output_path, flush=True)
    print("markdown:", markdown_path, flush=True)
    for mode, mode_summary in report["summary"]["mode_summary"].items():
        print(
            f"{mode}: primary_page_hit@{report['final_top_k']}="
            f"{mode_summary['hit_count']}/{mode_summary['total_cases']}",
            f"supporting_hit={mode_summary['supporting_page_hit_count']}",
            flush=True,
        )
    return report


def run_page_text_rrf_shadow(
    *,
    manifest: SmokeV2Manifest,
    cases: list[SmokeV2Case],
    baseline_by_case_id: dict[str, dict[str, Any]],
    baseline_report: dict[str, Any],
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
                baseline=baseline_by_case_id.get(case.case_id, unknown_baseline_context()),
                candidate_top_k=candidate_top_k,
                final_top_k=final_top_k,
            )
            for case in cases
        ]
    finally:
        client.close()

    summary = build_page_text_shadow_summary(
        case_results=case_results,
        final_top_k=final_top_k,
    )
    return {
        "version": "golden_queries_v2_page_text_rrf_shadow.v1",
        "manifest_path": str(manifest.source_path),
        "manifest_version": manifest.version,
        "baseline_report": baseline_report,
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "modes": list(PAGE_TEXT_SHADOW_MODES),
        "shadow_scope": {
            "chunk_kind": EvidenceKind.PAGE_TEXT.value,
            "answer_synthesis_run": False,
            "formal_chat_retrieval_modified": False,
            "bm25_corpus_scope": "single_document_page_text_chunks",
            "rrf_scope": "same_document_semantic_and_bm25_page_text_candidates",
        },
        "qdrant_path": str(qdrant_path),
        "total_wall_ms": _elapsed_ms(started_at),
        "summary": summary,
        "documents": {
            key: context["stats"]
            for key, context in sorted(document_contexts.items())
        },
        "cases": case_results,
    }


def build_shadow_mode_result(
    *,
    case: SmokeV2Case,
    mode: str,
    candidates: list[RetrievalCandidate],
    final_top_k: int,
    retrieval_ms: int,
) -> dict[str, Any]:
    if mode not in PAGE_TEXT_SHADOW_MODES:
        raise ValueError(f"Unknown page_text shadow mode: {mode}")
    non_page_text = [
        candidate.chunk.chunk_kind
        for candidate in candidates
        if candidate.chunk.chunk_kind != EvidenceKind.PAGE_TEXT.value
    ]
    if non_page_text:
        raise ValueError(
            "PageText RRF shadow only accepts page_text chunks; got "
            f"{sorted(set(str(kind) for kind in non_page_text))}"
        )
    return build_mode_result(
        case=case,
        mode=mode,
        candidates=candidates,
        final_top_k=final_top_k,
        retrieval_ms=retrieval_ms,
    )


def build_page_text_shadow_summary(
    *,
    case_results: list[dict[str, Any]],
    final_top_k: int,
) -> dict[str, Any]:
    mode_summary: dict[str, dict[str, Any]] = {}
    for mode in PAGE_TEXT_SHADOW_MODES:
        total_cases = len(case_results)
        hit_count = sum(1 for result in case_results if result["modes"][mode]["hit"])
        supporting_hit_count = sum(
            1 for result in case_results if result["modes"][mode].get("supporting_hit")
        )
        partial_support_only = [
            result["id"]
            for result in case_results
            if result["modes"][mode].get("page_match_status") == "partial_support_only"
        ]
        mode_summary[mode] = {
            "total_cases": total_cases,
            "hit_count": hit_count,
            "primary_page_hit_count": hit_count,
            "supporting_page_hit_count": supporting_hit_count,
            "partial_support_only_count": len(partial_support_only),
            "partial_support_only_cases": partial_support_only,
            "miss_count": total_cases - hit_count,
            "hit_rate": hit_count / total_cases if total_cases else 0.0,
            "miss_cases": [
                result["id"] for result in case_results if not result["modes"][mode]["hit"]
            ],
        }

    baseline_passed = [
        result
        for result in case_results
        if result["baseline"].get("status") == "passed"
    ]
    baseline_page_misses = [
        result
        for result in case_results
        if result["baseline"].get("page_hit_at_k") is False
    ]
    return {
        "total_cases": len(case_results),
        "page_hit_at_k": final_top_k,
        "mode_summary": mode_summary,
        "mode_hits": {
            mode: f"{payload['hit_count']}/{payload['total_cases']}"
            for mode, payload in mode_summary.items()
        },
        "by_intent": _grouped_hit_rates(
            case_results,
            group_key="expected_document_evidence_intent",
        ),
        "by_primary_evidence_kind": _grouped_hit_rates(
            case_results,
            group_key="primary_evidence_kind",
        ),
        "baseline_passed_total": len(baseline_passed),
        "baseline_page_miss_total": len(baseline_page_misses),
        "previous_passed_regressions": _previous_passed_regressions(baseline_passed),
        "baseline_page_miss_rescue": _baseline_page_miss_rescue(baseline_page_misses),
    }


def render_page_text_shadow_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    final_top_k = report["final_top_k"]
    lines = [
        "# Golden Queries v2 PageText BM25/RRF Shadow",
        "",
        "## 20-case 总结",
        "",
        f"- Manifest：`{report['manifest_path']}`",
        f"- Baseline report：`{report['baseline_report'].get('path', '-')}`",
        f"- Case 总数：`{summary['total_cases']}`",
        f"- Top K：`{final_top_k}`",
        "- 口径：只使用 `page_text` chunks；不运行 answer synthesis；不修改正式 ChatQAService/retrieval。",
    ]
    for mode in PAGE_TEXT_SHADOW_MODES:
        mode_summary = summary["mode_summary"][mode]
        lines.append(
            f"- `{mode}` page_hit@{final_top_k}："
            f"`{mode_summary['hit_count']}/{mode_summary['total_cases']}` "
            f"(primary); supporting_hit="
            f"`{mode_summary['supporting_page_hit_count']}`; "
            f"partial_support_only="
            f"`{mode_summary['partial_support_only_count']}`"
        )

    lines.extend(
        [
            "",
            "## Baseline Page-Miss Rescue",
            "",
            "| Case | baseline_status | baseline_page_hit | semantic | bm25 | hybrid_rrf | rescued_by |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["baseline_page_miss_rescue"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{escape_markdown_table(row['case_id'])}`",
                    escape_markdown_table(row["baseline_status"]),
                    _tri_state_label(row["baseline_page_hit_at_k"]),
                    _hit_label(row["semantic_page_text_hit"]),
                    _hit_label(row["bm25_page_text_hit"]),
                    _hit_label(row["hybrid_rrf_page_text_hit"]),
                    escape_markdown_table(", ".join(row["rescued_by_modes"]) or "-"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Baseline Passed Regression",
            "",
            "| Mode | regression_count | cases |",
            "|---|---:|---|",
        ]
    )
    for mode, rows in summary["previous_passed_regressions"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{mode}`",
                    str(len(rows)),
                    escape_markdown_table(", ".join(row["query_id"] for row in rows) or "-"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Intent 分组命中率",
            "",
            "| Intent | Mode | hits |",
            "|---|---|---:|",
        ]
    )
    for intent, modes in summary["by_intent"].items():
        for mode, payload in modes.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{escape_markdown_table(intent)}`",
                        f"`{mode}`",
                        f"{payload['hit_count']}/{payload['total_cases']}",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Primary Evidence Kind 分组命中率",
            "",
            "| Primary evidence kind | Mode | hits |",
            "|---|---|---:|",
        ]
    )
    for evidence_kind, modes in summary["by_primary_evidence_kind"].items():
        for mode, payload in modes.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{escape_markdown_table(evidence_kind)}`",
                        f"`{mode}`",
                        f"{payload['hit_count']}/{payload['total_cases']}",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Case 明细与 Top Snippets",
            "",
            "| Case | query | expected_pages | supporting_pages | baseline | semantic_pages | bm25_pages | hybrid_pages |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for result in report["cases"]:
        modes = result["modes"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{escape_markdown_table(result['id'])}`",
                    escape_markdown_table(result["query"]),
                    ", ".join(str(page) for page in result["expected_pages"]) or "-",
                    ", ".join(str(page) for page in result.get("supporting_pages", [])) or "-",
                    escape_markdown_table(_baseline_cell(result["baseline"])),
                    _mode_pages_cell(modes["semantic_page_text"]),
                    _mode_pages_cell(modes["bm25_page_text"]),
                    _mode_pages_cell(modes["hybrid_rrf_page_text"]),
                ]
            )
            + " |"
        )

    lines.append("")
    for result in report["cases"]:
        lines.append(f"### `{result['id']}`")
        lines.append(f"- Query：{result['query']}")
        lines.append(
            "- Expected pages："
            f"`{', '.join(str(page) for page in result['expected_pages']) or '-'}`"
        )
        lines.append(
            "- Supporting pages："
            f"`{', '.join(str(page) for page in result.get('supporting_pages', [])) or '-'}`"
        )
        for mode in PAGE_TEXT_SHADOW_MODES:
            mode_result = result["modes"][mode]
            lines.append(
                f"- `{mode}` retrieved_pages="
                f"`{', '.join(str(page) for page in mode_result['retrieved_pages']) or '-'}`"
            )
            for top in mode_result["top_results"][:3]:
                source_bits = ", ".join(
                    f"{source['source']}#{source['rank']}"
                    for source in top["rank_sources"]
                )
                lines.append(
                    "  - "
                    f"rank={top['rank']} page={top['page_number']} "
                    f"kind={top['chunk_kind']} source={source_bits}: "
                    f"{escape_markdown_table(top['preview'])}"
                )
        lines.append("")

    lines.extend(
        [
            "## 口径说明",
            "",
            "- semantic_page_text 使用现有 Qdrant + DocumentChunkRetriever，并显式过滤 `chunk_kind=page_text`。",
            "- bm25_page_text 的 corpus 只包含当前 document 的 page_text chunks，避免跨文档污染。",
            "- hybrid_rrf_page_text 只融合当前 document 内 semantic page_text 与 BM25 page_text candidates。",
            "- `rescued_by` 只代表 page_hit@K 的 retrieval shadow 命中，不代表 full live pilot 或答案合成通过。",
            "",
        ]
    )
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


def _run_case(
    *,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    baseline: dict[str, Any],
    candidate_top_k: int,
    final_top_k: int,
) -> dict[str, Any]:
    print(f"running {case.case_id}", flush=True)
    mode_results: dict[str, dict[str, Any]] = {}
    for mode in PAGE_TEXT_SHADOW_MODES:
        started_at = time.perf_counter()
        candidates = _retrieve_page_text_candidates(
            mode=mode,
            case=case,
            context=context,
            retriever=retriever,
            candidate_top_k=candidate_top_k,
        )
        mode_results[mode] = build_shadow_mode_result(
            case=case,
            mode=mode,
            candidates=candidates,
            final_top_k=final_top_k,
            retrieval_ms=_elapsed_ms(started_at),
        )

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
        "baseline": baseline,
        "modes": mode_results,
        "notes": case.notes,
    }


def _retrieve_page_text_candidates(
    *,
    mode: str,
    case: SmokeV2Case,
    context: dict[str, Any],
    retriever: DocumentChunkRetriever,
    candidate_top_k: int,
) -> list[RetrievalCandidate]:
    if mode == "semantic_page_text":
        return _semantic_page_text_candidates(
            retriever=retriever,
            document_id=context["document_id"],
            case=case,
            candidate_top_k=candidate_top_k,
        )
    if mode == "bm25_page_text":
        return _bm25_page_text_candidates(
            context=context,
            case=case,
            candidate_top_k=candidate_top_k,
        )
    if mode == "hybrid_rrf_page_text":
        semantic = _semantic_page_text_candidates(
            retriever=retriever,
            document_id=context["document_id"],
            case=case,
            candidate_top_k=candidate_top_k,
        )
        bm25 = _bm25_page_text_candidates(
            context=context,
            case=case,
            candidate_top_k=candidate_top_k,
        )
        return reciprocal_rank_fusion(semantic, bm25)
    raise ValueError(f"Unknown page_text shadow mode: {mode}")


def _semantic_page_text_candidates(
    *,
    retriever: DocumentChunkRetriever,
    document_id: str,
    case: SmokeV2Case,
    candidate_top_k: int,
) -> list[RetrievalCandidate]:
    chunks = retriever.retrieve(
        document_id=document_id,
        question=case.query,
        top_k=candidate_top_k,
        chunk_kind=EvidenceKind.PAGE_TEXT.value,
    )
    return rank_semantic_chunks(chunks, top_k=candidate_top_k)


def _bm25_page_text_candidates(
    *,
    context: dict[str, Any],
    case: SmokeV2Case,
    candidate_top_k: int,
) -> list[RetrievalCandidate]:
    index = context["bm25_page_text"]
    return index.search(case.query, top_k=candidate_top_k)


def _grouped_hit_rates(
    case_results: list[dict[str, Any]],
    *,
    group_key: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in case_results:
        grouped.setdefault(str(result.get(group_key) or "unknown"), []).append(result)

    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for group, rows in sorted(grouped.items()):
        summary[group] = {}
        for mode in PAGE_TEXT_SHADOW_MODES:
            hit_count = sum(1 for row in rows if row["modes"][mode]["hit"])
            total_cases = len(rows)
            summary[group][mode] = {
                "total_cases": total_cases,
                "hit_count": hit_count,
                "hit_rate": hit_count / total_cases if total_cases else 0.0,
            }
    return summary


def _previous_passed_regressions(
    baseline_passed: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    regressions: dict[str, list[dict[str, Any]]] = {}
    for mode in PAGE_TEXT_SHADOW_MODES:
        rows = []
        for result in baseline_passed:
            mode_result = result["modes"][mode]
            if mode_result["hit"]:
                continue
            rows.append(
                {
                    "case_id": result["id"],
                    "query_id": result["query_id"],
                "expected_pages": result["expected_pages"],
                "supporting_pages": result.get("supporting_pages", []),
                "retrieved_pages": mode_result["retrieved_pages"],
                "page_match_status": mode_result.get("page_match_status", "miss"),
            }
            )
        regressions[mode] = rows
    return regressions


def _baseline_page_miss_rescue(
    baseline_page_misses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in baseline_page_misses:
        hits = {
            mode: bool(result["modes"][mode]["hit"])
            for mode in PAGE_TEXT_SHADOW_MODES
        }
        rows.append(
            {
                "case_id": result["id"],
                "query_id": result["query_id"],
                "baseline_status": result["baseline"].get("status", "unknown"),
                "baseline_page_hit_at_k": result["baseline"].get("page_hit_at_k"),
                "expected_pages": result["expected_pages"],
                "supporting_pages": result.get("supporting_pages", []),
                "semantic_page_text_hit": hits["semantic_page_text"],
                "bm25_page_text_hit": hits["bm25_page_text"],
                "hybrid_rrf_page_text_hit": hits["hybrid_rrf_page_text"],
                "partial_support_modes": [
                    mode
                    for mode in PAGE_TEXT_SHADOW_MODES
                    if result["modes"][mode].get("page_match_status") == "partial_support_only"
                ],
                "rescued_by_modes": [
                    mode
                    for mode in ("bm25_page_text", "hybrid_rrf_page_text")
                    if hits[mode]
                ],
            }
        )
    rows.sort(key=lambda row: row["query_id"])
    return rows


def _first_score(scores: dict[str, Any], prefix: str) -> Any:
    for key, value in scores.items():
        if key.startswith(prefix):
            return value
    return None


def _baseline_cell(baseline: dict[str, Any]) -> str:
    return (
        f"status={baseline.get('status', 'unknown')} "
        f"page_hit={_tri_state_label(baseline.get('page_hit_at_k'))}"
    )


def _mode_pages_cell(mode_result: dict[str, Any]) -> str:
    pages = ", ".join(str(page) for page in mode_result["retrieved_pages"]) or "-"
    status = mode_result.get("page_match_status") or ("primary_hit" if mode_result["hit"] else "miss")
    return f"{status} `{escape_markdown_table(pages)}`"


def _hit_label(hit: bool) -> str:
    return "hit" if hit else "miss"


def _tri_state_label(value: object) -> str:
    if value is True:
        return "hit"
    if value is False:
        return "miss"
    return "unknown"


def escape_markdown_table(value: object) -> str:
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


def _resolve_shadow_qdrant_path(path: Path | None) -> Path:
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
