from __future__ import annotations

from collections import Counter
from typing import Any

from filingdelta.eval.retrieval_diagnosis import (
    DIAGNOSIS_MODES,
    DIAGNOSIS_SCOPE_ID,
    RetrievalCandidate,
    candidate_to_json,
    normalize_for_match,
    preview_text,
    tokenize_for_bm25,
)
from filingdelta.schemas.filing import EvidenceKind, EvidenceUnit, ParsedFiling


FAILURE_PROBE_TARGET_QUERY_IDS = (
    "HA-02",
    "HYDRO-01",
    "HA-03",
    "OTA-01",
    "SPORTS-01",
    "NEV-01",
    "BAIJIU-01",
    "BABA-01",
)

RANK_NOT_IN_TOP_CANDIDATES = "not_in_top_candidates"

_GENERIC_ROW_LABELS = {"营业收入", "归属股东净利润", "净资产收益率", "资本开支"}
_STATEMENT_NOTE_HINTS = (
    "财务报表附注",
    "財務報表附註",
    "合并财务报表",
    "綜合財務報表",
    "financial statements",
    "notes to",
    "consolidated statements",
)
_BUSINESS_SPECIFIC_QUERY_TERMS = (
    "住宿",
    "交通",
    "旅游",
    "商旅",
    "鞋服",
    "电池",
    "新能源汽车",
    "发电量",
    "上网电量",
    "飞天",
    "茅台酒",
    "核心本地商业",
    "云智能",
    "国际数字商业",
    "库存",
    "渠道",
)


def build_gold_page_coverage(
    *,
    parsed_filing: ParsedFiling,
    evidence_units: list[EvidenceUnit],
    expected_pages: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    pages_by_number = {page.page_number: page for page in parsed_filing.pages}
    evidence_by_page: dict[int, list[EvidenceUnit]] = {}
    for unit in evidence_units:
        evidence_by_page.setdefault(unit.metadata.page_number, []).append(unit)

    pages: list[dict[str, Any]] = []
    for page_number in expected_pages:
        page = pages_by_number.get(page_number)
        page_units = evidence_by_page.get(page_number, [])
        units_by_kind = {
            kind.value: [unit for unit in page_units if unit.metadata.chunk_kind == kind]
            for kind in EvidenceKind
        }
        pages.append(
            {
                "page_number": page_number,
                "parsed_page_exists": page is not None,
                "has_page_text": bool(units_by_kind[EvidenceKind.PAGE_TEXT.value]),
                "has_section_text": bool(units_by_kind[EvidenceKind.SECTION_TEXT.value]),
                "has_table_row": bool(units_by_kind[EvidenceKind.TABLE_ROW.value]),
                "evidence_counts": {
                    kind: len(units) for kind, units in units_by_kind.items()
                },
                "page_snippet": preview_text(page.text if page is not None else "", limit=220),
                "evidence_snippets": [
                    _evidence_unit_summary(unit)
                    for unit in _select_representative_units(page_units, limit=5)
                ],
                "table_row_labels": sorted(
                    {
                        unit.metadata.row_label
                        for unit in units_by_kind[EvidenceKind.TABLE_ROW.value]
                        if unit.metadata.row_label
                    }
                ),
                "section_headings": sorted(
                    {
                        unit.metadata.section_title
                        for unit in units_by_kind[EvidenceKind.SECTION_TEXT.value]
                        if unit.metadata.section_title
                    }
                ),
                "metric_tags": sorted(
                    {
                        tag
                        for unit in page_units
                        for tag in unit.metadata.metric_tags
                    }
                ),
            }
        )

    return {
        "expected_pages": list(expected_pages),
        "all_expected_pages_parsed": all(page["parsed_page_exists"] for page in pages),
        "any_expected_page_parsed": any(page["parsed_page_exists"] for page in pages),
        "any_evidence_unit_on_gold_page": any(
            any(page["evidence_counts"].values()) for page in pages
        ),
        "any_page_text_on_gold_page": any(page["has_page_text"] for page in pages),
        "any_section_text_on_gold_page": any(page["has_section_text"] for page in pages),
        "any_table_row_on_gold_page": any(page["has_table_row"] for page in pages),
        "pages": pages,
    }


def rank_expected_pages(
    *,
    candidates: list[RetrievalCandidate],
    expected_pages: list[int] | tuple[int, ...],
    final_top_k: int,
) -> dict[str, Any]:
    per_page: dict[str, Any] = {}
    best_rank: int | None = None
    best_final_rank: int | None = None

    for page in expected_pages:
        rank = next(
            (
                index
                for index, candidate in enumerate(candidates, start=1)
                if candidate.chunk.page_number == page
            ),
            None,
        )
        per_page[str(page)] = rank if rank is not None else RANK_NOT_IN_TOP_CANDIDATES
        if rank is None:
            continue
        best_rank = rank if best_rank is None else min(best_rank, rank)
        if rank <= final_top_k:
            best_final_rank = rank if best_final_rank is None else min(best_final_rank, rank)

    return {
        "status": "ranked" if best_rank is not None else RANK_NOT_IN_TOP_CANDIDATES,
        "best_rank": best_rank,
        "best_final_rank": best_final_rank,
        "final_top_k_hit": best_final_rank is not None,
        "per_expected_page": per_page,
    }


def build_false_positive_summaries(
    *,
    query: str,
    candidates: list[RetrievalCandidate],
    expected_pages: list[int] | tuple[int, ...],
    final_top_k: int,
    limit: int = 4,
) -> list[dict[str, Any]]:
    false_positives: list[dict[str, Any]] = []
    seen: set[tuple[int | None, str | None, str | None]] = set()
    for rank, candidate in enumerate(candidates[:final_top_k], start=1):
        chunk = candidate.chunk
        if chunk.page_number in expected_pages:
            continue
        key = (chunk.page_number, chunk.chunk_kind, chunk.row_label)
        if key in seen:
            continue
        seen.add(key)
        false_positives.append(
            {
                **candidate_to_json(candidate, rank=rank),
                "likely_rank_reasons": explain_false_positive(query=query, candidate=candidate),
            }
        )
        if len(false_positives) >= limit:
            break
    return false_positives


def explain_false_positive(*, query: str, candidate: RetrievalCandidate) -> list[str]:
    chunk = candidate.chunk
    reasons: list[str] = []
    overlap = _term_overlap(query, chunk.text)
    if overlap:
        reasons.append("terms_overlap:" + ",".join(overlap[:6]))
    if chunk.row_label in _GENERIC_ROW_LABELS:
        reasons.append("generic_metric_row")
    if (
        chunk.row_label in _GENERIC_ROW_LABELS
        and any(term in query for term in _BUSINESS_SPECIFIC_QUERY_TERMS)
    ):
        reasons.append("wrong_table_family")
    normalized_text = normalize_for_match(chunk.text)
    if any(normalize_for_match(hint) in normalized_text for hint in _STATEMENT_NOTE_HINTS):
        reasons.append("appendix_or_financial_statement_note")
    if chunk.page_number is not None and chunk.page_number >= 100:
        reasons.append("late_report_page")
    if chunk.chunk_kind == EvidenceKind.TABLE_ROW.value and chunk.row_label is None:
        reasons.append("table_row_without_specific_label")
    return reasons or ["unexplained_rank_signal"]


def classify_failure_category(
    *,
    expected_intent: str,
    live_observed_intent: str | None,
    gold_page_coverage: dict[str, Any],
    mode_rankings: dict[str, dict[str, Any]],
    top_false_positives: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    page_rescued_by = [
        mode
        for mode in ("bm25_only", "hybrid_rrf")
        if mode_rankings.get(mode, {}).get("final_top_k_hit")
        and not mode_rankings.get("semantic_only", {}).get("final_top_k_hit")
    ]

    if live_observed_intent and live_observed_intent != expected_intent:
        category = (
            "page_rescued_but_live_intent_mismatch"
            if page_rescued_by
            else "router_intent_mismatch"
        )
        return _classification(
            category,
            "调整 router intent 判别样例/规则，并先重跑 live pilot；不要把 page-hit-only 命中当作完整救回。",
            notes=[f"page_rescued_by={'+'.join(page_rescued_by) or '-'}"],
        )

    if not gold_page_coverage["all_expected_pages_parsed"]:
        return _classification(
            "page_number_mapping_suspect",
            "核对 parser 页码与 PDF 显示页/报告印刷页的映射，必要时给 parser 增加页码映射测试；不要直接改 gold。",
        )

    if not gold_page_coverage["any_evidence_unit_on_gold_page"]:
        return _classification(
            "gold_page_evidence_missing",
            "检查 evidence_builder 是否为 gold page 生成 page_text/section_text/table_row；补 typed evidence 构建回归测试。",
        )

    if expected_intent == "metric_value" and not gold_page_coverage["any_table_row_on_gold_page"]:
        return _classification(
            "table_extraction_gap",
            "扩展 table_row 抽取的行业指标别名/表格识别，让 gold page 的目标指标形成 typed table_row evidence，并加回归测试。",
        )

    if expected_intent in {"metric_attribution", "business_narrative"} and not (
        gold_page_coverage["any_section_text_on_gold_page"]
        or gold_page_coverage["any_table_row_on_gold_page"]
    ):
        return _classification(
            "chunking_or_page_anchor_gap",
            "调整 section/chunk evidence 构建，保留 gold page 上的标题和解释性上下文，并加页面级 evidence 覆盖测试。",
        )

    if page_rescued_by:
        return _classification(
            "gold_page_low_rank",
            "将 BM25 可命中的 gold-page 信号转成正式 retrieval rerank/boost 特征，并用 page-hit-only probe 做回归；不要改 gold。",
            notes=[f"page_rescued_by={'+'.join(page_rescued_by)}"],
        )

    if _false_positive_reasons_contain(
        top_false_positives,
        {"generic_metric_row", "wrong_table_family", "appendix_or_financial_statement_note"},
    ):
        return _classification(
            "generic_metric_row_dominance",
            "增加 metric/segment-aware table-row rerank 或 query-to-row-label 约束，降低通用财报行和附注页优先级。",
        )

    if any(
        ranking.get("status") == "ranked" and not ranking.get("final_top_k_hit")
        for ranking in mode_rankings.values()
    ):
        return _classification(
            "gold_page_low_rank",
            "在 typed evidence retrieval 后增加轻量 rerank/boost，优先提升 gold page 上的相关 evidence；用本 probe 输出的 rank 做回归。",
        )

    return _classification(
        "gold_page_low_rank",
        "检查本 probe 的 gold-page rank 与 false-positive snippets，优先实现 rerank/boost 或 evidence-kind 约束；不要改 gold。",
    )


def render_failure_probe_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Golden Queries v2 Failure Probe",
        "",
        "## 摘要",
        "",
        f"- Manifest：`{report['manifest_path']}`",
        f"- Retrieval diagnosis：`{report.get('retrieval_diagnosis_path', '-')}`",
        f"- Pilot report：`{report.get('pilot_report_path', '-')}`",
        f"- Case 总数：`{len(report['cases'])}`",
        f"- 诊断口径：`{DIAGNOSIS_SCOPE_ID}`；本报告不运行 answer synthesis，不修改 gold。",
        "",
        "## Case 归因汇总",
        "",
        "| Query ID | expected_pages | expected_intent | live_observed_intent | failure_category | recommended_next_fix |",
        "|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case['query_id']}`",
                    ", ".join(str(page) for page in case["expected_pages"]),
                    case["expected_intent"],
                    case.get("live_observed_intent") or "unknown",
                    case["failure_category"],
                    _md_escape(case["recommended_next_fix"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Case Details", ""])
    for case in report["cases"]:
        lines.extend(
            [
                f"### `{case['query_id']}`",
                "",
                f"- query：{_md_escape(case['query'])}",
                f"- expected_pages：`{', '.join(str(page) for page in case['expected_pages'])}`",
                f"- expected_intent：`{case['expected_intent']}`",
                f"- live_observed_intent：`{case.get('live_observed_intent') or 'unknown'}`；pilot_status=`{case.get('pilot_status', 'unknown')}`",
                f"- failure_category：`{case['failure_category']}`",
                f"- recommended_next_fix：{_md_escape(case['recommended_next_fix'])}",
                "",
                "Gold page coverage:",
            ]
        )
        for page in case["gold_page_coverage"]["pages"]:
            lines.append(
                "- "
                f"page={page['page_number']} parsed={page['parsed_page_exists']} "
                f"page_text={page['has_page_text']} section_text={page['has_section_text']} "
                f"table_row={page['has_table_row']} "
                f"rows=`{', '.join(page['table_row_labels']) or '-'}` "
                f"sections=`{', '.join(page['section_headings']) or '-'}` "
                f"metric_tags=`{', '.join(page['metric_tags']) or '-'}`"
            )
            if page["page_snippet"]:
                lines.append(f"  - snippet：{_md_escape(page['page_snippet'])}")
        lines.append("")
        lines.append("Rank positions:")
        for mode in DIAGNOSIS_MODES:
            ranking = case["mode_rankings"][mode]
            lines.append(
                "- "
                f"`{mode}` status=`{ranking['status']}` best_rank=`{ranking['best_rank']}` "
                f"final_top6_hit=`{ranking['final_top_k_hit']}` pages={ranking['per_expected_page']}"
            )
        lines.append("")
        lines.append("Top false positives:")
        for mode in DIAGNOSIS_MODES:
            lines.append(f"- `{mode}`")
            false_positives = case["top_false_positive_pages"].get(mode) or []
            if not false_positives:
                lines.append("  - none")
                continue
            for item in false_positives[:3]:
                reasons = ", ".join(item["likely_rank_reasons"])
                lines.append(
                    "  - "
                    f"rank={item['rank']} page={item['page_number']} kind={item['chunk_kind']} "
                    f"row={item.get('row_label') or '-'} reasons=`{_md_escape(reasons)}` "
                    f"snippet={_md_escape(item['preview'])}"
                )
        lines.append("")
    return "\n".join(lines)


def _classification(
    failure_category: str,
    recommended_next_fix: str,
    *,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "failure_category": failure_category,
        "recommended_next_fix": recommended_next_fix,
        "classification_notes": notes or [],
    }


def _evidence_unit_summary(unit: EvidenceUnit) -> dict[str, Any]:
    return {
        "evidence_kind": unit.metadata.chunk_kind.value,
        "snippet": preview_text(unit.text, limit=220),
        "row_label": unit.metadata.row_label,
        "section_title": unit.metadata.section_title,
        "section_type": unit.metadata.section_type,
        "metric_tags": unit.metadata.metric_tags,
    }


def _select_representative_units(
    units: list[EvidenceUnit],
    *,
    limit: int,
) -> list[EvidenceUnit]:
    selected: list[EvidenceUnit] = []
    for kind in (EvidenceKind.TABLE_ROW, EvidenceKind.SECTION_TEXT, EvidenceKind.PAGE_TEXT):
        selected.extend(unit for unit in units if unit.metadata.chunk_kind == kind)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _term_overlap(query: str, text: str) -> list[str]:
    query_terms = Counter(tokenize_for_bm25(query))
    if not query_terms:
        return []
    text_terms = set(tokenize_for_bm25(text))
    return [
        term
        for term, _ in query_terms.most_common()
        if term in text_terms and len(term) > 1
    ][:8]


def _false_positive_reasons_contain(
    top_false_positives: dict[str, list[dict[str, Any]]],
    reason_names: set[str],
) -> bool:
    reason_counts = Counter(
        reason.split(":", 1)[0]
        for items in top_false_positives.values()
        for item in items
        for reason in item.get("likely_rank_reasons", [])
    )
    return any(reason_counts[name] >= 2 for name in reason_names)


def _md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
