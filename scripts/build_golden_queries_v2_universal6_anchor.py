from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any, Literal

from filingdelta.core.config import REPO_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from search_golden_queries_v2_evidence import (  # noqa: E402
    PAGE_NUMBER_POLICY,
    _auto_anchor_status,
    _dedupe_terms,
    _load_pages,
    _query_terms,
    _search_pages_with_terms,
)


DEFAULT_CANDIDATE_MATRIX = Path("data/outputs/eval/golden_queries_v2_candidate_matrix.json")
DEFAULT_RAW_REGISTRY = Path("data/outputs/eval/raw_document_registry.json")
DEFAULT_REVIEW_NOTES = Path("docs/eval_inputs/golden_queries_v2_universal6_review_notes.json")
DEFAULT_JSON_OUTPUT = Path("data/outputs/eval/golden_queries_v2_universal6_anchor_matrix.json")
DEFAULT_MD_OUTPUT = Path("docs/golden_queries_v2_universal6_anchor_review_packet.md")
DEFAULT_TOP_PAGES = 5

Route = Literal["document_only", "concept_only", "mixed", "unsupported"]
Intent = Literal["metric_value", "metric_attribution", "business_narrative", "fallback"]
EvidenceKind = Literal["table_row", "section_text", "page_text"]


@dataclass(frozen=True)
class UniversalQueryDefinition:
    query_id: str
    company: str
    query: str
    expected_route: Route
    expected_document_evidence_intent: Intent
    evidence_kinds: tuple[EvidenceKind, ...]
    area: str
    expected_answer_field_ids: tuple[str, ...]
    expected_answer_fields_label: str
    forbidden_failure_modes: tuple[str, ...]
    search_terms: tuple[str, ...]


UNIVERSAL6_DEFINITIONS: tuple[UniversalQueryDefinition, ...] = (
    UniversalQueryDefinition(
        query_id="U-01",
        company="招商银行",
        query="招商银行本报告期营业收入、归母净利润和 ROE/ROAE 分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row", "page_text"),
        area="financial highlights, ROE/ROAE",
        expected_answer_field_ids=(
            "operating_income",
            "net_profit_attributable_parent",
            "roe_roae",
        ),
        expected_answer_fields_label="营业收入、归母净利润、ROE/ROAE",
        forbidden_failure_modes=("uses interim figures for annual query",),
        search_terms=(
            "营业收入",
            "归属于本行股东的净利润",
            "归母净利润",
            "净利润",
            "平均总资产收益率",
            "加权平均净资产收益率",
            "ROE",
            "ROAE",
            "主要财务指标",
            "经营业绩",
        ),
    ),
    UniversalQueryDefinition(
        query_id="U-02",
        company="腾讯控股",
        query="腾讯控股收入按业务分部如何构成？哪个分部最大？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row", "page_text"),
        area="business segment revenue composition",
        expected_answer_field_ids=("segment_names", "segment_revenue", "largest_segment", "share"),
        expected_answer_fields_label="分部名称、分部收入、最大分部、占比/构成",
        forbidden_failure_modes=("uses product narrative without segment revenue table",),
        search_terms=(
            "增值服务",
            "营销服务",
            "金融科技及企业服务",
            "收入",
            "分部",
            "业务分部",
            "收入构成",
            "总收入",
        ),
    ),
    UniversalQueryDefinition(
        query_id="U-03",
        company="贵州茅台",
        query="贵州茅台本期经营活动现金流净额是多少？与净利润相比如何？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row", "page_text"),
        area="operating cash flow and net profit comparison",
        expected_answer_field_ids=("operating_cash_flow_net", "net_profit", "cash_profit_comparison"),
        expected_answer_fields_label="经营活动现金流净额、净利润、现金流与利润对比",
        forbidden_failure_modes=("uses investing or financing cash flow as operating cash flow",),
        search_terms=(
            "经营活动产生的现金流量净额",
            "经营活动现金流量净额",
            "经营活动",
            "现金流量表",
            "净利润",
            "归属于上市公司股东的净利润",
            "现金流",
        ),
    ),
    UniversalQueryDefinition(
        query_id="U-04",
        company="阿里巴巴",
        query="阿里巴巴收入或利润变化的主要原因是什么？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("section_text", "table_row", "page_text"),
        area="revenue or profit change attribution",
        expected_answer_field_ids=("revenue_change", "profit_change", "management_attribution"),
        expected_answer_fields_label="收入变化、利润变化、管理层归因",
        forbidden_failure_modes=("invents macro reasons not stated in the filing",),
        search_terms=(
            "收入增长",
            "收入变动",
            "利润",
            "经营利润",
            "净利润",
            "主要由于",
            "原因",
            "驱动",
            "淘天",
            "云智能",
            "菜鸟",
            "本地生活",
        ),
    ),
    UniversalQueryDefinition(
        query_id="U-06",
        company="比亚迪",
        query="比亚迪本期研发投入、资本开支或长期投资有什么披露？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row", "section_text", "page_text"),
        area="R&D, capex, long-term investment",
        expected_answer_field_ids=("r_and_d_expense", "capex", "long_term_investment"),
        expected_answer_fields_label="研发投入、资本开支、长期投资相关披露",
        forbidden_failure_modes=("uses planned investment as current-period actual without label",),
        search_terms=(
            "研发投入",
            "研发费用",
            "资本开支",
            "资本性支出",
            "购建固定资产",
            "无形资产和其他长期资产支付的现金",
            "长期投资",
            "在建工程",
            "固定资产",
            "开发支出",
        ),
    ),
    UniversalQueryDefinition(
        query_id="U-08",
        company="中国平安",
        query="中国平安披露了哪些主要风险以及应对措施？",
        expected_route="document_only",
        expected_document_evidence_intent="business_narrative",
        evidence_kinds=("section_text", "page_text"),
        area="major risks and mitigation measures",
        expected_answer_field_ids=("risk_categories", "risk_response"),
        expected_answer_fields_label="主要风险类别、风险管理/应对措施",
        forbidden_failure_modes=("turns risk disclosure into investment advice",),
        search_terms=(
            "主要风险",
            "风险管理",
            "风险因素",
            "风险应对",
            "风险控制",
            "信用风险",
            "市场风险",
            "流动性风险",
            "操作风险",
            "保险风险",
            "应对措施",
        ),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Universal 6 non-gold evidence anchor matrix and review packet."
    )
    parser.add_argument("--candidate-matrix", type=Path, default=DEFAULT_CANDIDATE_MATRIX)
    parser.add_argument("--raw-registry", type=Path, default=DEFAULT_RAW_REGISTRY)
    parser.add_argument("--review-notes", type=Path, default=DEFAULT_REVIEW_NOTES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--top-pages", type=int, default=DEFAULT_TOP_PAGES)
    args = parser.parse_args(argv)

    if args.top_pages < 1:
        raise SystemExit("--top-pages must be >= 1.")

    report = build_universal6_anchor_report(
        candidate_matrix_path=_resolve(args.candidate_matrix),
        raw_registry_path=_resolve(args.raw_registry),
        review_notes_path=_resolve(args.review_notes),
        top_pages=args.top_pages,
    )
    json_output = _resolve(args.json_output)
    md_output = _resolve(args.md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_output.write_text(render_review_packet(report), encoding="utf-8")

    summary = report["summary"]
    print(
        "universal6_anchor "
        f"rows={summary['total_rows']} "
        f"high={summary['auto_anchor_high_confidence']} "
        f"low={summary['auto_anchor_low_confidence']} "
        f"manual={summary['needs_manual_probe']} "
        f"json={json_output} md={md_output}"
    )
    return 0


def build_universal6_anchor_report(
    *,
    candidate_matrix_path: Path,
    raw_registry_path: Path,
    review_notes_path: Path | None = None,
    top_pages: int = DEFAULT_TOP_PAGES,
) -> dict[str, Any]:
    candidate_matrix = _load_json(candidate_matrix_path)
    raw_registry = _load_json(raw_registry_path)
    review_by_key = _load_review_notes(review_notes_path)
    raw_by_key = {
        str(document.get("document_key")): document
        for document in raw_registry.get("documents", [])
        if isinstance(document, dict) and document.get("document_key")
    }
    primary_by_company = _primary_candidates_by_company(candidate_matrix)

    rows: list[dict[str, Any]] = []
    documents: dict[str, dict[str, Any]] = {}
    page_cache: dict[str, list[tuple[int, str]]] = {}

    for definition in UNIVERSAL6_DEFINITIONS:
        filing = _select_filing(definition=definition, primary_by_company=primary_by_company)
        raw_document = raw_by_key.get(str(filing["document_key"]), {})
        row = _base_row(definition=definition, filing=filing, raw_document=raw_document)
        row.update(
            _search_anchor_candidates(
                row=row,
                definition=definition,
                page_cache=page_cache,
                top_pages=top_pages,
            )
        )
        _apply_review_note(row, review_by_key.get((row["company"], row["query_id"])))
        row["expected_pages"] = _expected_pages_from_human(row)
        row["manifest_readiness"] = _manifest_readiness(row)
        row["mvp_status"] = row["manifest_readiness"]
        row["status_sort"] = _status_sort(row["manifest_readiness"])
        rows.append(row)
        documents[str(filing["document_key"])] = _document_metadata(
            filing=filing,
            raw_document=raw_document,
        )

    return {
        "schema_version": "golden_queries_v2_universal6_anchor_matrix.v1",
        "generated_at": date.today().isoformat(),
        "source_files": {
            "candidate_matrix": _display_path(candidate_matrix_path),
            "raw_registry": _display_path(raw_registry_path),
            "review_notes": (
                _display_path(review_notes_path)
                if review_notes_path is not None and review_notes_path.exists()
                else None
            ),
        },
        "policy": {
            "not_a_runnable_manifest": True,
            "does_not_modify_smoke_manifest": True,
            "expected_pages_policy": (
                "expected_pages only come from human_confirmed_pages plus "
                "human_corrected_pages; candidate_pages are not promoted."
            ),
            "merge_plan": (
                "Later merge by concatenating this report's rows with the industry "
                "evidence matrix rows, then run build_golden_queries_v2_smoke_manifest.py "
                "only after human_confirmed_pages or human_corrected_pages are populated."
            ),
            "page_number_policy": PAGE_NUMBER_POLICY,
        },
        "summary": _summarize(rows),
        "selected_query_ids": [definition.query_id for definition in UNIVERSAL6_DEFINITIONS],
        "documents": sorted(documents.values(), key=lambda document: document["document_key"]),
        "rows": rows,
    }


def render_review_packet(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Golden Queries v2 Universal 6 Anchor Review Packet",
        "",
        f"更新时间：{report['generated_at']}",
        "",
        "这是把 smoke set 从 14 扩到 20 之前的 universal/general anchor 候选包，不是 runnable manifest。",
        "",
        "## 摘要",
        "",
        f"- Universal 6 行数：`{summary['total_rows']}`",
        f"- 高置信候选：`{summary['auto_anchor_high_confidence']}`",
        f"- 低置信候选：`{summary['auto_anchor_low_confidence']}`",
        f"- no-hit / 需继续 probe：`{summary['needs_manual_probe']}`",
        f"- ready_for_manifest：`{summary['ready_for_manifest']}`",
        "",
        "## 口径",
        "",
        "- `candidate_pages` 是原文搜索候选页，不等于 `expected_pages`。",
        "- 只有 `human_confirmed_pages` / `human_corrected_pages` 能生成 `expected_pages`。",
        "- `codex_suggested_gold_pages` 只是 Codex 建议页，仍需你最终确认，不会直接进入 manifest。",
        "- 页码使用 runtime / PyMuPDF 的 1-based page number，不使用 PDF 印刷页码。",
        "- `score_band` 只是页级搜索得分等级；最终 anchor 状态以 `status` 和人工页码确认为准。",
        "",
        "## 逐条候选",
        "",
        "| 公司 | Query ID | 完整 query | 候选页 | human 页 | Codex 建议页 | review_status | status | score | snippet |",
        "|---|---:|---|---|---|---|---|---|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(_review_packet_row(row))
    lines.append("")
    return "\n".join(lines)


def _primary_candidates_by_company(candidate_matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in candidate_matrix.get("primary_current_year_candidates", []):
        if not isinstance(row, dict):
            continue
        company = str(row.get("company") or "")
        if company:
            rows[company] = row
    return rows


def _select_filing(
    *,
    definition: UniversalQueryDefinition,
    primary_by_company: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    filing = primary_by_company.get(definition.company)
    if not filing:
        raise ValueError(f"{definition.query_id}: missing candidate matrix row for {definition.company}")
    universal_candidates = {str(query_id) for query_id in filing.get("universal_candidates", [])}
    if definition.query_id not in universal_candidates:
        raise ValueError(
            f"{definition.query_id}: candidate matrix row for {definition.company} "
            "does not include this universal template."
        )
    return filing


def _base_row(
    *,
    definition: UniversalQueryDefinition,
    filing: dict[str, Any],
    raw_document: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": f"{filing['document_key']}::{definition.query_id}",
        "query_id": definition.query_id,
        "tier": "smoke_v2",
        "company": definition.company,
        "industry": "universal_general",
        "document_key": filing["document_key"],
        "local_path": filing["local_path"],
        "document_metadata": _document_metadata(filing=filing, raw_document=raw_document),
        "query": definition.query,
        "expected_route": definition.expected_route,
        "expected_document_evidence_intent": definition.expected_document_evidence_intent,
        "primary_evidence_kind": definition.evidence_kinds[0],
        "secondary_evidence_kinds": list(definition.evidence_kinds[1:]),
        "candidate_pages": [],
        "candidate_snippets": [],
        "matched_terms": [],
        "forbidden_matched_terms": [],
        "page_hits": [],
        "candidate_confidence": "none",
        "evidence_search_score": 0,
        "auto_anchor_status": "needs_manual_probe",
        "review_priority": "codex_probe",
        "codex_probe_status": "not_requested",
        "codex_probe_pages": [],
        "codex_probe_snippets": [],
        "codex_probe_matched_terms": [],
        "codex_probe_score": 0,
        "codex_probe_notes": "",
        "page_number_policy": PAGE_NUMBER_POLICY,
        "anchor_review_status": "not_reviewed",
        "human_confirmed_pages": [],
        "human_corrected_pages": [],
        "human_rejected_candidate_pages": [],
        "human_missing_fields": [],
        "human_review_notes": "",
        "codex_suggested_gold_pages": [],
        "codex_suggested_gold_notes": "",
        "expected_pages": [],
        "expected_row_labels": [],
        "expected_metric_tags": [],
        "expected_section_types": [],
        "expected_document_area_ids": [],
        "expected_answer_field_ids": list(definition.expected_answer_field_ids),
        "expected_answer_fields_label": definition.expected_answer_fields_label,
        "forbidden_failure_modes": list(definition.forbidden_failure_modes),
        "answer_hygiene_checks": _hygiene_checks(definition),
        "area": definition.area,
        "matrix_status": filing.get("status", "candidate_anchor_pending"),
        "anchor_status": "universal_anchor_search",
        "manifest_readiness": "needs_manual_probe",
        "mvp_status": "needs_manual_probe",
        "notes": "Universal 6 候选页仍需人工确认；candidate_pages 不会自动升格为 expected_pages。",
        "status_sort": 9,
        "search_terms": _search_terms(definition),
    }


def _load_review_notes(review_notes_path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if review_notes_path is None or not review_notes_path.exists():
        return {}
    payload = _load_json(review_notes_path)
    review_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        company = case.get("company")
        query_id = case.get("query_id")
        if isinstance(company, str) and isinstance(query_id, str):
            review_by_key[(company, query_id)] = case
    return review_by_key


def _apply_review_note(row: dict[str, Any], review: dict[str, Any] | None) -> None:
    if review is None:
        return
    row["anchor_review_status"] = str(review.get("status") or "human_reviewed")
    row["human_confirmed_pages"] = _int_list(review.get("human_confirmed_pages", []))
    row["human_corrected_pages"] = _int_list(review.get("human_corrected_pages", []))
    row["human_rejected_candidate_pages"] = _int_list(
        review.get("human_rejected_candidate_pages", [])
    )
    row["human_missing_fields"] = [
        str(field) for field in review.get("human_missing_fields", []) if field
    ]
    row["human_review_notes"] = str(review.get("human_review_notes") or "")
    row["codex_suggested_gold_pages"] = _int_list(
        review.get("codex_suggested_gold_pages", [])
    )
    row["codex_suggested_gold_notes"] = str(review.get("codex_suggested_gold_notes") or "")


def _int_list(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, int)]


def _document_metadata(*, filing: dict[str, Any], raw_document: dict[str, Any]) -> dict[str, Any]:
    fiscal_period = str(raw_document.get("fiscal_period") or raw_document.get("inferred_fiscal_year") or "2025")
    if fiscal_period == "2025":
        fiscal_period = "2025 annual report"
    return {
        "document_key": filing["document_key"],
        "source_path": filing["local_path"],
        "company_name": raw_document.get("inferred_company_name") or filing.get("company"),
        "ticker": raw_document.get("ticker"),
        "market": raw_document.get("inferred_market") or _infer_market(filing),
        "doc_type": raw_document.get("inferred_doc_type") or _infer_doc_type(filing),
        "fiscal_period": fiscal_period,
        "language": raw_document.get("language") or "zh",
        "filing_class": filing.get("filing_class"),
    }


def _search_anchor_candidates(
    *,
    row: dict[str, Any],
    definition: UniversalQueryDefinition,
    page_cache: dict[str, list[tuple[int, str]]],
    top_pages: int,
) -> dict[str, Any]:
    local_path = row.get("local_path")
    if not local_path:
        return {
            "anchor_notes": "缺 local_path，无法读取原文。",
            "manifest_readiness": "needs_manual_probe",
        }

    source_path = _resolve(Path(str(local_path)))
    cache_key = str(source_path)
    try:
        if cache_key not in page_cache:
            page_cache[cache_key] = _load_pages(source_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "anchor_notes": f"读取原文失败：{type(exc).__name__}: {exc}",
            "manifest_readiness": "needs_manual_probe",
        }

    hits = _search_pages_with_terms(
        row=row,
        pages=page_cache[cache_key],
        terms=_search_terms(definition),
        top_pages=top_pages,
    )
    if not hits:
        return {
            "candidate_pages": [],
            "candidate_snippets": [],
            "matched_terms": [],
            "forbidden_matched_terms": [],
            "page_hits": [],
            "candidate_confidence": "none",
            "evidence_search_score": 0,
            "auto_anchor_status": "needs_manual_probe",
            "review_priority": "codex_probe",
            "anchor_notes": "原文搜索没有找到可用候选页；继续作为 Codex probe 队列，不写 expected_pages。",
        }

    top_hit = hits[0]
    auto_status = _auto_anchor_status(row, top_hit)
    return {
        "candidate_pages": [hit.page for hit in hits],
        "candidate_snippets": [snippet for hit in hits[:2] for snippet in hit.snippets[:1]],
        "matched_terms": sorted({term for hit in hits for term in hit.matched_terms}),
        "forbidden_matched_terms": sorted(
            {term for hit in hits for term in hit.forbidden_matched_terms}
        ),
        "page_hits": [hit.to_json() for hit in hits],
        "candidate_confidence": top_hit.confidence,
        "evidence_search_score": top_hit.score,
        "auto_anchor_status": auto_status,
        "review_priority": "later_anchor_confirmation",
        "anchor_notes": "候选页来自原文搜索；仅供人工复核，不写 expected_pages。",
    }


def _search_terms(definition: UniversalQueryDefinition) -> list[str]:
    terms = list(definition.search_terms)
    terms.extend(_query_terms(definition.query))
    terms.extend(definition.expected_answer_field_ids)
    return _dedupe_terms(terms)


def _expected_pages_from_human(row: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    for key in ("human_confirmed_pages", "human_corrected_pages"):
        values = row.get(key, [])
        if not isinstance(values, list):
            continue
        for page in values:
            if isinstance(page, int) and page not in pages:
                pages.append(page)
    return pages


def _manifest_readiness(row: dict[str, Any]) -> str:
    if row.get("human_missing_fields"):
        return "needs_anchor_confirmation"
    if _expected_pages_from_human(row):
        return "ready_for_manifest"
    if row.get("candidate_pages"):
        return "needs_anchor_confirmation"
    return "needs_manual_probe"


def _hygiene_checks(definition: UniversalQueryDefinition) -> list[str]:
    checks = ["no_raw_metadata", "no_empty_parentheses"]
    if definition.expected_document_evidence_intent in {"metric_value", "metric_attribution"}:
        checks.append("unit_period_present")
    return checks


def _summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_rows": len(rows),
        "candidate_rows": sum(bool(row["candidate_pages"]) for row in rows),
        "auto_anchor_high_confidence": sum(
            row["auto_anchor_status"] == "auto_anchor_high_confidence" for row in rows
        ),
        "auto_anchor_low_confidence": sum(
            row["auto_anchor_status"] == "auto_anchor_low_confidence" for row in rows
        ),
        "needs_manual_probe": sum(row["auto_anchor_status"] == "needs_manual_probe" for row in rows),
        "human_confirmed_rows": sum(bool(row["human_confirmed_pages"]) for row in rows),
        "human_corrected_rows": sum(bool(row["human_corrected_pages"]) for row in rows),
        "ready_for_manifest": sum(bool(row["expected_pages"]) for row in rows),
    }


def _review_packet_row(row: dict[str, Any]) -> str:
    pages = _join_pages(row.get("candidate_pages", []))
    snippets = " / ".join(str(snippet) for snippet in row.get("candidate_snippets", [])[:2]) or "-"
    human_pages = _join_pages(_expected_pages_from_human(row))
    suggested_pages = _join_pages(row.get("codex_suggested_gold_pages", []))
    return (
        f"| {_esc(row['company'])} | `{row['query_id']}` | {_esc(row['query'])} | "
        f"{pages} | {human_pages} | {suggested_pages} | "
        f"`{row.get('anchor_review_status', 'not_reviewed')}` | "
        f"`{row['auto_anchor_status']}` | "
        f"{row.get('evidence_search_score', 0)} | "
        f"{_esc(_review_notes_summary(row, snippets))} |"
    )


def _review_notes_summary(row: dict[str, Any], snippets: str) -> str:
    notes = [
        note
        for note in (
            row.get("human_review_notes"),
            row.get("codex_suggested_gold_notes"),
            snippets,
        )
        if note
    ]
    return " / ".join(str(note) for note in notes) or "-"


def _status_sort(status: str) -> int:
    return {
        "needs_anchor_confirmation": 0,
        "needs_manual_probe": 1,
        "blocked_missing_raw": 2,
        "ready_for_manifest": 3,
    }.get(status, 9)


def _infer_market(filing: dict[str, Any]) -> str:
    document_key = str(filing.get("document_key") or "").casefold()
    if any(token in document_key for token in ("baba", "tencent", "腾讯", "阿里")):
        return "h_share"
    return "other"


def _infer_doc_type(filing: dict[str, Any]) -> str:
    if str(filing.get("filing_class")) == "20f":
        return "20f"
    return "annual_report"


def _join_pages(pages: object) -> str:
    if not isinstance(pages, list):
        return "-"
    joined = ", ".join(str(page) for page in pages if isinstance(page, int))
    return joined or "-"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _esc(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
