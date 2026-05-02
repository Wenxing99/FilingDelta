from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any

from filingdelta.core.config import REPO_ROOT


DEFAULT_MATRIX = Path("data/outputs/eval/golden_queries_v2_industry_evidence_matrix.json")
DEFAULT_JSON_OUTPUT = Path("data/outputs/eval/golden_queries_v2_smoke.json")
DEFAULT_MD_OUTPUT = Path("docs/golden_queries_v2_smoke_manifest_summary.md")
MANIFEST_VERSION = "golden_queries_v2_smoke_anchor_confirmed_v0"


class SmokeManifestBuildError(ValueError):
    """Raised when smoke manifest inputs cannot be merged safely."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an anchor-confirmed golden_queries_v2 smoke manifest draft."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        action="append",
        default=None,
        help="Matrix JSON input. Repeat to merge multiple matrices.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args(argv)

    matrix_paths = [_resolve(path) for path in (args.matrix or [DEFAULT_MATRIX])]
    report = build_manifest_report(matrix_paths=matrix_paths)
    json_output = _resolve(args.json_output)
    md_output = _resolve(args.md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report["manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_output.write_text(render_summary_markdown(report), encoding="utf-8")

    summary = report["summary"]
    print(
        "smoke_manifest "
        f"included={summary['included_cases']} "
        f"excluded={summary['excluded_cases']} "
        f"documents={summary['documents']} "
        f"json={json_output} md={md_output}"
    )
    return 0


def build_manifest_report(
    *,
    matrix_path: Path | None = None,
    matrix_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if matrix_paths is None:
        if matrix_path is None:
            matrix_paths = [DEFAULT_MATRIX]
        else:
            matrix_paths = [matrix_path]
    if not matrix_paths:
        raise SmokeManifestBuildError("At least one matrix path is required.")
    rows = _load_rows_from_matrices(matrix_paths)
    included_rows = [row for row in rows if _include_row(row)]
    excluded_rows = [row for row in rows if not _include_row(row)]

    documents = _manifest_documents(included_rows)
    queries = [_manifest_query(row) for row in included_rows]
    source_matrices = [_display_path(path) for path in matrix_paths]
    manifest = {
        "version": MANIFEST_VERSION,
        "suite": "golden_queries_v2",
        "default_top_k": 6,
        "metadata": {
            "generated_at": date.today().isoformat(),
            "source_matrix": source_matrices[0] if len(source_matrices) == 1 else None,
            "source_matrices": source_matrices,
            "anchor_policy": (
                "expected_pages only come from human_confirmed_pages plus "
                "human_corrected_pages; candidate_pages, codex_anchor_pages, and "
                "codex_suggested_gold_pages are not promoted."
            ),
            "page_order_policy": "preserve human feedback order, de-duplicated",
            "not_full_golden_queries_v2": True,
        },
        "documents": documents,
        "queries": queries,
    }
    return {
        "schema_version": "golden_queries_v2_smoke_manifest_build.v1",
        "generated_at": date.today().isoformat(),
        "source_files": {"matrices": source_matrices},
        "summary": {
            "total_rows": len(rows),
            "included_cases": len(queries),
            "excluded_cases": len(excluded_rows),
            "documents": len(documents),
            "excluded_missing_human_pages": sum(
                _exclude_reason(row) == "missing_human_confirmed_or_corrected_pages"
                for row in excluded_rows
            ),
            "excluded_partial_field_gap": sum(
                _exclude_reason(row) == "human_missing_fields" for row in excluded_rows
            ),
            "excluded_no_hit_or_deferred": sum(
                _exclude_reason(row) == "no_hit_or_deferred" for row in excluded_rows
            ),
        },
        "included_cases": [_summary_row(row) for row in included_rows],
        "excluded_cases": [_excluded_summary_row(row) for row in excluded_rows],
        "manifest": manifest,
    }


def _load_rows_from_matrices(matrix_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_case_ids: dict[str, str] = {}
    for matrix_path in matrix_paths:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        matrix_name = _display_path(matrix_path)
        for index, row in enumerate(matrix.get("rows", []), start=1):
            if not isinstance(row, dict):
                continue
            case_id = str(row.get("case_id") or "")
            if not case_id:
                raise SmokeManifestBuildError(f"{matrix_name} row #{index}: missing case_id")
            if case_id in seen_case_ids:
                raise SmokeManifestBuildError(
                    f"Duplicate case_id {case_id!r} in {matrix_name}; "
                    f"already seen in {seen_case_ids[case_id]}"
                )
            seen_case_ids[case_id] = matrix_name
            rows.append(row)
    return rows


def _include_row(row: dict[str, Any]) -> bool:
    if row.get("manifest_readiness") == "blocked_missing_raw":
        return False
    if row.get("auto_anchor_status") == "needs_manual_probe":
        return False
    if row.get("human_missing_fields"):
        return False
    return bool(_expected_pages(row))


def _manifest_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        document_key = str(row["document_key"])
        if document_key in documents_by_key:
            continue
        document_metadata = row.get("document_metadata")
        if isinstance(document_metadata, dict):
            documents_by_key[document_key] = {
                "document_key": document_key,
                "source_path": document_metadata.get("source_path") or row["local_path"],
                "company_name": document_metadata.get("company_name") or row["company"],
                "ticker": document_metadata.get("ticker"),
                "market": document_metadata.get("market") or _infer_market(row),
                "doc_type": document_metadata.get("doc_type") or _infer_doc_type(row),
                "fiscal_period": document_metadata.get("fiscal_period") or "2025 annual report",
                "language": document_metadata.get("language") or _infer_language(row),
                "industry": row.get("industry"),
            }
            continue
        documents_by_key[document_key] = {
            "document_key": document_key,
            "source_path": row["local_path"],
            "company_name": row["company"],
            "ticker": None,
            "market": _infer_market(row),
            "doc_type": _infer_doc_type(row),
            "fiscal_period": "2025 annual report",
            "language": _infer_language(row),
            "industry": row.get("industry"),
        }
    return sorted(documents_by_key.values(), key=lambda document: document["document_key"])


def _manifest_query(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["case_id"],
        "tier": "smoke_v2",
        "company": row["company"],
        "industry": row["industry"],
        "document_key": row["document_key"],
        "query": row["query"],
        "query_aliases": [],
        "expected_route": row["expected_route"],
        "expected_document_evidence_intent": row["expected_document_evidence_intent"],
        "primary_evidence_kind": row["primary_evidence_kind"],
        "secondary_evidence_kinds": list(row.get("secondary_evidence_kinds", [])),
        "expected_pages": _expected_pages(row),
        "supporting_pages": _supporting_pages(row),
        "expected_row_labels": list(row.get("expected_row_labels", [])),
        "expected_metric_tags": list(row.get("expected_metric_tags", [])),
        "expected_section_types": list(row.get("expected_section_types", [])),
        "expected_document_area_ids": [],
        "expected_answer_field_ids": list(row.get("expected_answer_field_ids", [])),
        "forbidden_failure_modes": list(row.get("forbidden_failure_modes", [])),
        "answer_hygiene_checks": list(row.get("answer_hygiene_checks", [])),
        "mvp_status": "anchor_confirmed_draft",
        "notes": _manifest_notes(row),
    }


def _expected_pages(row: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    for key in ("human_confirmed_pages", "human_corrected_pages"):
        for page in row.get(key, []):
            if isinstance(page, int) and page not in pages:
                pages.append(page)
    return pages


def _supporting_pages(row: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    for page in row.get("human_supporting_pages", []):
        if isinstance(page, int) and page not in pages:
            pages.append(page)
    return pages


def _manifest_notes(row: dict[str, Any]) -> str:
    return (
        "expected_pages_source=human_confirmed_pages+human_corrected_pages; "
        f"anchor_review_status={row.get('anchor_review_status', 'not_reviewed')}; "
        f"supporting_pages={_supporting_pages(row)}; "
        f"human_review_notes={row.get('human_review_notes', '')}; "
        f"codex_suggested_gold_pages_ignored={row.get('codex_suggested_gold_pages', [])}"
    )


def _exclude_reason(row: dict[str, Any]) -> str:
    if row.get("manifest_readiness") == "blocked_missing_raw":
        return "blocked_missing_raw"
    if row.get("auto_anchor_status") == "needs_manual_probe":
        return "no_hit_or_deferred"
    if row.get("human_missing_fields"):
        return "human_missing_fields"
    if not _expected_pages(row):
        return "missing_human_confirmed_or_corrected_pages"
    return "not_included"


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "company": row["company"],
        "query_id": row["query_id"],
        "query": row["query"],
        "document_key": row["document_key"],
        "expected_pages": _expected_pages(row),
        "expected_pages_source": "human_confirmed_pages+human_corrected_pages",
        "human_confirmed_pages": list(row.get("human_confirmed_pages", [])),
        "human_corrected_pages": list(row.get("human_corrected_pages", [])),
        "human_supporting_pages": _supporting_pages(row),
        "human_review_notes": row.get("human_review_notes", ""),
    }


def _excluded_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "company": row["company"],
        "query_id": row["query_id"],
        "query": row["query"],
        "reason": _exclude_reason(row),
        "human_missing_fields": list(row.get("human_missing_fields", [])),
        "human_review_notes": row.get("human_review_notes", ""),
        "auto_anchor_status": row.get("auto_anchor_status"),
    }


def render_summary_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Golden Queries v2 Smoke Manifest Summary",
        "",
        f"更新时间：{report['generated_at']}",
        "",
        "这是 anchor-confirmed smoke_v2 manifest 草案摘要，不是全量 golden_queries_v2。",
        "",
        "## 摘要",
        "",
        f"- matrix 总行数：`{summary['total_rows']}`",
        f"- 纳入 runnable manifest 草案：`{summary['included_cases']}`",
        f"- 排除：`{summary['excluded_cases']}`",
        f"- manifest documents：`{summary['documents']}`",
        f"- 排除：缺人工页码：`{summary['excluded_missing_human_pages']}`",
        f"- 排除：人工发现字段缺口：`{summary['excluded_partial_field_gap']}`",
        f"- 排除：no-hit / defer：`{summary['excluded_no_hit_or_deferred']}`",
        "",
        "## 口径",
        "",
        "- `expected_pages` 只来自 `human_confirmed_pages + human_corrected_pages`。",
        "- `human_supporting_pages` / `supporting_pages` 只用于离线诊断展示，不进入 `expected_pages`。",
        "- `candidate_pages`、`codex_anchor_pages` 和 `codex_suggested_gold_pages` 不能自动升格为 `expected_pages`。",
        "- 页码顺序保留用户反馈顺序，并去重。",
        "",
        "## 纳入 Case",
        "",
        "| 公司 | Query ID | 问题 | expected_pages | supporting_pages | 页码来源 |",
        "|---|---:|---|---|---|---|",
    ]
    for case in report["included_cases"]:
        lines.append(_included_markdown_row(case))
    lines.extend(
        [
            "",
            "## 排除 Case",
            "",
            "| 公司 | Query ID | 问题 | 原因 | 备注 |",
            "|---|---:|---|---|---|",
        ]
    )
    for case in report["excluded_cases"]:
        if case["reason"] == "missing_human_confirmed_or_corrected_pages":
            continue
        lines.append(_excluded_markdown_row(case))
    lines.append("")
    return "\n".join(lines)


def _included_markdown_row(case: dict[str, Any]) -> str:
    pages = ", ".join(str(page) for page in case["expected_pages"])
    supporting_pages = ", ".join(str(page) for page in case.get("human_supporting_pages", [])) or "-"
    source = (
        f"confirmed={case['human_confirmed_pages'] or '-'}; "
        f"corrected={case['human_corrected_pages'] or '-'}"
    )
    return (
        f"| {_esc(case['company'])} | `{case['query_id']}` | {_esc(case['query'])} | "
        f"{pages} | {supporting_pages} | {_esc(source)} |"
    )


def _excluded_markdown_row(case: dict[str, Any]) -> str:
    notes = case.get("human_review_notes") or case.get("auto_anchor_status") or "-"
    return (
        f"| {_esc(case['company'])} | `{case['query_id']}` | {_esc(case['query'])} | "
        f"`{case['reason']}` | {_esc(notes)} |"
    )


def _infer_market(row: dict[str, Any]) -> str:
    document_key = str(row.get("document_key") or "").casefold()
    local_path = str(row.get("local_path") or "").casefold()
    if "20f" in document_key or "20f" in local_path or "adr" in document_key:
        return "adr"
    if any(token in document_key for token in ("baba", "trip", "pop", "anta")):
        return "h_share"
    return "other"


def _infer_doc_type(row: dict[str, Any]) -> str:
    document_key = str(row.get("document_key") or "").casefold()
    local_path = str(row.get("local_path") or "").casefold()
    if "20f" in document_key or "20f" in local_path:
        return "20f"
    if "earnings_release" in document_key:
        return "earnings_release"
    return "annual_report"


def _infer_language(row: dict[str, Any]) -> str:
    document_key = str(row.get("document_key") or "").casefold()
    if "trip" in document_key:
        return "en"
    return "zh"


def _esc(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
