from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filingdelta.core.config import REPO_ROOT, get_settings
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.schemas.filing import EvidenceKind, FilingDocType, FilingSource, Market


DEFAULT_OUTPUT = Path("data/outputs/eval/section_text_audit.json")


@dataclass(frozen=True)
class AuditDocument:
    document_key: str
    source: FilingSource
    expected_heading_groups: tuple[tuple[str, tuple[str, ...]], ...]


_DATE_TITLE_RE = re.compile(r"^\d{1,4}[年\-/.]\d{1,2}(?:[月\-/.]\d{1,2}日?)?$")
_MONTH_DAY_RE = re.compile(r"^\d{1,2}月\d{1,2}日$")
_PERCENT_TITLE_RE = re.compile(r"^\d+(?:\.\d+)?[%％]$")
_NUMERIC_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)?")

_GENERIC_REPEATED_TITLES = {
    "第一章 公司简介",
    "第二章 会计数据和财务指标摘要",
    "第三章 管理层讨论与分析",
    "管理层讨论与分析",
    "管理層討論及分析",
    "企業管治報告",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit FilingDelta section_text evidence quality.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the section_text audit JSON.",
    )
    return parser


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = FilingIngestionPipeline(settings=get_settings())
    documents = _build_documents()

    results = []
    for document in documents:
        print(f"auditing {document.document_key}: {document.source.source_path}")
        ingestion = pipeline.run(document.source)
        section_units = [
            unit
            for unit in ingestion.evidence_units
            if unit.metadata.chunk_kind == EvidenceKind.SECTION_TEXT
        ]
        results.append(_audit_document(document=document, section_units=section_units))

    report = {
        "documents": results,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("report:", output_path)
    for result in results:
        print(
            f"{result['document_key']}: section_text={result['section_text_count']} "
            f"unique_titles={result['unique_title_count']} "
            f"suspicious={result['suspicious_title_count']} "
            f"missing_expected={len(result['missing_expected_heading_groups'])}"
        )


def _build_documents() -> tuple[AuditDocument, ...]:
    return (
        AuditDocument(
            document_key="cmb_2025_annual",
            source=FilingSource(
                source_path=Path("data/raw/招商银行2025年度报告.pdf").resolve(),
                company_name="招商银行",
                market=Market.A_SHARE,
                doc_type=FilingDocType.ANNUAL_REPORT,
            ),
            expected_heading_groups=(
                ("ai_first", ("3.8.5", "ai first", "数智化", "科技基础设施")),
                ("risk_management", ("3.8.6", "风险", "房地产", "地方政府隐性债务", "零售贷款")),
                ("outlook_2026", ("2026", "展望", "净利息收益率", "经营中关注的重点问题")),
            ),
        ),
        AuditDocument(
            document_key="tcehy_2025_annual",
            source=FilingSource(
                source_path=Path("data/raw/腾讯控股2025年度报告.pdf").resolve(),
                company_name="腾讯控股",
                market=Market.H_SHARE,
                doc_type=FilingDocType.ANNUAL_REPORT,
            ),
            expected_heading_groups=(
                ("mda_revenue", ("收入", "管理層討論及分析")),
                ("video_accounts", ("視頻號", "视频号")),
                ("ai", ("ai", "混元", "广告", "廣告")),
                ("risk_market", ("市場風險", "市场风险", "財務風險管理")),
                ("risk_credit", ("信貸風險", "信贷风险")),
                ("risk_liquidity", ("流動性風險", "流动性风险")),
            ),
        ),
    )


def _audit_document(*, document: AuditDocument, section_units: list[Any]) -> dict[str, Any]:
    title_counter = Counter(_clean_title(unit.metadata.section_title) for unit in section_units)
    unique_titles = [title for title in title_counter if title]

    suspicious = []
    for unit in section_units:
        title = _clean_title(unit.metadata.section_title)
        reasons = _suspicious_reasons(title=title, section_type=unit.metadata.section_type or "")
        if not reasons:
            continue
        suspicious.append(
            {
                "page_number": unit.metadata.page_number,
                "section_title": title,
                "section_type": unit.metadata.section_type,
                "reasons": reasons,
            }
        )

    repeated_titles = [
        {"section_title": title, "count": count}
        for title, count in title_counter.most_common()
        if title and count > 1
    ]
    strong_titles = [
        title
        for title in unique_titles
        if not _suspicious_reasons(title=title, section_type="")
    ][:15]

    covered_expected = []
    missing_expected = []
    searchable_records = [
        {
            "title": _normalize_for_match(unit.metadata.section_title or ""),
            "text": _normalize_for_match(unit.text[:1200]),
        }
        for unit in section_units
    ]
    for group_name, clues in document.expected_heading_groups:
        normalized_clues = tuple(_normalize_for_match(clue) for clue in clues)
        matched = any(
            any(clue and (clue in record["title"] or clue in record["text"]) for clue in normalized_clues)
            for record in searchable_records
        )
        bucket = covered_expected if matched else missing_expected
        bucket.append(
            {
                "group": group_name,
                "clues": list(clues),
            }
        )

    section_type_counts = Counter(unit.metadata.section_type or "other" for unit in section_units)

    return {
        "document_key": document.document_key,
        "source_path": str(document.source.source_path),
        "section_text_count": len(section_units),
        "unique_title_count": len(unique_titles),
        "repeated_title_count": sum(1 for item in repeated_titles),
        "section_type_counts": dict(section_type_counts),
        "strong_title_sample": strong_titles,
        "repeated_title_sample": repeated_titles[:20],
        "suspicious_title_count": len(suspicious),
        "suspicious_title_sample": suspicious[:30],
        "covered_expected_heading_groups": covered_expected,
        "missing_expected_heading_groups": missing_expected,
        "noise_rate": round(len(suspicious) / len(section_units), 3) if section_units else 0.0,
    }


def _suspicious_reasons(*, title: str, section_type: str) -> list[str]:
    if not title:
        return ["empty_title"]

    reasons: list[str] = []
    stripped = title.strip()
    normalized = _normalize_for_match(stripped)

    if _PERCENT_TITLE_RE.match(stripped):
        reasons.append("naked_percent")
    if _DATE_TITLE_RE.match(stripped) or _MONTH_DAY_RE.match(stripped):
        reasons.append("date_like")
    if any(token in stripped for token in ("樓", "楼", "室", "大厦", "大廈", "街", "路", "號", "号")):
        reasons.append("address_like")
    if "..." in stripped or stripped.endswith(("…", "...")):
        reasons.append("truncated")
    if len(stripped) > 1 and len(stripped) <= 4 and section_type == "other":
        reasons.append("short_other")
    if stripped in _GENERIC_REPEATED_TITLES:
        reasons.append("generic_wrapper")
    if _looks_like_numeric_fragment(stripped):
        reasons.append("numeric_fragment")
    if normalized.startswith("略愿景") or normalized.startswith("務收入加速增長"):
        reasons.append("fragment_like")

    return reasons


def _looks_like_numeric_fragment(title: str) -> bool:
    if not title:
        return False
    match = _NUMERIC_PREFIX_RE.match(title)
    if match is None:
        return False
    rest = title[match.end() :].strip()
    if not rest:
        return True
    return rest.startswith(("月", "日", "%", "％", "點", "点", "倍", "次", "戶", "户"))


def _clean_title(title: str | None) -> str:
    if title is None:
        return ""
    return " ".join(title.split())


def _normalize_for_match(text: str) -> str:
    return "".join(text.lower().split())


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
