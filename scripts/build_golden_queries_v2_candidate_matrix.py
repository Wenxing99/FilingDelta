from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from filingdelta.core.config import REPO_ROOT


DEFAULT_RAW_REGISTRY = Path("data/outputs/eval/raw_document_registry.json")
DEFAULT_PARSER_SMOKE_REPORT = Path("data/outputs/eval/parser_smoke_report.json")
DEFAULT_JSON_OUTPUT = Path("data/outputs/eval/golden_queries_v2_candidate_matrix.json")
DEFAULT_MD_OUTPUT = Path("docs/golden_queries_v2_candidate_matrix.md")

ALL_UNIVERSAL_TEMPLATES = tuple(f"U-{index:02d}" for index in range(1, 11))
EARNINGS_RELEASE_UNIVERSAL_TEMPLATES = ("U-01", "U-03", "U-04", "U-05", "U-10")

UNIVERSAL_TEMPLATE_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "id": "U-01",
        "short_description": "营业收入、归母净利润、ROE/ROAE",
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence": "table_row",
    },
    {
        "id": "U-02",
        "short_description": "收入按业务分部或产品构成，以及最大分部",
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence": "table_row",
    },
    {
        "id": "U-03",
        "short_description": "经营活动现金流净额与净利润对比",
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence": "table_row",
    },
    {
        "id": "U-04",
        "short_description": "收入或利润变化的主要原因",
        "expected_document_evidence_intent": "metric_attribution",
        "primary_evidence": "section_text",
    },
    {
        "id": "U-05",
        "short_description": "毛利率、费用率或净利率变化解释",
        "expected_document_evidence_intent": "metric_attribution",
        "primary_evidence": "section_text",
    },
    {
        "id": "U-06",
        "short_description": "资本开支、研发投入或长期投资披露",
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence": "table_row",
    },
    {
        "id": "U-07",
        "short_description": "分红或股息方案",
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence": "table_row",
    },
    {
        "id": "U-08",
        "short_description": "主要风险及应对措施",
        "expected_document_evidence_intent": "business_narrative",
        "primary_evidence": "section_text",
    },
    {
        "id": "U-09",
        "short_description": "未来经营重点或战略方向",
        "expected_document_evidence_intent": "business_narrative",
        "primary_evidence": "section_text",
    },
    {
        "id": "U-10",
        "short_description": "利润与现金流背离及年报解释",
        "expected_document_evidence_intent": "metric_attribution",
        "primary_evidence": "table_row",
    },
)


@dataclass(frozen=True)
class CompanyCandidateConfig:
    company: str
    aliases: tuple[str, ...]
    industry_candidates: tuple[str, ...]
    notes: tuple[str, ...]
    annual_universal_candidates: tuple[str, ...] = ALL_UNIVERSAL_TEMPLATES
    earnings_universal_candidates: tuple[str, ...] = EARNINGS_RELEASE_UNIVERSAL_TEMPLATES


COMPANY_CONFIGS: tuple[CompanyCandidateConfig, ...] = (
    CompanyCandidateConfig(
        company="PDD",
        aliases=("PDD",),
        industry_candidates=(),
        notes=("业绩公告特例；不要强行套分红、风险或战略类模板。",),
    ),
    CompanyCandidateConfig(
        company="Trip.com",
        aliases=("Trip.com", "trip-com"),
        industry_candidates=("OTA-01", "OTA-02"),
        notes=("英文 20-F；evidence 标签可能不同于中文年报。",),
    ),
    CompanyCandidateConfig(
        company="中国平安",
        aliases=("中国平安",),
        industry_candidates=("INS-01", "INS-02", "INS-03"),
        notes=("金融/保险口径需要特别处理，尤其是 U-02、U-05、U-06。",),
    ),
    CompanyCandidateConfig(
        company="中国海洋石油",
        aliases=("中国海洋石油", "中国海油"),
        industry_candidates=("OIL-01", "OIL-02"),
        notes=("需要显式标注与设计文档“中国海油”的 alias 关系。",),
    ),
    CompanyCandidateConfig(
        company="中国石油",
        aliases=("中国石油", "PetroChina"),
        industry_candidates=("PTR-01",),
        notes=("新增 raw 后纳入油气专项；注意历史实际数与未来目标区分。",),
    ),
    CompanyCandidateConfig(
        company="中国神华",
        aliases=("中国神华",),
        industry_candidates=("COAL-01", "COAL-02"),
        notes=("煤炭、电力、铁路、港口、航运、化工一体化适合做 section_text 检查。",),
    ),
    CompanyCandidateConfig(
        company="中远海控",
        aliases=("中远海控",),
        industry_candidates=("SHIP-01", "SHIP-02", "SHIP-03"),
        notes=("同时覆盖集装箱指标、运价/利润归因和红海叙事。",),
    ),
    CompanyCandidateConfig(
        company="长江电力",
        aliases=("长江电力", "Yangtze Power"),
        industry_candidates=("HYDRO-01", "HYDRO-02", "HYDRO-03"),
        notes=("新增 raw 后纳入水电专项；注意发电量、上网电量、售电量和电价单位。",),
    ),
    CompanyCandidateConfig(
        company="宁德时代",
        aliases=("宁德时代",),
        industry_candidates=("BAT-01", "BAT-02"),
        notes=("动力/储能收入和盈利归因需要一起确认。",),
    ),
    CompanyCandidateConfig(
        company="安踏体育",
        aliases=("安踏体育",),
        industry_candidates=("SPORTS-01", "SPORTS-02"),
        notes=("品牌收入、库存周转等 anchor 必须精确。",),
    ),
    CompanyCandidateConfig(
        company="招商银行",
        aliases=("招商银行",),
        industry_candidates=(),
        notes=("银行 universal/general 覆盖；如果需要银行专项题，后续另设 BANK-*。",),
    ),
    CompanyCandidateConfig(
        company="比亚迪",
        aliases=("比亚迪",),
        industry_candidates=("NEV-01", "NEV-02"),
        notes=("汽车业务收入和新能源车销量归因需要精确 anchor。",),
    ),
    CompanyCandidateConfig(
        company="分众传媒",
        aliases=("分众传媒", "Focus Media"),
        industry_candidates=("MEDIA-01", "MEDIA-02"),
        notes=("新增 raw 后纳入广告/传媒专项；注意生活圈媒体覆盖与收入占比不要混用。",),
    ),
    CompanyCandidateConfig(
        company="泡泡玛特",
        aliases=("泡泡玛特",),
        industry_candidates=("IP-01", "IP-02"),
        notes=("IP 收入排序应当是强 table_row 检查。",),
    ),
    CompanyCandidateConfig(
        company="海尔智家",
        aliases=("海尔智家",),
        industry_candidates=("HA-02", "HA-03"),
        notes=("如果 evidence 支持，HA-03 可以在海尔和美的都实例化。",),
    ),
    CompanyCandidateConfig(
        company="美团",
        aliases=("美团",),
        industry_candidates=("LOCAL-01", "LOCAL-02"),
        notes=("分部 taxonomy 必须匹配年报当前原文。",),
    ),
    CompanyCandidateConfig(
        company="美的",
        aliases=("美的",),
        industry_candidates=("HA-01", "HA-03"),
        notes=("如果 evidence 支持，HA-03 可以在美的和海尔都实例化。",),
    ),
    CompanyCandidateConfig(
        company="腾讯控股",
        aliases=("腾讯控股",),
        industry_candidates=(),
        notes=("互联网平台 universal 覆盖；当前行业专项表没有腾讯专项题。",),
    ),
    CompanyCandidateConfig(
        company="贵州茅台",
        aliases=("贵州茅台",),
        industry_candidates=("BAIJIU-01", "BAIJIU-02"),
        notes=("产品收入和渠道收入表应当是强 anchor。",),
    ),
    CompanyCandidateConfig(
        company="阿里巴巴",
        aliases=("阿里巴巴",),
        industry_candidates=("BABA-01", "BABA-02"),
        notes=("必须区分业务分部收入和收入类型，不要混淆。",),
    ),
)

BLOCKABLE_INDUSTRY_QUERIES: tuple[dict[str, str], ...] = (
    {
        "query_id": "PTR-01",
        "required_company": "中国石油 / PetroChina",
        "required_filing": "annual_report",
    },
    {
        "query_id": "HYDRO-01",
        "required_company": "长江电力 / Yangtze Power",
        "required_filing": "annual_report",
    },
    {
        "query_id": "HYDRO-02",
        "required_company": "长江电力 / Yangtze Power",
        "required_filing": "annual_report",
    },
    {
        "query_id": "HYDRO-03",
        "required_company": "长江电力 / Yangtze Power",
        "required_filing": "annual_report",
    },
    {
        "query_id": "MEDIA-01",
        "required_company": "分众传媒 / Focus Media",
        "required_filing": "annual_report",
    },
    {
        "query_id": "MEDIA-02",
        "required_company": "分众传媒 / Focus Media",
        "required_filing": "annual_report",
    },
)

BLOCKED_QUERY_TO_COMPANY: dict[str, str] = {
    "PTR-01": "中国石油",
    "HYDRO-01": "长江电力",
    "HYDRO-02": "长江电力",
    "HYDRO-03": "长江电力",
    "MEDIA-01": "分众传媒",
    "MEDIA-02": "分众传媒",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the golden_queries_v2 candidate matrix from raw registry outputs."
    )
    parser.add_argument("--raw-registry", type=Path, default=DEFAULT_RAW_REGISTRY)
    parser.add_argument("--parser-smoke-report", type=Path, default=DEFAULT_PARSER_SMOKE_REPORT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args(argv)

    report = build_candidate_matrix_report(
        raw_registry_path=_resolve(args.raw_registry),
        parser_smoke_report_path=_resolve(args.parser_smoke_report),
    )
    json_output = _resolve(args.json_output)
    md_output = _resolve(args.md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_output.write_text(_render_markdown(report), encoding="utf-8")

    counts = report["counts"]
    print(
        "candidate_matrix "
        f"primary_industry={counts['primary_industry_candidate_instances']} "
        f"blocked={counts['blocked_design_queries']} "
        f"json={json_output} md={md_output}"
    )
    return 0


def build_candidate_matrix_report(
    *,
    raw_registry_path: Path,
    parser_smoke_report_path: Path,
) -> dict[str, Any]:
    raw_registry = _load_json(raw_registry_path)
    parser_smoke_report = _load_json(parser_smoke_report_path)
    smoke_by_key = {
        document["document_key"]: document
        for document in parser_smoke_report.get("documents", [])
        if isinstance(document, dict)
    }

    primary_candidates: list[dict[str, Any]] = []
    secondary_entries: list[dict[str, Any]] = []
    primary_query_ids: set[str] = set()

    for entry in raw_registry.get("documents", []):
        if not isinstance(entry, dict):
            continue
        config = _match_company_config(entry)
        if config is None:
            continue
        filing_class = _filing_class(entry)
        if filing_class is None:
            continue
        if _parser_status(entry, smoke_by_key) != "passed":
            continue
        if _is_primary_candidate(entry, filing_class):
            row = _primary_candidate_row(entry, config=config, filing_class=filing_class)
            primary_candidates.append(row)
            primary_query_ids.update(row["industry_candidates"])
        elif _is_secondary_candidate(entry, filing_class):
            secondary_entries.append(
                _secondary_candidate_row(entry, config=config, filing_class=filing_class)
            )

    primary_candidates.sort(key=_candidate_sort_key)
    secondary_entries.sort(key=_candidate_sort_key)
    blocked_queries = [
        {
            **blocked,
            "status": "blocked_missing_raw",
        }
        for blocked in BLOCKABLE_INDUSTRY_QUERIES
        if blocked["query_id"] not in primary_query_ids
    ]

    counts = _counts(primary_candidates, secondary_entries, blocked_queries)
    return {
        "schema_version": "golden_queries_v2_candidate_matrix.v1",
        "generated_at": date.today().isoformat(),
        "source_files": {
            "design_doc": "docs/golden_queries_v2_design.md",
            "coverage_doc": "docs/golden_queries_v2_coverage_mapping.md",
            "matrix_doc": "docs/golden_queries_v2_candidate_matrix.md",
            "raw_registry": _display_path(raw_registry_path),
            "parser_smoke_report": _display_path(parser_smoke_report_path),
        },
        "case_generation_policy": {
            "full_manifest_cap": None,
            "do_not_cap_at_30_36": True,
            "rule": (
                "对每份 parser-smoke-passed filing，纳入所有适用的 universal template，"
                "以及该 filing 对应的所有行业/公司专项 query。"
            ),
            "small_subset_policy": "`30-36` 风格的小包只是未来可选的快速回归 subset，不限制完整 manifest。",
            "runnable_manifest_requires": [
                "expected_pages",
                "expected_row_labels or expected_section_types where applicable",
                "expected_metric_tags where applicable",
                "primary_evidence_kind",
                "secondary_evidence_kinds",
                "expected_answer_field_ids",
                "answer_hygiene_checks",
            ],
        },
        "universal_templates": list(UNIVERSAL_TEMPLATE_DEFINITIONS),
        "primary_current_year_candidates": primary_candidates,
        "secondary_filing_expansion": secondary_entries,
        "blocked_design_queries": blocked_queries,
        "counts": counts,
    }


def _primary_candidate_row(
    entry: dict[str, Any],
    *,
    config: CompanyCandidateConfig,
    filing_class: str,
) -> dict[str, Any]:
    universal_candidates = (
        list(config.earnings_universal_candidates)
        if filing_class == "earnings_release"
        else list(config.annual_universal_candidates)
    )
    return {
        "company": config.company,
        "document_key": entry["document_key"],
        "local_path": entry["local_path"],
        "filing_class": filing_class,
        "universal_candidates": universal_candidates,
        "industry_candidates": list(config.industry_candidates),
        "status": "candidate_anchor_pending",
        "notes": list(config.notes),
    }


def _secondary_candidate_row(
    entry: dict[str, Any],
    *,
    config: CompanyCandidateConfig,
    filing_class: str,
) -> dict[str, Any]:
    universal_candidates = (
        list(config.annual_universal_candidates)
        if filing_class in {"annual_report_historical", "annual_summary"}
        else list(config.earnings_universal_candidates)
    )
    return {
        "company": config.company,
        "document_key": entry["document_key"],
        "local_path": entry["local_path"],
        "filing_class": filing_class,
        "deferred_universal_candidates": universal_candidates,
        "status": "defer_secondary_filing",
    }


def _is_primary_candidate(entry: dict[str, Any], filing_class: str) -> bool:
    return entry.get("inferred_fiscal_year") == 2025 and filing_class in {
        "annual_report",
        "20f",
        "earnings_release",
    }


def _is_secondary_candidate(entry: dict[str, Any], filing_class: str) -> bool:
    return filing_class in {
        "annual_report_historical",
        "annual_summary",
        "interim_report",
        "quarterly_report",
    }


def _match_company_config(entry: dict[str, Any]) -> CompanyCandidateConfig | None:
    company_name = str(entry.get("inferred_company_name") or "")
    filename = str(entry.get("filename") or "")
    searchable = f"{company_name} {filename}".casefold()
    for config in COMPANY_CONFIGS:
        if any(alias.casefold() in searchable for alias in config.aliases):
            return config
    return None


def _filing_class(entry: dict[str, Any]) -> str | None:
    doc_type = str(entry.get("inferred_doc_type") or "")
    filename = str(entry.get("filename") or "")
    fiscal_year = entry.get("inferred_fiscal_year")
    if doc_type in {"20f", "form_20f"}:
        return "20f"
    if doc_type == "earnings_release":
        return "earnings_release"
    if doc_type == "annual_report":
        if "摘要" in filename or "summary" in filename.casefold():
            return "annual_summary"
        return "annual_report" if fiscal_year == 2025 else "annual_report_historical"
    if doc_type == "interim_report":
        if any(token in filename for token in ("季度", "一季", "三季")) or "q" in filename.casefold():
            return "quarterly_report"
        return "interim_report"
    return None


def _parser_status(entry: dict[str, Any], smoke_by_key: dict[str, dict[str, Any]]) -> str | None:
    smoke = smoke_by_key.get(str(entry.get("document_key")))
    return str(smoke.get("status")) if smoke else None


def _counts(
    primary_candidates: list[dict[str, Any]],
    secondary_entries: list[dict[str, Any]],
    blocked_queries: list[dict[str, Any]],
) -> dict[str, int]:
    primary_full = sum(
        candidate["filing_class"] in {"annual_report", "20f"}
        for candidate in primary_candidates
    )
    primary_earnings = sum(
        candidate["filing_class"] == "earnings_release" for candidate in primary_candidates
    )
    primary_universal = sum(len(candidate["universal_candidates"]) for candidate in primary_candidates)
    primary_industry = sum(len(candidate["industry_candidates"]) for candidate in primary_candidates)
    return {
        "primary_full_annual_or_20f_filings": primary_full,
        "primary_earnings_release_special_filings": primary_earnings,
        "primary_universal_candidate_upper_bound": primary_universal,
        "primary_industry_candidate_instances": primary_industry,
        "primary_candidate_upper_bound_before_evidence_filter": primary_universal + primary_industry,
        "secondary_filing_entries": len(secondary_entries),
        "blocked_design_queries": len(blocked_queries),
    }


def _candidate_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    class_rank = {
        "earnings_release": 0,
        "20f": 1,
        "annual_report": 2,
        "annual_report_historical": 3,
        "interim_report": 4,
        "quarterly_report": 5,
        "annual_summary": 6,
    }.get(str(row.get("filing_class")), 9)
    return (class_rank, str(row.get("company") or ""), str(row.get("document_key") or ""))


def _render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Golden Queries v2 Candidate Matrix",
        "",
        f"更新时间：{report['generated_at']}",
        "",
        "这是 golden_queries_v2 的候选全集工作表，不是 runnable manifest。",
        "",
        "## 摘要",
        "",
        f"- primary full annual / 20-F filings：`{counts['primary_full_annual_or_20f_filings']}`",
        f"- primary earnings release special filings：`{counts['primary_earnings_release_special_filings']}`",
        f"- primary industry candidate instances：`{counts['primary_industry_candidate_instances']}`",
        f"- blocked design queries：`{counts['blocked_design_queries']}`",
        f"- secondary filing entries：`{counts['secondary_filing_entries']}`",
        "",
        "## Primary Current-Year Candidates",
        "",
        "| 公司 | document_key | filing_class | universal | industry | notes |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in report["primary_current_year_candidates"]:
        lines.append(
            "| "
            f"{_esc(row['company'])} | `{row['document_key']}` | `{row['filing_class']}` | "
            f"{len(row['universal_candidates'])} | "
            f"{', '.join(f'`{query_id}`' for query_id in row['industry_candidates']) or '-'} | "
            f"{_esc('; '.join(row.get('notes', [])))} |"
        )

    lines.extend(
        [
            "",
            "## Blocked Design Queries",
            "",
            "| Query ID | Required filing | status |",
            "|---|---|---|",
        ]
    )
    if report["blocked_design_queries"]:
        for row in report["blocked_design_queries"]:
            lines.append(
                f"| `{row['query_id']}` | {_esc(row['required_company'])} 年报 | `{row['status']}` |"
            )
    else:
        lines.append("| - | - | - |")

    lines.extend(
        [
            "",
            "## Secondary Filing Expansion",
            "",
            "| 公司 | document_key | filing_class | status |",
            "|---|---|---|---|",
        ]
    )
    for row in report["secondary_filing_expansion"]:
        lines.append(
            f"| {_esc(row['company'])} | `{row['document_key']}` | "
            f"`{row['filing_class']}` | `{row['status']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _esc(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
