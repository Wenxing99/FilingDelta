from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

from filingdelta.core.config import REPO_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from search_golden_queries_v2_evidence import (  # noqa: E402
    FIELD_TERMS,
    PAGE_NUMBER_POLICY,
    _field_id_terms,
    _load_pages,
    _matched_terms,
    _normalize,
    _snippets,
)


DEFAULT_MATRIX = Path("data/outputs/eval/golden_queries_v2_industry_evidence_matrix.json")
DEFAULT_JSON_OUTPUT = Path("data/outputs/eval/golden_queries_v2_anchor_confirmation_draft.json")
DEFAULT_MD_OUTPUT = Path("docs/golden_queries_v2_anchor_review_packet.md")

USER_REVIEW_KEYS = frozenset(
    {
        ("中远海控", "SHIP-01"),
        ("分众传媒", "MEDIA-01"),
        ("泡泡玛特", "IP-01"),
        ("阿里巴巴", "BABA-01"),
        ("海尔智家", "HA-02"),
        ("海尔智家", "HA-03"),
        ("美团", "LOCAL-02"),
    }
)

FIELD_TERM_OVERRIDES: dict[str, tuple[str, ...]] = {
    "accommodation_revenue": ("accommodation", "住宿", "酒店", "收入"),
    "aidc_revenue": ("AIDC", "国际数字商业", "國際數字商業", "收入"),
    "anta_revenue": ("ANTA", "安踏", "收入"),
    "artist_ips": ("artist IP", "artist IPs", "艺术家", "藝術家"),
    "asp": ("平均售价", "平均售價", "销售价格", "銷售價格", "ASP"),
    "auto_revenue": ("汽车业务", "汽車業務", "汽车", "汽車", "收入"),
    "building_media_revenue": ("楼宇媒体", "樓宇媒體", "楼宇", "樓宇", "收入"),
    "cainiao_revenue": ("菜鸟", "菜鳥", "收入"),
    "capacity_or_route_effect": ("运力", "運力", "绕航", "繞航", "航线", "航線"),
    "capex_split_or_use": ("资本开支", "資本開支", "资本支出", "勘探", "开发"),
    "channel_contribution": ("渠道", "贡献", "收入"),
    "chemical": ("化工",),
    "china_drivers": ("中国市场", "中國市場", "国内市场", "國內市場", "中国", "国内"),
    "city_coverage": ("城市", "覆盖", "覆蓋"),
    "cloud_revenue": ("云智能", "雲智能", "云", "雲", "收入"),
    "cloud_service_revenue": ("云服务", "雲服務", "收入"),
    "coal": ("煤炭", "煤"),
    "coal_sales_volume": ("煤炭销售量", "煤炭銷售量", "销售量", "銷售量"),
    "combined_ratio": ("综合成本率", "綜合成本率"),
    "commercial_coal_production": ("商品煤产量", "商品煤產量", "商品煤", "产量"),
    "corporate_travel_revenue": ("corporate travel", "商旅", "收入"),
    "customer_management_revenue": ("客户管理", "客戶管理", "收入"),
    "device_types": ("设备", "設備", "点位", "屏幕"),
    "direct_revenue": ("直销", "直銷", "收入"),
    "energy_storage": ("储能", "儲能"),
    "ess_revenue": ("储能电池", "儲能電池", "储能", "收入"),
    "ev_battery_revenue": ("动力电池", "動力電池", "收入"),
    "examples": ("项目", "布局", "例", "案例"),
    "fila_revenue": ("FILA", "斐乐", "收入"),
    "freight_rate_or_capacity_driver": ("运价", "運價", "运力", "運力", "货量", "貨量"),
    "gas_output": ("天然气产量", "天然氣產量", "天然气", "天然氣"),
    "generation": ("发电量", "發電量"),
    "goods_sales_revenue": ("商品销售", "商品銷售", "收入"),
    "growth": ("增长", "增長", "同比"),
    "inflow_or_generation_explanation": ("来水", "發電", "发电", "发电量"),
    "international_ota": ("international", "outbound", "inbound", "global"),
    "ip_names": ("THE MONSTERS", "MOLLY", "SKULLPANDA", "IP"),
    "local_services_revenue": ("本地生活", "收入"),
    "logistics_service_revenue": ("物流服务", "物流服務", "收入"),
    "management_attribution": ("管理层", "管理層", "原因", "由于", "因"),
    "margin_or_cost_explanation": ("毛利率", "成本", "费用", "費用"),
    "mobile_components_revenue": ("手机部件", "手機部件", "组装", "組裝", "收入"),
    "moutai_liquor_revenue": ("茅台酒", "收入"),
    "nbv": ("新业务价值", "新業務價值", "NBV"),
    "net_profit": ("净利润", "淨利潤", "归母", "歸母"),
    "new_energy": ("新能源",),
    "oil_gas_equivalent_output": ("油气当量", "油氣當量", "产量", "產量"),
    "on_grid_volume": ("上网电量", "上網電量"),
    "operating_profit": ("经营利润", "經營利潤", "经营溢利", "經營溢利"),
    "other_brands_revenue": ("其他品牌", "收入"),
    "other_media": ("其他媒体", "其他媒體", "影院", "收入"),
    "outbound_or_inbound_indicators": ("outbound", "inbound", "入境", "出境"),
    "overseas_coverage": ("海外", "境外", "海外市场"),
    "overseas_drivers": ("海外市场", "海外市場", "海外", "境外"),
    "packaged_tour_revenue": ("packaged-tour", "packaged tour", "旅游度假", "收入"),
    "port": ("港口",),
    "power": ("电力", "電力"),
    "product_or_channel_examples": ("产品", "產品", "渠道", "案例", "例"),
    "production": ("净产量", "淨產量", "产量", "產量"),
    "profit_change": ("利润", "利潤", "增长", "增長"),
    "profit_linkage": ("利润", "利潤", "发电量", "發電量", "来水"),
    "proprietary_products": ("Proprietary products", "自有产品", "自有產品"),
    "proved_reserves": ("证实储量", "證實儲量", "储量", "儲量"),
    "pumped_storage": ("抽蓄", "抽水蓄能"),
    "quality_explanation": ("质量", "質量", "业务质量", "業務質量"),
    "railway": ("铁路", "鐵路"),
    "ranking": ("排名", "位列", "top"),
    "rate_effect": ("运价", "運價", "费率", "費率"),
    "red_sea_wording": ("红海", "紅海"),
    "refining_throughput": ("炼油加工量", "煉油加工量", "加工量"),
    "reserve_life": ("储量寿命", "儲量壽命", "储量年限"),
    "revenue_change": ("收入", "增长", "增長", "同比"),
    "route_disruption_if_disclosed": ("红海", "紅海", "绕航", "繞航", "航线"),
    "sales_volume": ("销量", "銷量", "销售量", "銷售量"),
    "segment_names": ("分部", "业务", "業務", "segment"),
    "series_liquor_revenue": ("系列酒", "收入"),
    "shipping": ("航运", "航運"),
    "synergy_explanation": ("协同", "協同", "一体化", "一體化"),
    "taobao_tmall_revenue": ("淘天", "淘宝", "天猫", "淘寶", "天貓", "收入"),
    "tariff": ("上网电价", "上網電價", "电价", "電價"),
    "teu_volume": ("货运量", "貨運量", "箱量", "TEU", "标准箱", "標準箱"),
    "total_capex": ("资本开支", "資本開支", "资本支出"),
    "transportation_revenue": ("transportation", "票务", "票務", "收入"),
    "volume_driver": ("货量", "貨量", "箱量", "运量", "運量"),
    "wholesale_revenue": ("批发", "批發", "代理", "收入"),
    "yoy_or_pps": ("同比", "百分点", "百分點"),
    "yoy_or_share": ("同比", "占比", "佔比", "份额", "份額"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a non-manifest Codex anchor confirmation draft."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args(argv)

    report = build_confirmation_report(matrix_path=_resolve(args.matrix))
    json_output = _resolve(args.json_output)
    md_output = _resolve(args.md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_output.write_text(render_review_packet(report), encoding="utf-8")

    summary = report["summary"]
    print(
        "anchor_confirmation "
        f"rows={summary['total_rows']} "
        f"candidate_rows={summary['candidate_rows']} "
        f"full={summary['codex_anchor_confirmed_rows']} "
        f"partial={summary['codex_anchor_partial_rows']} "
        f"deferred={summary['codex_anchor_deferred_rows']} "
        f"review_packet={summary['review_packet_rows']} "
        f"json={json_output} md={md_output}"
    )
    return 0


def build_confirmation_report(*, matrix_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    page_cache: dict[str, dict[int, str]] = {}
    cases: list[dict[str, Any]] = []

    for row in matrix["rows"]:
        case = _confirmation_case(row=row, page_cache=page_cache)
        cases.append(case)

    review_packet_cases = [
        case
        for case in cases
        if (str(case["company"]), str(case["query_id"])) in USER_REVIEW_KEYS
    ]
    known_caveats = [
        _review_caveat(row)
        for row in matrix["rows"]
        if _has_review_caveat(row)
        and (str(row["company"]), str(row["query_id"])) not in USER_REVIEW_KEYS
    ]
    summary = _summarize(cases=cases, review_packet_cases=review_packet_cases)
    return {
        "schema_version": "golden_queries_v2_anchor_confirmation_draft.v1",
        "generated_at": date.today().isoformat(),
        "source_files": {"industry_matrix": _display_path(matrix_path)},
        "policy": {
            "not_a_runnable_manifest": True,
            "does_not_write_expected_pages": True,
            "does_not_write_human_fields": True,
            "page_number_policy": PAGE_NUMBER_POLICY,
            "codex_anchor_pages_policy": (
                "codex_anchor_pages 是 Codex 对候选页的原文核对草案，不等于 expected_pages。"
            ),
        },
        "summary": summary,
        "cases": cases,
        "review_packet": {
            "cases": review_packet_cases,
            "known_caveats": known_caveats,
        },
    }


def _confirmation_case(
    *,
    row: dict[str, Any],
    page_cache: dict[str, dict[int, str]],
) -> dict[str, Any]:
    base = _base_case(row)
    candidate_pages = [page for page in row.get("candidate_pages", []) if isinstance(page, int)]
    if not candidate_pages:
        return {
            **base,
            "codex_anchor_status": "codex_anchor_deferred_no_hit",
            "codex_anchor_pages": [],
            "codex_anchor_field_hits": [],
            "codex_anchor_missing_fields": list(row.get("expected_answer_field_ids", [])),
            "codex_anchor_snippets": [],
            "codex_anchor_notes": _no_hit_notes(row),
        }

    local_path = row.get("local_path")
    if not local_path:
        return {
            **base,
            "codex_anchor_status": "codex_anchor_load_error",
            "codex_anchor_pages": [],
            "codex_anchor_field_hits": [],
            "codex_anchor_missing_fields": list(row.get("expected_answer_field_ids", [])),
            "codex_anchor_snippets": [],
            "codex_anchor_notes": "缺 local_path，无法读取原文页。",
        }

    try:
        page_texts = _page_texts_for_path(local_path=str(local_path), page_cache=page_cache)
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            **base,
            "codex_anchor_status": "codex_anchor_load_error",
            "codex_anchor_pages": [],
            "codex_anchor_field_hits": [],
            "codex_anchor_missing_fields": list(row.get("expected_answer_field_ids", [])),
            "codex_anchor_snippets": [],
            "codex_anchor_notes": f"读取原文失败：{type(exc).__name__}: {exc}",
        }

    field_hits = _field_hits(row=row, page_texts=page_texts, candidate_pages=candidate_pages)
    missing_fields = [
        field
        for field in row.get("expected_answer_field_ids", [])
        if field not in {hit["field_id"] for hit in field_hits}
    ]
    anchor_pages = _anchor_pages(
        row=row,
        page_texts=page_texts,
        candidate_pages=candidate_pages,
        field_hits=field_hits,
    )
    snippets = _anchor_snippets(
        row=row,
        page_texts=page_texts,
        pages=anchor_pages or candidate_pages[:1],
        field_hits=field_hits,
    )
    status = _anchor_status(
        anchor_pages=anchor_pages,
        field_hits=field_hits,
        missing_fields=missing_fields,
    )
    return {
        **base,
        "codex_anchor_status": status,
        "codex_anchor_pages": anchor_pages,
        "codex_anchor_field_hits": field_hits,
        "codex_anchor_missing_fields": missing_fields,
        "codex_anchor_snippets": snippets,
        "codex_anchor_notes": _anchor_notes(status=status, missing_fields=missing_fields),
    }


def _base_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "query_id": row["query_id"],
        "company": row["company"],
        "industry": row["industry"],
        "document_key": row.get("document_key"),
        "local_path": row.get("local_path"),
        "query": row["query"],
        "primary_evidence_kind": row["primary_evidence_kind"],
        "secondary_evidence_kinds": list(row.get("secondary_evidence_kinds", [])),
        "candidate_pages": list(row.get("candidate_pages", [])),
        "candidate_snippets": list(row.get("candidate_snippets", [])),
        "evidence_search_score": row.get("evidence_search_score", 0),
        "auto_anchor_status": row.get("auto_anchor_status", ""),
        "review_priority": row.get("review_priority", ""),
        "codex_probe_status": row.get("codex_probe_status", "not_requested"),
        "codex_probe_pages": list(row.get("codex_probe_pages", [])),
        "expected_answer_field_ids": list(row.get("expected_answer_field_ids", [])),
        "forbidden_failure_modes": list(row.get("forbidden_failure_modes", [])),
        "manifest_readiness": row.get("manifest_readiness", ""),
        "ready_for_manifest": False,
        "page_number_policy": PAGE_NUMBER_POLICY,
        "anchor_review_status": row.get("anchor_review_status", "not_reviewed"),
        "user_review_confirmed_pages": list(row.get("human_confirmed_pages", [])),
        "user_review_corrected_pages": list(row.get("human_corrected_pages", [])),
        "user_review_missing_fields": list(row.get("human_missing_fields", [])),
        "user_review_notes": row.get("human_review_notes", ""),
    }


def _page_texts_for_path(
    *,
    local_path: str,
    page_cache: dict[str, dict[int, str]],
) -> dict[int, str]:
    source_path = _resolve(Path(local_path))
    cache_key = str(source_path)
    if cache_key not in page_cache:
        page_cache[cache_key] = dict(_load_pages(source_path))
    return page_cache[cache_key]


def _field_hits(
    *,
    row: dict[str, Any],
    page_texts: dict[int, str],
    candidate_pages: list[int],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for field_id in row.get("expected_answer_field_ids", []):
        terms = _field_terms(str(field_id))
        page_hits: list[int] = []
        matched_terms: set[str] = set()
        snippets: list[str] = []
        for page in candidate_pages:
            text = page_texts.get(page, "")
            page_matched_terms = _matched_terms(text, terms)
            if not page_matched_terms:
                continue
            page_hits.append(page)
            matched_terms.update(page_matched_terms)
            for snippet in _snippets(text=text, matched_terms=page_matched_terms):
                if snippet not in snippets:
                    snippets.append(snippet)
        if page_hits:
            hits.append(
                {
                    "field_id": field_id,
                    "pages": page_hits,
                    "matched_terms": sorted(matched_terms, key=_normalize),
                    "snippets": snippets[:2],
                }
            )
    return hits


def _anchor_pages(
    *,
    row: dict[str, Any],
    page_texts: dict[int, str],
    candidate_pages: list[int],
    field_hits: list[dict[str, Any]],
) -> list[int]:
    field_pages = {page for hit in field_hits for page in hit["pages"]}
    signal_terms = _signal_terms(row)
    signal_pages = {
        page
        for page in candidate_pages
        if _matched_terms(page_texts.get(page, ""), signal_terms)
    }
    anchor_pages = [page for page in candidate_pages if page in field_pages or page in signal_pages]
    return anchor_pages[:5]


def _anchor_snippets(
    *,
    row: dict[str, Any],
    page_texts: dict[int, str],
    pages: list[int],
    field_hits: list[dict[str, Any]],
) -> list[str]:
    snippets: list[str] = []
    for hit in field_hits:
        for snippet in hit["snippets"]:
            if snippet not in snippets:
                snippets.append(snippet)
    if len(snippets) >= 3:
        return snippets[:3]

    terms = _signal_terms(row)
    for page in pages:
        text = page_texts.get(page, "")
        matched = _matched_terms(text, terms)
        for snippet in _snippets(text=text, matched_terms=matched):
            if snippet not in snippets:
                snippets.append(snippet)
            if len(snippets) >= 3:
                return snippets[:3]
    return snippets[:3]


def _anchor_status(
    *,
    anchor_pages: list[int],
    field_hits: list[dict[str, Any]],
    missing_fields: list[str],
) -> str:
    if not anchor_pages:
        return "codex_anchor_unresolved_candidate"
    if not missing_fields:
        return "codex_anchor_confirmed_candidate"
    if field_hits:
        return "codex_anchor_partial_fields"
    return "codex_anchor_unresolved_candidate"


def _field_terms(field_id: str) -> list[str]:
    terms: list[str] = []
    terms.extend(FIELD_TERMS.get(field_id, ()))
    terms.extend(FIELD_TERM_OVERRIDES.get(field_id, ()))
    terms.extend(_field_id_terms(field_id))
    return _dedupe_terms(terms)


def _signal_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    terms.extend(str(term) for term in row.get("matched_terms", []) if term)
    terms.extend(str(term) for term in row.get("codex_probe_matched_terms", []) if term)
    for field_id in row.get("expected_answer_field_ids", []):
        terms.extend(_field_terms(str(field_id)))
    return _dedupe_terms(terms)


def _dedupe_terms(terms: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if len(cleaned) < 2:
            continue
        key = _normalize(cleaned)
        if key and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


def _no_hit_notes(row: dict[str, Any]) -> str:
    if row.get("query_id") == "LOCAL-02":
        return "当前美团 PDF 文本抽取存在乱码，关键词不可稳定命中；继续 defer，不要求用户翻 PDF。"
    return "没有候选页；保留为 defer，不写 expected_pages。"


def _anchor_notes(*, status: str, missing_fields: list[str]) -> str:
    if status == "codex_anchor_confirmed_candidate":
        return "候选页覆盖必需字段；仍只是 Codex 草案，不能直接升格为 expected_pages。"
    if status == "codex_anchor_partial_fields":
        return (
            "候选页命中部分字段，缺口："
            f"{', '.join(missing_fields)}；后续 manifest 前仍需确认。"
        )
    if status == "codex_anchor_unresolved_candidate":
        return "候选页存在，但字段信号不集中；后续不交给用户全量翻 PDF，先保留为草案缺口。"
    return "当前无法确认 anchor。"


def _has_review_caveat(row: dict[str, Any]) -> bool:
    return bool(
        row.get("human_corrected_pages")
        or row.get("human_missing_fields")
        or row.get("human_rejected_candidate_pages")
    )


def _review_caveat(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "company": row["company"],
        "query_id": row["query_id"],
        "review_status": row.get("anchor_review_status", "not_reviewed"),
        "confirmed_pages": list(row.get("human_confirmed_pages", [])),
        "corrected_pages": list(row.get("human_corrected_pages", [])),
        "rejected_candidate_pages": list(row.get("human_rejected_candidate_pages", [])),
        "missing_fields": list(row.get("human_missing_fields", [])),
        "notes": row.get("human_review_notes", ""),
    }


def _summarize(*, cases: list[dict[str, Any]], review_packet_cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_rows": len(cases),
        "candidate_rows": sum(bool(case["candidate_pages"]) for case in cases),
        "no_candidate_rows": sum(not case["candidate_pages"] for case in cases),
        "codex_anchor_confirmed_rows": sum(
            case["codex_anchor_status"] == "codex_anchor_confirmed_candidate"
            for case in cases
        ),
        "codex_anchor_partial_rows": sum(
            case["codex_anchor_status"] == "codex_anchor_partial_fields" for case in cases
        ),
        "codex_anchor_unresolved_rows": sum(
            case["codex_anchor_status"] == "codex_anchor_unresolved_candidate"
            for case in cases
        ),
        "codex_anchor_deferred_rows": sum(
            case["codex_anchor_status"] == "codex_anchor_deferred_no_hit" for case in cases
        ),
        "codex_anchor_load_error_rows": sum(
            case["codex_anchor_status"] == "codex_anchor_load_error" for case in cases
        ),
        "review_packet_rows": len(review_packet_cases),
        "review_packet_pending_rows": sum(
            case["anchor_review_status"] == "not_reviewed"
            and case["codex_anchor_status"] != "codex_anchor_deferred_no_hit"
            for case in review_packet_cases
        ),
        "review_packet_reviewed_rows": sum(
            case["anchor_review_status"] != "not_reviewed" for case in review_packet_cases
        ),
        "review_packet_deferred_rows": sum(
            case["codex_anchor_status"] == "codex_anchor_deferred_no_hit"
            for case in review_packet_cases
        ),
        "ready_for_manifest": sum(bool(case["ready_for_manifest"]) for case in cases),
    }


def render_review_packet(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Golden Queries v2 Anchor Review Packet",
        "",
        f"更新时间：{report['generated_at']}",
        "",
        "这是给用户抽查的精简包，不是 runnable manifest。",
        "",
        "## 摘要",
        "",
        f"- 总行业专项行数：`{summary['total_rows']}`",
        f"- 有候选页行数：`{summary['candidate_rows']}`",
        f"- 无候选页 / defer 行数：`{summary['no_candidate_rows']}`",
        f"- Codex 字段全覆盖草案：`{summary['codex_anchor_confirmed_rows']}`",
        f"- Codex 字段部分覆盖草案：`{summary['codex_anchor_partial_rows']}`",
        f"- 抽查包总行数：`{summary['review_packet_rows']}`",
        f"- 已收到用户反馈：`{summary['review_packet_reviewed_rows']}`",
        f"- 仍需用户抽查：`{summary['review_packet_pending_rows']}`",
        f"- defer / 不要求用户翻 PDF：`{summary['review_packet_deferred_rows']}`",
        f"- ready_for_manifest：`{summary['ready_for_manifest']}`",
        "",
        "## 口径",
        "",
        "- `codex_anchor_pages` 是 Codex 原文核对草案，不等于 `expected_pages`。",
        "- 本阶段不写 `expected_pages`，不生成 runnable manifest。",
        "- `LOCAL-02` 保持 no-hit/defer，不要求用户翻 PDF。",
    ]

    packet_cases = report["review_packet"]["cases"]
    pending_cases = [
        case
        for case in packet_cases
        if case["anchor_review_status"] == "not_reviewed"
        and case["codex_anchor_status"] != "codex_anchor_deferred_no_hit"
    ]
    reviewed_cases = [
        case for case in packet_cases if case["anchor_review_status"] != "not_reviewed"
    ]
    deferred_cases = [
        case
        for case in packet_cases
        if case["codex_anchor_status"] == "codex_anchor_deferred_no_hit"
    ]

    if pending_cases:
        lines.extend(
            [
                "",
                "## 需要用户抽查",
                "",
                "| 公司 | Query ID | 问题 | 状态 | 候选页 | Codex 页 | 缺字段 | 片段 / 备注 |",
                "|---|---:|---|---|---|---|---|---|",
            ]
        )
    for case in pending_cases:
        lines.append(_review_case_row(case))

    if reviewed_cases:
        lines.extend(
            [
                "",
                "## 已收到用户反馈",
                "",
                "| 公司 | Query ID | 问题 | 用户状态 | 用户确认页 | 用户修正页 | 字段缺口 | 备注 |",
                "|---|---:|---|---|---|---|---|---|",
            ]
        )
        for case in reviewed_cases:
            lines.append(_reviewed_case_row(case))

    if deferred_cases:
        lines.extend(
            [
                "",
                "## Defer / 不要求用户翻 PDF",
                "",
                "| 公司 | Query ID | 问题 | 状态 | 说明 |",
                "|---|---:|---|---|---|",
            ]
        )
        for case in deferred_cases:
            lines.append(_deferred_case_row(case))

    caveats = report["review_packet"].get("known_caveats", [])
    if caveats:
        lines.extend(
            [
                "",
                "## 其他已知 Caveat",
                "",
                "| 公司 | Query ID | 状态 | 修正页 | 字段缺口 | 备注 |",
                "|---|---:|---|---|---|---|",
            ]
        )
        for caveat in caveats:
            lines.append(_caveat_row(caveat))
    lines.append("")
    return "\n".join(lines)


def _review_case_row(case: dict[str, Any]) -> str:
    candidate_pages = _join_pages(case.get("candidate_pages", []))
    anchor_pages = _join_pages(case.get("codex_anchor_pages", []))
    missing = ", ".join(case.get("codex_anchor_missing_fields", [])) or "-"
    detail = " / ".join(case.get("codex_anchor_snippets", [])[:2])
    if not detail:
        detail = case.get("codex_anchor_notes", "-")
    else:
        detail = f"{detail} / {case.get('codex_anchor_notes', '-')}"
    return (
        f"| {_esc(case['company'])} | `{case['query_id']}` | "
        f"{_esc(case['query'])} | `{case['codex_anchor_status']}` | "
        f"{candidate_pages} | {anchor_pages} | "
        f"{_esc(missing)} | {_esc(detail)} |"
    )


def _caveat_row(caveat: dict[str, Any]) -> str:
    corrected = _join_pages(caveat.get("corrected_pages", []))
    missing = ", ".join(caveat.get("missing_fields", [])) or "-"
    return (
        f"| {_esc(caveat['company'])} | `{caveat['query_id']}` | "
        f"`{caveat['review_status']}` | {corrected} | {_esc(missing)} | "
        f"{_esc(caveat.get('notes', '-'))} |"
    )


def _reviewed_case_row(case: dict[str, Any]) -> str:
    confirmed = _join_pages(case.get("user_review_confirmed_pages", []))
    corrected = _join_pages(case.get("user_review_corrected_pages", []))
    missing = ", ".join(case.get("user_review_missing_fields", [])) or "-"
    notes = case.get("user_review_notes") or "-"
    return (
        f"| {_esc(case['company'])} | `{case['query_id']}` | {_esc(case['query'])} | "
        f"`{case['anchor_review_status']}` | {confirmed} | {corrected} | "
        f"{_esc(missing)} | {_esc(notes)} |"
    )


def _deferred_case_row(case: dict[str, Any]) -> str:
    return (
        f"| {_esc(case['company'])} | `{case['query_id']}` | {_esc(case['query'])} | "
        f"`{case['codex_anchor_status']}` | {_esc(case['codex_anchor_notes'])} |"
    )


def _join_pages(pages: object) -> str:
    if not isinstance(pages, list):
        return "-"
    text = ", ".join(str(page) for page in pages if isinstance(page, int))
    return text or "-"


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
