from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from filingdelta.core.config import REPO_ROOT


DEFAULT_CANDIDATE_MATRIX = Path("data/outputs/eval/golden_queries_v2_candidate_matrix.json")
DEFAULT_ANCHOR_PROBE = Path("data/outputs/eval/golden_queries_v2_anchor_probe.json")
DEFAULT_EVIDENCE_SEARCH = Path("data/outputs/eval/golden_queries_v2_evidence_search.json")
DEFAULT_ANCHOR_REVIEW_NOTES = Path("data/outputs/eval/golden_queries_v2_anchor_review_notes.json")
DEFAULT_JSON_OUTPUT = Path("data/outputs/eval/golden_queries_v2_industry_evidence_matrix.json")
DEFAULT_MD_OUTPUT = Path("docs/golden_queries_v2_industry_evidence_matrix.md")

Route = Literal["document_only", "concept_only", "mixed", "unsupported"]
Intent = Literal["metric_value", "metric_attribution", "business_narrative", "fallback"]
EvidenceKind = Literal["table_row", "section_text", "page_text"]


@dataclass(frozen=True)
class IndustryQueryDefinition:
    query_id: str
    industry: str
    query: str
    expected_route: Route
    expected_document_evidence_intent: Intent
    evidence_kinds: tuple[EvidenceKind, ...]
    area: str
    expected_answer_field_ids: tuple[str, ...]
    expected_answer_fields_label: str
    forbidden_failure_modes: tuple[str, ...]


QUERY_DEFINITIONS: dict[str, IndustryQueryDefinition] = {
    "HA-01": IndustryQueryDefinition(
        query_id="HA-01",
        industry="home_appliances",
        query="美的集团智能家居和商业及工业解决方案收入分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="segment revenue",
        expected_answer_field_ids=("segment_names", "segment_revenue", "yoy_or_share"),
        expected_answer_fields_label="segment names, revenue, YoY/share if disclosed",
        forbidden_failure_modes=("product row treated as segment row without caveat",),
    ),
    "HA-02": IndustryQueryDefinition(
        query_id="HA-02",
        industry="home_appliances",
        query="海尔智家国内和海外业务增长分别由哪些因素驱动？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("section_text",),
        area="business/geographic review",
        expected_answer_field_ids=(
            "china_drivers",
            "overseas_drivers",
            "product_or_channel_examples",
        ),
        expected_answer_fields_label="China drivers, overseas drivers, product/channel examples",
        forbidden_failure_modes=("macro causes invented outside MD&A",),
    ),
    "HA-03": IndustryQueryDefinition(
        query_id="HA-03",
        industry="home_appliances",
        query="家电企业存货或渠道库存是否异常？公司如何描述渠道效率？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("table_row", "section_text"),
        area="inventory, channel review",
        expected_answer_field_ids=("inventory_value_or_turnover", "channel_explanation"),
        expected_answer_fields_label="inventory value/turnover if disclosed, channel explanation",
        forbidden_failure_modes=("all inventory increases described as unsold goods",),
    ),
    "SHIP-01": IndustryQueryDefinition(
        query_id="SHIP-01",
        industry="shipping",
        query="中远海控集装箱航运业务货运量和收入分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="container shipping table",
        expected_answer_field_ids=("teu_volume", "revenue", "yoy", "unit"),
        expected_answer_fields_label="TEU volume, revenue, YoY, unit",
        forbidden_failure_modes=("terminal throughput confused with shipping volume",),
    ),
    "SHIP-02": IndustryQueryDefinition(
        query_id="SHIP-02",
        industry="shipping",
        query="中远海控利润增长主要由哪些因素驱动？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("section_text", "table_row"),
        area="MD&A, industry trend",
        expected_answer_field_ids=(
            "volume_driver",
            "freight_rate_or_capacity_driver",
            "route_disruption_if_disclosed",
        ),
        expected_answer_fields_label=(
            "volume, freight rate/effective capacity, route disruption if disclosed"
        ),
        forbidden_failure_modes=('"demand growth" only, no rate/capacity driver',),
    ),
    "SHIP-03": IndustryQueryDefinition(
        query_id="SHIP-03",
        industry="shipping",
        query="红海局势在年报中如何影响航运供需和运价？",
        expected_route="document_only",
        expected_document_evidence_intent="business_narrative",
        evidence_kinds=("section_text",),
        area="industry trend, outlook/risk",
        expected_answer_field_ids=("red_sea_wording", "capacity_or_route_effect", "rate_effect"),
        expected_answer_fields_label="explicit Red Sea wording, capacity/route/rate effect",
        forbidden_failure_modes=("geopolitical speculation beyond report",),
    ),
    "OIL-01": IndustryQueryDefinition(
        query_id="OIL-01",
        industry="energy_oil_gas",
        query="中国海油油气净产量、净证实储量和储量寿命分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="reserves and production",
        expected_answer_field_ids=("production", "proved_reserves", "reserve_life", "unit"),
        expected_answer_fields_label="production, proved reserves, reserve life, unit",
        forbidden_failure_modes=("barrels/BOE/cubic feet mixed without label",),
    ),
    "OIL-02": IndustryQueryDefinition(
        query_id="OIL-02",
        industry="energy_oil_gas",
        query="中国海油资本开支是多少？主要投向哪些环节？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row", "section_text"),
        area="capex, business review",
        expected_answer_field_ids=("total_capex", "capex_split_or_use"),
        expected_answer_fields_label="total capex, split/use if disclosed",
        forbidden_failure_modes=("planned capex used as actual without label",),
    ),
    "PTR-01": IndustryQueryDefinition(
        query_id="PTR-01",
        industry="energy_oil_gas",
        query="中国石油油气当量产量、天然气产量和炼油加工量分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="operating statistics",
        expected_answer_field_ids=(
            "oil_gas_equivalent_output",
            "gas_output",
            "refining_throughput",
            "unit",
        ),
        expected_answer_fields_label="output, gas, refining throughput, units",
        forbidden_failure_modes=("future target used as historical actual",),
    ),
    "COAL-01": IndustryQueryDefinition(
        query_id="COAL-01",
        industry="energy_coal",
        query="中国神华商品煤产量、煤炭销售量和平均售价是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="coal production/sales table",
        expected_answer_field_ids=("commercial_coal_production", "coal_sales_volume", "asp", "yoy"),
        expected_answer_fields_label="production, sales, ASP, YoY",
        forbidden_failure_modes=("self-produced coal confused with total sales",),
    ),
    "COAL-02": IndustryQueryDefinition(
        query_id="COAL-02",
        industry="energy_coal",
        query="中国神华一体化运营模式包括哪些环节？它如何影响抗周期能力？",
        expected_route="document_only",
        expected_document_evidence_intent="business_narrative",
        evidence_kinds=("section_text",),
        area="business overview, MD&A",
        expected_answer_field_ids=(
            "coal",
            "power",
            "railway",
            "port",
            "shipping",
            "chemical",
            "synergy_explanation",
        ),
        expected_answer_fields_label=(
            "coal, power, railway, port, shipping, chemical; synergies"
        ),
        forbidden_failure_modes=("says integration fully eliminates commodity risk",),
    ),
    "HYDRO-01": IndustryQueryDefinition(
        query_id="HYDRO-01",
        industry="hydropower",
        query="长江电力发电量、上网电量、售电量和上网电价分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="electricity volume/tariff table",
        expected_answer_field_ids=("generation", "on_grid_volume", "sales_volume", "tariff"),
        expected_answer_fields_label="generation, on-grid volume, sales volume, tariff",
        forbidden_failure_modes=("万千瓦时 and 亿千瓦时 confused",),
    ),
    "HYDRO-02": IndustryQueryDefinition(
        query_id="HYDRO-02",
        industry="hydropower",
        query="长江电力业绩增长与来水和发电量有什么关系？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("section_text", "table_row"),
        area="hydrology, business review",
        expected_answer_field_ids=("inflow_or_generation_explanation", "profit_linkage"),
        expected_answer_fields_label="inflow/generation explanation, profit linkage if stated",
        forbidden_failure_modes=("tariff rise invented when not disclosed",),
    ),
    "HYDRO-03": IndustryQueryDefinition(
        query_id="HYDRO-03",
        industry="hydropower",
        query="公司如何描述抽蓄、新能源和储能布局？",
        expected_route="document_only",
        expected_document_evidence_intent="business_narrative",
        evidence_kinds=("section_text",),
        area="strategy, business review",
        expected_answer_field_ids=("pumped_storage", "new_energy", "energy_storage", "examples"),
        expected_answer_fields_label=(
            "pumped storage, water-wind-solar integration, project examples"
        ),
        forbidden_failure_modes=("project mentions converted into revenue contribution",),
    ),
    "BAIJIU-01": IndustryQueryDefinition(
        query_id="BAIJIU-01",
        industry="baijiu",
        query="贵州茅台茅台酒和系列酒收入分别是多少？哪个增长更快？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="product revenue",
        expected_answer_field_ids=("moutai_liquor_revenue", "series_liquor_revenue", "growth"),
        expected_answer_fields_label="product revenue, YoY, share",
        forbidden_failure_modes=("approximate narrative used when table has exact value",),
    ),
    "BAIJIU-02": IndustryQueryDefinition(
        query_id="BAIJIU-02",
        industry="baijiu",
        query="贵州茅台直销和批发代理渠道收入结构如何变化？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="channel revenue",
        expected_answer_field_ids=("direct_revenue", "wholesale_revenue", "share", "yoy"),
        expected_answer_fields_label="direct revenue, wholesale revenue, share, YoY",
        forbidden_failure_modes=("ecommerce commentary treated as direct-sales number",),
    ),
    "MEDIA-01": IndustryQueryDefinition(
        query_id="MEDIA-01",
        industry="advertising_media",
        query="分众传媒楼宇媒体收入占主营业务收入的比例是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="product/segment revenue",
        expected_answer_field_ids=("building_media_revenue", "share", "other_media"),
        expected_answer_fields_label="building media revenue, share, other media",
        forbidden_failure_modes=("product description used as numeric share",),
    ),
    "MEDIA-02": IndustryQueryDefinition(
        query_id="MEDIA-02",
        industry="advertising_media",
        query="分众传媒生活圈媒体网络覆盖哪些城市和设备类型？",
        expected_route="document_only",
        expected_document_evidence_intent="business_narrative",
        evidence_kinds=("section_text", "table_row"),
        area="business overview",
        expected_answer_field_ids=("city_coverage", "device_types", "overseas_coverage"),
        expected_answer_fields_label="city coverage, elevator TV/poster devices, overseas coverage",
        forbidden_failure_modes=("interim data mixed with annual without label",),
    ),
    "INS-01": IndustryQueryDefinition(
        query_id="INS-01",
        industry="insurance",
        query="中国平安归母营运利润、归母净利润和营业收入分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="financial highlights",
        expected_answer_field_ids=("operating_profit", "net_profit", "revenue", "yoy"),
        expected_answer_fields_label="operating profit, net profit, revenue, YoY",
        forbidden_failure_modes=("operating profit confused with IFRS operating income",),
    ),
    "INS-02": IndustryQueryDefinition(
        query_id="INS-02",
        industry="insurance",
        query="平安寿险及健康险新业务价值 NBV 增长了多少？主要渠道贡献如何？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("table_row", "section_text"),
        area="life and health review",
        expected_answer_field_ids=("nbv", "yoy", "channel_contribution"),
        expected_answer_fields_label="NBV, YoY, channel contribution",
        forbidden_failure_modes=("NBV confused with premium income",),
    ),
    "INS-03": IndustryQueryDefinition(
        query_id="INS-03",
        industry="insurance",
        query="平安产险综合成本率是多少？年报如何解释业务质量变化？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("table_row", "section_text"),
        area="P&C review",
        expected_answer_field_ids=("combined_ratio", "yoy_or_pps", "quality_explanation"),
        expected_answer_fields_label="combined ratio, YoY/pps, underwriting/risk explanation",
        forbidden_failure_modes=("loss ratio used as combined ratio",),
    ),
    "NEV-01": IndustryQueryDefinition(
        query_id="NEV-01",
        industry="new_energy_vehicle",
        query="比亚迪汽车业务和手机部件及组装业务收入分别是多少？占比如何？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="product/segment revenue",
        expected_answer_field_ids=("auto_revenue", "mobile_components_revenue", "share"),
        expected_answer_fields_label="segment revenue, share, YoY",
        forbidden_failure_modes=("BYD Electronic standalone data treated as group segment",),
    ),
    "NEV-02": IndustryQueryDefinition(
        query_id="NEV-02",
        industry="new_energy_vehicle",
        query="比亚迪新能源汽车销量是多少？销量增长如何影响收入？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("table_row", "section_text"),
        area="sales volume, business review",
        expected_answer_field_ids=("sales_volume", "yoy", "management_attribution"),
        expected_answer_fields_label="sales volume, YoY, management attribution",
        forbidden_failure_modes=("production volume used when sales volume is asked",),
    ),
    "BAT-01": IndustryQueryDefinition(
        query_id="BAT-01",
        industry="battery",
        query="宁德时代动力电池和储能电池收入分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="product revenue",
        expected_answer_field_ids=("ev_battery_revenue", "ess_revenue", "share", "yoy"),
        expected_answer_fields_label="EV battery revenue, ESS revenue, share, YoY",
        forbidden_failure_modes=("H1 data used for annual query",),
    ),
    "BAT-02": IndustryQueryDefinition(
        query_id="BAT-02",
        industry="battery",
        query="宁德时代为什么出现收入下降但净利润增长？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("section_text", "table_row"),
        area="MD&A, gross margin, raw materials",
        expected_answer_field_ids=("revenue_change", "profit_change", "margin_or_cost_explanation"),
        expected_answer_fields_label="revenue change, profit change, margin/cost explanation",
        forbidden_failure_modes=("demand collapse invented without filing evidence",),
    ),
    "OTA-01": IndustryQueryDefinition(
        query_id="OTA-01",
        industry="ota",
        query="携程住宿预订、交通票务、旅游度假和商旅收入分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="revenue by service",
        expected_answer_field_ids=(
            "accommodation_revenue",
            "transportation_revenue",
            "packaged_tour_revenue",
            "corporate_travel_revenue",
        ),
        expected_answer_fields_label="four service revenues, YoY, share",
        forbidden_failure_modes=("Q4 numbers mixed with full-year numbers",),
    ),
    "OTA-02": IndustryQueryDefinition(
        query_id="OTA-02",
        industry="ota",
        query="携程国际业务增长有哪些披露？",
        expected_route="document_only",
        expected_document_evidence_intent="business_narrative",
        evidence_kinds=("section_text",),
        area="business updates",
        expected_answer_field_ids=("international_ota", "outbound_or_inbound_indicators"),
        expected_answer_fields_label="international OTA, outbound/inbound indicators",
        forbidden_failure_modes=("global travel trend cited without company disclosure",),
    ),
    "BABA-01": IndustryQueryDefinition(
        query_id="BABA-01",
        industry="ecommerce_cloud",
        query="阿里巴巴各业务分部收入分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="segment revenue",
        expected_answer_field_ids=(
            "taobao_tmall_revenue",
            "cloud_revenue",
            "aidc_revenue",
            "cainiao_revenue",
            "local_services_revenue",
        ),
        expected_answer_fields_label=(
            "Taobao/Tmall, Cloud, AIDC, Cainiao, Local Services, etc."
        ),
        forbidden_failure_modes=("segment reclassification ignored",),
    ),
    "BABA-02": IndustryQueryDefinition(
        query_id="BABA-02",
        industry="ecommerce_cloud",
        query="阿里收入按类型如何构成？客户管理、云服务、物流服务分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="revenue by type notes",
        expected_answer_field_ids=(
            "customer_management_revenue",
            "cloud_service_revenue",
            "logistics_service_revenue",
            "goods_sales_revenue",
        ),
        expected_answer_fields_label="customer management, cloud, logistics, goods sales",
        forbidden_failure_modes=("segment revenue confused with revenue type",),
    ),
    "LOCAL-01": IndustryQueryDefinition(
        query_id="LOCAL-01",
        industry="local_services",
        query="美团核心本地商业和新业务收入分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="segment revenue",
        expected_answer_field_ids=("core_local_commerce_revenue", "new_initiatives_revenue", "yoy"),
        expected_answer_fields_label="core local commerce revenue, new initiatives revenue, total, YoY",
        forbidden_failure_modes=("old segment names used without mapping",),
    ),
    "LOCAL-02": IndustryQueryDefinition(
        query_id="LOCAL-02",
        industry="local_services",
        query="美团新业务亏损为什么收窄？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_attribution",
        evidence_kinds=("section_text", "table_row"),
        area="MD&A, segment profit/loss",
        expected_answer_field_ids=("operating_loss", "management_stated_changes"),
        expected_answer_fields_label="operating loss amount, management-stated changes",
        forbidden_failure_modes=("loss narrowing described as segment profit",),
    ),
    "IP-01": IndustryQueryDefinition(
        query_id="IP-01",
        industry="ip_retail",
        query="泡泡玛特自有产品和艺术家 IP 收入占比是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="revenue by product/IP",
        expected_answer_field_ids=("proprietary_products", "artist_ips", "revenue", "share"),
        expected_answer_fields_label="proprietary products, artist IPs, revenue, share",
        forbidden_failure_modes=("current Labubu news used for historical filing query",),
    ),
    "IP-02": IndustryQueryDefinition(
        query_id="IP-02",
        industry="ip_retail",
        query="THE MONSTERS、MOLLY、SKULLPANDA 等 IP 收入排名如何？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="IP revenue table",
        expected_answer_field_ids=("ip_names", "revenue", "share", "ranking"),
        expected_answer_fields_label="IP names, revenue, share, ranking",
        forbidden_failure_modes=("IP names translated/merged inconsistently",),
    ),
    "SPORTS-01": IndustryQueryDefinition(
        query_id="SPORTS-01",
        industry="sportswear",
        query="安踏体育 ANTA、FILA 和其他品牌收入分别是多少？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="brand/segment revenue",
        expected_answer_field_ids=("anta_revenue", "fila_revenue", "other_brands_revenue"),
        expected_answer_fields_label="brand revenue, YoY, operating margin if disclosed",
        forbidden_failure_modes=("Amer Sports JV treated as consolidated brand revenue",),
    ),
    "SPORTS-02": IndustryQueryDefinition(
        query_id="SPORTS-02",
        industry="sportswear",
        query="安踏电商收入占比和库存周转天数如何变化？",
        expected_route="document_only",
        expected_document_evidence_intent="metric_value",
        evidence_kinds=("table_row",),
        area="operating metrics",
        expected_answer_field_ids=("ecommerce_contribution", "inventory_turnover_days", "change"),
        expected_answer_fields_label="ecommerce contribution, inventory turnover days, YoY/change",
        forbidden_failure_modes=("inventory balance confused with turnover days",),
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the query-level industry evidence matrix for golden_queries_v2."
    )
    parser.add_argument("--candidate-matrix", type=Path, default=DEFAULT_CANDIDATE_MATRIX)
    parser.add_argument("--anchor-probe", type=Path, default=DEFAULT_ANCHOR_PROBE)
    parser.add_argument("--evidence-search", type=Path, default=DEFAULT_EVIDENCE_SEARCH)
    parser.add_argument("--anchor-review-notes", type=Path, default=DEFAULT_ANCHOR_REVIEW_NOTES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args(argv)

    report = build_matrix_report(
        candidate_matrix_path=_resolve(args.candidate_matrix),
        anchor_probe_path=_resolve(args.anchor_probe),
        evidence_search_path=_resolve(args.evidence_search),
        anchor_review_notes_path=_resolve(args.anchor_review_notes),
    )
    json_output = _resolve(args.json_output)
    md_output = _resolve(args.md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_output.write_text(_render_markdown(report), encoding="utf-8")

    summary = report["summary"]
    print(
        "industry_matrix "
        f"rows={summary['total_rows']} "
        f"probe_hits={summary['anchor_probe_hit']} "
        f"auto_high={summary['auto_anchor_high_confidence']} "
        f"auto_low={summary['auto_anchor_low_confidence']} "
        f"manual_probe={summary['auto_needs_manual_probe']} "
        f"review_rows={summary['review_rows']} "
        f"blocked={summary['blocked_missing_raw']} "
        f"json={json_output} md={md_output}"
    )
    return 0


def build_matrix_report(
    *,
    candidate_matrix_path: Path,
    anchor_probe_path: Path,
    evidence_search_path: Path,
    anchor_review_notes_path: Path | None = None,
) -> dict[str, Any]:
    candidate_matrix = _load_json(candidate_matrix_path)
    anchor_probe = _load_json(anchor_probe_path)
    probe_by_case_id = {case["case_id"]: case for case in anchor_probe.get("cases", [])}
    search_by_case_id = _load_search_cases(evidence_search_path)
    review_by_key = _load_anchor_review_notes(anchor_review_notes_path)

    rows = _build_primary_rows(candidate_matrix, probe_by_case_id, search_by_case_id)
    rows.extend(_build_blocked_rows(candidate_matrix))
    _apply_anchor_review_notes(rows, review_by_key)
    _assign_review_priorities(rows)
    rows.sort(key=lambda row: (row["status_sort"], row["company"] or "", row["query_id"]))
    for row in rows:
        row.pop("status_sort", None)

    summary = {
        "total_rows": len(rows),
        "primary_candidate_rows": sum(row["matrix_status"] != "blocked_missing_raw" for row in rows),
        "blocked_missing_raw": sum(row["matrix_status"] == "blocked_missing_raw" for row in rows),
        "anchor_probe_hit": sum(row["anchor_status"] == "anchor_probe_hit" for row in rows),
        "needs_manual_probe": sum(row["anchor_status"] == "anchor_probe_no_hit" for row in rows),
        "auto_anchor_high_confidence": sum(
            row["auto_anchor_status"] == "auto_anchor_high_confidence" for row in rows
        ),
        "auto_anchor_low_confidence": sum(
            row["auto_anchor_status"] == "auto_anchor_low_confidence" for row in rows
        ),
        "auto_needs_manual_probe": sum(
            row["auto_anchor_status"] == "needs_manual_probe" for row in rows
        ),
        "codex_probe_rows": sum(row["review_priority"] == "codex_probe" for row in rows),
        "codex_probe_candidate_rows": sum(
            row["codex_probe_status"] == "codex_probe_candidate_found" for row in rows
        ),
        "codex_probe_no_hit_rows": sum(
            row["codex_probe_status"] == "codex_probe_no_hit_after_deep_search"
            for row in rows
        ),
        "review_rows": sum(_requires_user_priority_review(row) for row in rows),
        "human_reviewed_rows": sum(row["anchor_review_status"] != "not_reviewed" for row in rows),
        "human_corrected_rows": sum(
            row["anchor_review_status"] == "human_corrected_page" for row in rows
        ),
        "human_partial_field_gap_rows": sum(
            row["anchor_review_status"] == "human_partial_field_gap" for row in rows
        ),
        "ready_for_manifest": sum(row["manifest_readiness"] == "ready_for_manifest" for row in rows),
        "needs_anchor_confirmation": sum(
            row["manifest_readiness"] == "needs_anchor_confirmation" for row in rows
        ),
    }
    return {
        "schema_version": "golden_queries_v2_industry_evidence_matrix.v1",
        "generated_at": date.today().isoformat(),
        "source_files": {
            "candidate_matrix": _display_path(candidate_matrix_path),
            "anchor_probe": _display_path(anchor_probe_path),
            "evidence_search": _display_path(evidence_search_path),
            "anchor_review_notes": (
                _display_path(anchor_review_notes_path)
                if anchor_review_notes_path is not None and anchor_review_notes_path.exists()
                else None
            ),
            "design_doc": "docs/golden_queries_v2_design.md",
        },
        "policy": {
            "not_a_runnable_manifest": True,
            "page_number_policy": (
                "candidate_pages 来自 PyMuPDF/runtime 1-based page_number，不是 PDF 印刷页码。"
            ),
            "anchor_policy": (
                "candidate_pages 只是候选页；expected_pages 必须在 evidence-location pass "
                "确认后才能写入 runnable manifest。"
            ),
            "manual_probe_policy": (
                "needs_manual_probe 由 Codex 继续原文细搜；没有候选页时不要求用户人工翻 PDF。"
            ),
        },
        "summary": summary,
        "rows": rows,
    }


def _load_search_cases(evidence_search_path: Path) -> dict[str, dict[str, Any]]:
    if not evidence_search_path.exists():
        return {}
    payload = _load_json(evidence_search_path)
    return {case["case_id"]: case for case in payload.get("cases", [])}


def _load_anchor_review_notes(
    anchor_review_notes_path: Path | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if anchor_review_notes_path is None or not anchor_review_notes_path.exists():
        return {}
    payload = _load_json(anchor_review_notes_path)
    review_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for case in payload.get("cases", []):
        company = case.get("company")
        query_id = case.get("query_id")
        if isinstance(company, str) and isinstance(query_id, str):
            review_by_key[(company, query_id)] = case
    return review_by_key


def _apply_anchor_review_notes(
    rows: list[dict[str, Any]],
    review_by_key: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for row in rows:
        review = review_by_key.get((row["company"], row["query_id"]))
        if review is None:
            continue
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


def _int_list(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, int)]


def _assign_review_priorities(rows: list[dict[str, Any]]) -> None:
    sampled_industries: set[str] = set()
    for row in rows:
        if row["auto_anchor_status"] == "blocked_missing_raw":
            row["review_priority"] = "must_review"
        elif row["auto_anchor_status"] == "needs_manual_probe":
            row["review_priority"] = "codex_probe"

    for row in rows:
        if _requires_user_priority_review(row):
            continue
        if row["auto_anchor_status"] != "auto_anchor_high_confidence":
            continue
        if not row["candidate_pages"]:
            continue
        industry = row["industry"]
        if industry in sampled_industries:
            continue
        row["review_priority"] = "sample_review"
        sampled_industries.add(industry)


def _requires_user_priority_review(row: dict[str, Any]) -> bool:
    return row.get("review_priority") in {"must_review", "sample_review"}


def _build_primary_rows(
    candidate_matrix: dict[str, Any],
    probe_by_case_id: dict[str, dict[str, Any]],
    search_by_case_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filing in candidate_matrix["primary_current_year_candidates"]:
        for query_id in filing.get("industry_candidates", []):
            definition = QUERY_DEFINITIONS[query_id]
            case_id = f"{filing['document_key']}::{query_id}"
            probe = probe_by_case_id.get(case_id, {})
            search = search_by_case_id.get(case_id, {})
            candidate_pages = [
                page["page"] for page in probe.get("matched_pages", []) if isinstance(page.get("page"), int)
            ]
            search_pages = [
                page for page in search.get("candidate_pages", []) if isinstance(page, int)
            ]
            anchor_status = probe.get("status") or "anchor_probe_missing"
            readiness = (
                "needs_anchor_confirmation"
                if anchor_status == "anchor_probe_hit"
                else "needs_manual_probe"
            )
            auto_anchor_status = search.get("status") or "evidence_search_missing"
            if auto_anchor_status == "auto_anchor_high_confidence":
                readiness = "needs_anchor_confirmation"
            elif auto_anchor_status == "auto_anchor_low_confidence":
                readiness = "needs_anchor_confirmation"
            elif auto_anchor_status == "needs_manual_probe":
                readiness = "needs_manual_probe"
            rows.append(
                _base_row(
                    definition=definition,
                    case_id=case_id,
                    company=filing["company"],
                    document_key=filing["document_key"],
                    local_path=filing["local_path"],
                    matrix_status=filing["status"],
                    anchor_status=anchor_status,
                    auto_anchor_status=auto_anchor_status,
                    review_priority=search.get("review_priority")
                    or "later_anchor_confirmation",
                    manifest_readiness=readiness,
                    candidate_pages=search_pages or candidate_pages,
                    candidate_snippets=search.get("candidate_snippets", []),
                    matched_terms=search.get("matched_terms", []),
                    forbidden_matched_terms=search.get("forbidden_matched_terms", []),
                    codex_probe_status=search.get("codex_probe_status", "not_requested"),
                    codex_probe_pages=search.get("codex_probe_pages", []),
                    codex_probe_snippets=search.get("codex_probe_snippets", []),
                    codex_probe_matched_terms=search.get("codex_probe_matched_terms", []),
                    codex_probe_score=int(search.get("codex_probe_score") or 0),
                    codex_probe_notes=str(search.get("codex_probe_notes") or ""),
                    evidence_search_score=int(search.get("evidence_search_score") or 0),
                    notes="候选页仍需人工或更强 parser 确认，不能直接升格为 expected_pages。",
                )
            )
    return rows


def _build_blocked_rows(candidate_matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for blocked in candidate_matrix["blocked_design_queries"]:
        query_id = blocked["query_id"]
        definition = QUERY_DEFINITIONS[query_id]
        rows.append(
            _base_row(
                definition=definition,
                case_id=f"blocked::{query_id}",
                company=blocked["required_company"],
                document_key=None,
                local_path=None,
                matrix_status=blocked.get("status") or "blocked_missing_raw",
                anchor_status=blocked.get("status") or "blocked_missing_raw",
                auto_anchor_status=blocked.get("status") or "blocked_missing_raw",
                review_priority="must_review",
                manifest_readiness="blocked_missing_raw",
                candidate_pages=[],
                candidate_snippets=[],
                matched_terms=[],
                forbidden_matched_terms=[],
                codex_probe_status="not_requested",
                codex_probe_pages=[],
                codex_probe_snippets=[],
                codex_probe_matched_terms=[],
                codex_probe_score=0,
                codex_probe_notes="",
                evidence_search_score=0,
                notes="当前缺 raw filing；先不进入 evidence-location pass。",
            )
        )
    return rows


def _base_row(
    *,
    definition: IndustryQueryDefinition,
    case_id: str,
    company: str | None,
    document_key: str | None,
    local_path: str | None,
    matrix_status: str,
    anchor_status: str,
    auto_anchor_status: str,
    review_priority: str,
    manifest_readiness: str,
    candidate_pages: list[int],
    candidate_snippets: list[str],
    matched_terms: list[str],
    forbidden_matched_terms: list[str],
    codex_probe_status: str,
    codex_probe_pages: list[int],
    codex_probe_snippets: list[str],
    codex_probe_matched_terms: list[str],
    codex_probe_score: int,
    codex_probe_notes: str,
    evidence_search_score: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "query_id": definition.query_id,
        "tier": "smoke_v2",
        "company": company,
        "industry": definition.industry,
        "document_key": document_key,
        "local_path": local_path,
        "query": definition.query,
        "expected_route": definition.expected_route,
        "expected_document_evidence_intent": definition.expected_document_evidence_intent,
        "primary_evidence_kind": definition.evidence_kinds[0],
        "secondary_evidence_kinds": list(definition.evidence_kinds[1:]),
        "candidate_pages": candidate_pages,
        "candidate_snippets": candidate_snippets,
        "matched_terms": matched_terms,
        "forbidden_matched_terms": forbidden_matched_terms,
        "codex_probe_status": codex_probe_status,
        "codex_probe_pages": codex_probe_pages,
        "codex_probe_snippets": codex_probe_snippets,
        "codex_probe_matched_terms": codex_probe_matched_terms,
        "codex_probe_score": codex_probe_score,
        "codex_probe_notes": codex_probe_notes,
        "evidence_search_score": evidence_search_score,
        "auto_anchor_status": auto_anchor_status,
        "review_priority": review_priority,
        "page_number_policy": "runtime_pymupdf_1_based_page_number",
        "anchor_review_status": "not_reviewed",
        "human_confirmed_pages": [],
        "human_corrected_pages": [],
        "human_rejected_candidate_pages": [],
        "human_missing_fields": [],
        "human_review_notes": "",
        "expected_pages": [],
        "expected_row_labels": [],
        "expected_metric_tags": [],
        "expected_section_types": [],
        "expected_answer_field_ids": list(definition.expected_answer_field_ids),
        "expected_answer_fields_label": definition.expected_answer_fields_label,
        "forbidden_failure_modes": list(definition.forbidden_failure_modes),
        "answer_hygiene_checks": _hygiene_checks(definition),
        "area": definition.area,
        "matrix_status": matrix_status,
        "anchor_status": anchor_status,
        "manifest_readiness": manifest_readiness,
        "mvp_status": manifest_readiness,
        "notes": notes,
        "status_sort": _status_sort(manifest_readiness),
    }


def _hygiene_checks(definition: IndustryQueryDefinition) -> list[str]:
    checks = ["no_raw_metadata", "no_empty_parentheses"]
    if definition.expected_document_evidence_intent in {"metric_value", "metric_attribution"}:
        checks.append("unit_period_present")
    return checks


def _status_sort(status: str) -> int:
    return {
        "needs_anchor_confirmation": 0,
        "needs_manual_probe": 1,
        "unsupported_or_no_clear_anchor": 2,
        "blocked_missing_raw": 3,
        "ready_for_manifest": 3,
    }.get(status, 9)


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Golden Queries v2 行业专项 Evidence Matrix",
        "",
        f"更新时间：{report['generated_at']}",
        "",
        "这是给用户检查的完整行业专项工作表，不是 runnable manifest。",
        "",
        "## 摘要",
        "",
        f"- 总行数：`{summary['total_rows']}`",
        f"- 当前 raw 可映射候选：`{summary['primary_candidate_rows']}`",
        f"- 缺 raw 阻塞：`{summary['blocked_missing_raw']}`",
        f"- 旧关键词 probe 有候选页：`{summary['anchor_probe_hit']}`",
        f"- 旧关键词 probe 无命中：`{summary['needs_manual_probe']}`",
        "",
        "## 口径",
        "",
        "- `candidate_pages` 是关键词 probe 找到的候选页，不等于最终 `expected_pages`。",
        "- `expected_pages`、`expected_row_labels`、`expected_metric_tags`、"
        "`expected_section_types` 现在故意留空，等 evidence-location pass 确认后再填。",
        "- `review_priority=later_anchor_confirmation` 表示不进入用户优先复核表，"
        "但仍必须在生成 runnable manifest 前完成 anchor confirmation。",
        "- `review_priority=codex_probe` 表示 Codex 继续找原文证据；没有候选页时不交给用户翻 PDF。",
        "- 页码使用 runtime / PyMuPDF 的 1-based page number，不使用 PDF 印刷页码。",
        "",
        "## 完整表",
        "",
        "| 状态 | 公司 | Query ID | Query | Intent | Evidence | 候选页 | 必需字段 | 禁止失败 | 备注 |",
        "|---|---|---:|---|---|---|---|---|---|---|",
    ]
    lines = lines[:-5]
    lines.extend(
        [
            "",
            "## 原文搜索结果",
            "",
            f"- 原文搜索高置信：`{summary['auto_anchor_high_confidence']}`",
            f"- 原文搜索低置信：`{summary['auto_anchor_low_confidence']}`",
            f"- 原文搜索后仍需 Codex 继续 probe：`{summary['auto_needs_manual_probe']}`",
            f"- Codex 继续 probe 队列：`{summary['codex_probe_rows']}`",
            f"- Codex deep probe 找到候选页：`{summary['codex_probe_candidate_rows']}`",
            f"- Codex deep probe 后仍 no-hit：`{summary['codex_probe_no_hit_rows']}`",
            f"- 建议用户优先复核行数：`{summary['review_rows']}`",
            f"- 已记录人工抽查反馈：`{summary['human_reviewed_rows']}`",
            f"- 人工修正候选页：`{summary['human_corrected_rows']}`",
            f"- 人工发现字段缺口：`{summary['human_partial_field_gap_rows']}`",
        ]
    )
    reviewed_rows = [
        row for row in report["rows"] if row.get("anchor_review_status") != "not_reviewed"
    ]
    if reviewed_rows:
        lines.extend(
            [
                "",
                "## 人工抽查反馈",
                "",
                "这些记录来自用户抽查，不代表全量 anchor confirmation。",
                "",
                "| 状态 | 公司 | Query ID | 人工确认页 | 人工修正页 | 字段缺口 | 备注 |",
                "|---|---|---:|---|---|---|---|",
            ]
        )
        for row in reviewed_rows:
            lines.append(_human_review_markdown_row(row))
    review_rows = [
        row for row in report["rows"] if _requires_user_priority_review(row)
    ]
    if review_rows:
        lines.extend(
            [
                "",
                "## 用户优先复核表",
                "",
                "| 复核优先级 | 自动状态 | 公司 | Query ID | Query | 需要核查的证据 | 候选页 | 分数 | 片段 |",
                "|---|---|---|---:|---|---|---|---:|---|",
            ]
        )
        for row in review_rows:
            lines.append(_review_markdown_row(row))
    codex_probe_rows = [
        row
        for row in report["rows"]
        if row.get("review_priority") == "codex_probe"
        or row.get("codex_probe_status") != "not_requested"
    ]
    if codex_probe_rows:
        lines.extend(
            [
                "",
                "## Codex probe 结果",
                "",
                "这些记录来自 Codex 原文细搜，不写入 `human_*`，也不等于 `expected_pages`。",
                "",
                "| Codex 状态 | 自动状态 | 公司 | Query ID | Query | Codex 候选页 | 分数 | 片段 / 备注 |",
                "|---|---|---|---:|---|---|---:|---|",
            ]
        )
        for row in codex_probe_rows:
            lines.append(_codex_probe_markdown_row(row))
    lines.extend(
        [
            "",
            "## 完整自动矩阵",
            "",
            "| 状态 | 公司 | Query ID | Query | Intent | Evidence | 候选页 | 必需字段 | 禁止失败 | 备注 |",
            "|---|---|---:|---|---|---|---|---|---|---|",
        ]
    )
    for row in report["rows"]:
        lines.append(_markdown_row(row))
    lines.append("")
    return "\n".join(lines)


def _markdown_row(row: dict[str, Any]) -> str:
    evidence = ", ".join([row["primary_evidence_kind"], *row["secondary_evidence_kinds"]])
    pages = ", ".join(str(page) for page in row["candidate_pages"]) or "-"
    fields = ", ".join(row["expected_answer_field_ids"])
    forbidden = "; ".join(row["forbidden_failure_modes"])
    snippets = " / ".join(row.get("candidate_snippets", [])[:2]) or "-"
    notes = (
        f"review={row.get('review_priority', '-')}; "
        f"auto={row.get('auto_anchor_status', '-')}; "
        f"codex={row.get('codex_probe_status', 'not_requested')}; "
        f"score={row.get('evidence_search_score', 0)}; "
        f"human={row.get('anchor_review_status', 'not_reviewed')}; "
        f"snippets={snippets}; "
        f"{row['notes']}"
    )
    return (
        f"| `{row['manifest_readiness']}` | {_esc(row['company'])} | `{row['query_id']}` | "
        f"{_esc(row['query'])} | `{row['expected_document_evidence_intent']}` | "
        f"`{evidence}` | {pages} | {_esc(fields)} | {_esc(forbidden)} | {_esc(notes)} |"
    )


def _review_markdown_row(row: dict[str, Any]) -> str:
    pages = ", ".join(str(page) for page in row["candidate_pages"]) or "-"
    snippets = " / ".join(row.get("candidate_snippets", [])[:2]) or "-"
    fields = ", ".join(row["expected_answer_field_ids"])
    return (
        f"| `{row.get('review_priority', '-')}` | `{row.get('auto_anchor_status', '-')}` | "
        f"{_esc(row['company'])} | `{row['query_id']}` | {_esc(row['query'])} | "
        f"{_esc(fields)} | {pages} | "
        f"{row.get('evidence_search_score', 0)} | {_esc(snippets)} |"
    )


def _human_review_markdown_row(row: dict[str, Any]) -> str:
    confirmed = ", ".join(str(page) for page in row.get("human_confirmed_pages", [])) or "-"
    corrected = ", ".join(str(page) for page in row.get("human_corrected_pages", [])) or "-"
    missing = ", ".join(row.get("human_missing_fields", [])) or "-"
    notes = row.get("human_review_notes") or "-"
    return (
        f"| `{row.get('anchor_review_status', '-')}` | {_esc(row['company'])} | "
        f"`{row['query_id']}` | {confirmed} | {corrected} | {_esc(missing)} | "
        f"{_esc(notes)} |"
    )


def _codex_probe_markdown_row(row: dict[str, Any]) -> str:
    pages = ", ".join(str(page) for page in row.get("codex_probe_pages", [])) or "-"
    snippets = " / ".join(row.get("codex_probe_snippets", [])[:2])
    notes = row.get("codex_probe_notes") or "-"
    detail = " / ".join(part for part in (snippets, notes) if part and part != "-") or "-"
    return (
        f"| `{row.get('codex_probe_status', '-')}` | "
        f"`{row.get('auto_anchor_status', '-')}` | {_esc(row['company'])} | "
        f"`{row['query_id']}` | {_esc(row['query'])} | {pages} | "
        f"{row.get('codex_probe_score', 0)} | {_esc(detail)} |"
    )


def _esc(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
