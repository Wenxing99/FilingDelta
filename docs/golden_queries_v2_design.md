# Golden Queries v2 Design

## Purpose

This document integrates local financial-statement reading notes, cross-industry annual-report research, and the current FilingDelta typed evidence implementation. The longer local research notes are intentionally treated as background material; this file is the implementation-facing design that can be shared with the repo.

The goal is to design a cross-industry golden-query set for FilingDelta. v2 should test whether the system can retrieve and answer from the right kind of evidence, not just whether a page-level semantic search can find one familiar Tencent or CMB page.

The core design target is:

- `page_text`: stable citation anchor and fallback.
- `section_text`: business review, risk, strategy, management discussion, and metric attribution.
- `table_row`: exact metrics, amounts, ratios, operating KPIs, segment rows, and period/unit-sensitive table evidence.

## Current FilingDelta State

As of the current implementation:

- `EvidenceKind` supports `page_text`, `section_text`, and `table_row`.
- `build_evidence_units(...)` indexes page chunks, section evidence, and conservative table-row evidence into the same Qdrant-backed retrieval layer.
- The chat router emits `document_evidence_intent`:
  - `metric_value`
  - `metric_attribution`
  - `business_narrative`
  - `fallback`
- `ChatQAService` maps intent to retrieval strategy:
  - `metric_value` -> `table_row` first, with `page_text` fallback included.
  - `metric_attribution` -> `section_text` first, then `table_row`, then `page_text`.
  - `business_narrative` -> `section_text` first, then `page_text`.
  - `fallback` -> current page-level behavior.

Existing eval coverage is useful but narrow:

- `golden_queries_v1_1` covers CMB and Tencent only.
- `section_text` retrieval comparison shows narrative retrieval is viable on CMB/Tencent, but not yet cross-industry.
- `table_row` retrieval eval covers 7 metric/table cases across CMB/Tencent and shows table-row retrieval improves row/metric hits, but the row taxonomy is still conservative.

Therefore v2 should not be "more CMB/Tencent cases". It should become a cross-industry retrieval and answer-quality design.

## Design Principles

1. Golden queries are evidence tests, not just answer tests.
2. Each query should have a primary evidence kind and, when needed, a secondary evidence kind.
3. Every query should define forbidden failure modes.
4. Industry-specific KPIs must keep their names and meanings; do not normalize everything into generic revenue/profit.
5. `page_text` remains the citation anchor even when `section_text` or `table_row` is the primary retrieval object.
6. Do not require external data unless the query is explicitly marked `mixed`.
7. Do not ask for derived metrics unless the document explicitly discloses them or the query is marked as requiring future capability.

## Route And Evidence Taxonomy

`golden_queries_v2` must keep chat route and document evidence intent separate.

Chat route is the top-level QA path:

| Expected route | Meaning | Example |
|---|---|---|
| `document_only` | the filing alone should answer the question | "公司收入增长的主要原因是什么？" |
| `concept_only` | the question asks only for external concept/background | "什么是 ROE？" |
| `mixed` | the answer needs both an external concept and document-specific facts | "什么是 ROE？结合当前文档说明它意味着什么。" |
| `unsupported` | out of current filing-analysis scope | investment advice / target price requests |

Document evidence intent is the retrieval intent inside `document_only` or `mixed` routes:

| Document evidence intent | Meaning | Primary evidence | Secondary evidence | Typical answer contract |
|---|---|---|---|---|
| `metric_value` | asks for value, amount, ratio, volume, share, or direct change | `table_row` | `page_text` | metric, value, unit, period, scope, citation |
| `metric_attribution` | asks why a metric changed or what drove growth/decline | `section_text` | `table_row`, `page_text` | value anchor, management-stated drivers, boundary |
| `business_narrative` | asks how the company describes business, strategy, risks, products, or actions | `section_text` | `page_text`, sometimes `table_row` | company-specific points, evidence quote/page, no generic prose |
| `fallback` | no document evidence needed or intent unclear | `page_text` | none | conservative fallback |

Important: `mixed` is a route, not a document evidence intent. A mixed query should still specify the document-side evidence intent, for example:

- "什么是 ROE？结合当前文档说明它意味着什么。" -> `expected_route=mixed`, `expected_document_evidence_intent=metric_value`
- "资本开支增加通常意味着什么？结合当前文档回答。" -> `expected_route=mixed`, `expected_document_evidence_intent=metric_attribution`

## Implementation-Ready Manifest Contract

Future JSON should use one top-level document registry and one query list. This avoids repeating source paths and makes it possible to run by company, industry, tier, or intent.

Top-level document entry:

```json
{
  "document_key": "catl_2024_annual",
  "source_path": "data/raw/宁德时代2024年度报告.pdf",
  "company_name": "宁德时代",
  "ticker": "300750",
  "market": "a_share",
  "doc_type": "annual_report",
  "fiscal_period": "2024 annual report",
  "language": "zh",
  "industry": "新能源汽车/电池"
}
```

Query entry:

```json
{
  "id": "BAT-02",
  "tier": "smoke_v2",
  "company": "宁德时代",
  "industry": "新能源汽车/电池",
  "document_key": "catl_2024_annual",
  "query": "宁德时代为什么出现收入下降但净利润增长？",
  "query_aliases": [],
  "expected_route": "document_only",
  "expected_document_evidence_intent": "metric_attribution",
  "primary_evidence_kind": "section_text",
  "secondary_evidence_kinds": ["table_row", "page_text"],
  "expected_pages": [],
  "expected_row_labels": ["营业收入", "归属股东净利润"],
  "expected_metric_tags": ["revenue", "profit"],
  "expected_section_types": ["business_review", "profitability_quality"],
  "expected_document_area_ids": ["mda", "revenue_cost_analysis", "gross_margin_discussion"],
  "expected_answer_field_ids": ["revenue_change", "net_profit_change", "margin_or_cost_driver", "management_attribution"],
  "forbidden_failure_modes": [
    "把收入下降归因于需求崩盘但没有文档依据",
    "只给数字不解释管理层归因",
    "使用半年报数据回答全年问题"
  ],
  "answer_hygiene_checks": ["no_raw_metadata", "no_empty_parentheses", "unit_period_present"],
  "mvp_status": "immediate"
}
```

Required query fields for the first implementation:

- `id`
- `tier`
- `document_key`
- `query`
- `expected_route`
- `expected_document_evidence_intent`
- `primary_evidence_kind`
- `secondary_evidence_kinds`
- `expected_pages`
- `forbidden_failure_modes`
- `mvp_status`

Strongly recommended fields:

- `query_aliases`
- `expected_row_labels`
- `expected_metric_tags`
- `expected_section_types`
- `expected_document_area_ids`
- `expected_answer_field_ids`
- `answer_hygiene_checks`

`expected_document_area_ids` and `expected_answer_field_ids` should be stable enum-like IDs, not arbitrary prose. Human-readable notes can be added later, but runner-facing fields should remain normalized.

Recommended scoring fields:

- `route_hit`
- `intent_hit`
- `evidence_kind_hit@k`
- `page_hit@k`
- `table_row_label_hit@k`
- `metric_tag_hit@k`
- `section_type_hit@k`
- `required_fields_present`
- `forbidden_failure_absent`
- `citation_anchor_valid`

Metric mapping:

- `route_hit` compares runtime route with `expected_route`.
- `intent_hit` compares runtime `document_evidence_intent` with `expected_document_evidence_intent`.
- `page_hit@k` uses `expected_pages`.
- `table_row_label_hit@k` uses `expected_row_labels`.
- `metric_tag_hit@k` uses `expected_metric_tags`.
- `section_type_hit@k` uses `expected_section_types`.
- `required_fields_present` should initially check stable `expected_answer_field_ids` with simple alias/matcher rules, not free-form prose.

## v2 Suite Shape

Use three tiers:

| Tier | Size | Purpose |
|---|---:|---|
| `smoke_v2` | 30-36 queries | fast cross-industry regression after retrieval/router/answerer changes |
| `core_v2` | 60-90 queries | broader industry coverage across 10-12 business models |
| `future_v2` | open-ended | note-heavy, derived, cross-company, external-data, or deep accounting cases |

For the next implementation step, start with `smoke_v2`.

## Universal Query Templates

These apply to almost every annual report. In JSON, instantiate them per document only when the report actually discloses the requested metric. Unless explicitly stated otherwise, these templates default to `expected_route=document_only`; the `Document evidence intent` column maps to `expected_document_evidence_intent`.

| ID | Query template | Document evidence intent | Primary evidence | Expected area | Forbidden failures |
|---|---|---|---|---|---|
| U-01 | 公司本报告期营业收入、归母净利润和 ROE/ROAE 分别是多少？ | `metric_value` | `table_row` | financial summary, key metrics | wrong period; missing unit; ROAA mistaken for ROAE; adjusted and unadjusted mixed |
| U-02 | 公司收入按业务分部或产品如何构成？哪个分部最大？ | `metric_value` | `table_row` | segment revenue, revenue by type | segment table confused with geographic or product table |
| U-03 | 本期经营活动现金流净额是多少？与净利润相比如何？ | `metric_value` | `table_row` | cash flow statement, financial review | net cash change used as operating cash flow |
| U-04 | 公司收入或利润变化的主要原因是什么？ | `metric_attribution` | `section_text` | MD&A, business review | only values returned; external macro reasons invented |
| U-05 | 毛利率、费用率或净利率变化说明什么？ | `metric_attribution` | `section_text` | profitability discussion | net margin cause used for gross margin without evidence |
| U-06 | 本期资本开支、研发投入或长期投资有什么披露？ | `metric_value` | `table_row` | capex, R&D, investment plan | R&D expense, capex, investment cash flow mixed without labels |
| U-07 | 分红或股息方案是什么？ | `metric_value` | `table_row` | dividend, profit distribution | historical dividend used as current proposal; currency omitted |
| U-08 | 公司披露了哪些主要风险以及应对措施？ | `business_narrative` | `section_text` | risk factors, risk management | generic risk list without document-specific citations |
| U-09 | 公司未来经营重点或战略方向是什么？ | `business_narrative` | `section_text` | outlook, strategy | historical results presented as future plan |
| U-10 | 公司是否存在利润和现金流背离？年报如何解释？ | `metric_attribution` | `table_row` | cash flow, working capital, MD&A | says "profit is not cash" without document evidence |

## Recommended `smoke_v2` Cross-Industry Set

This is the first target set after the relevant annual reports are added to `data/raw`. It is intentionally not exhaustive. All listed cases default to `expected_route=document_only`; future concept-plus-document cases should explicitly set `expected_route=mixed` and still specify a document-side evidence intent.

### Home Appliances And Smart Home

Representative companies: 美的集团、海尔智家.

| ID | Query | Document evidence intent | Evidence | Area | Expected answer fields | Forbidden failures |
|---|---|---|---|---|---|---|
| HA-01 | 美的集团智能家居和商业及工业解决方案收入分别是多少？ | `metric_value` | `table_row` | segment revenue | segment names, revenue, YoY/share if disclosed | product row treated as segment row without caveat |
| HA-02 | 海尔智家国内和海外业务增长分别由哪些因素驱动？ | `metric_attribution` | `section_text` | business/geographic review | China drivers, overseas drivers, product/channel examples | macro causes invented outside MD&A |
| HA-03 | 家电企业存货或渠道库存是否异常？公司如何描述渠道效率？ | `metric_attribution` | `table_row` + `section_text` | inventory, channel review | inventory value/turnover if disclosed, channel explanation | all inventory increases described as unsold goods |

### Shipping And Container Logistics

Representative company: 中远海控.

| ID | Query | Document evidence intent | Evidence | Area | Expected answer fields | Forbidden failures |
|---|---|---|---|---|---|---|
| SHIP-01 | 中远海控集装箱航运业务货运量和收入分别是多少？ | `metric_value` | `table_row` | container shipping table | TEU volume, revenue, YoY, unit | terminal throughput confused with shipping volume |
| SHIP-02 | 中远海控利润增长主要由哪些因素驱动？ | `metric_attribution` | `section_text` + `table_row` | MD&A, industry trend | volume, freight rate/effective capacity, route disruption if disclosed | "demand growth" only, no rate/capacity driver |
| SHIP-03 | 红海局势在年报中如何影响航运供需和运价？ | `business_narrative` | `section_text` | industry trend, outlook/risk | explicit Red Sea wording, capacity/route/rate effect | geopolitical speculation beyond report |

### Oil, Gas, Coal, And Integrated Energy

Representative companies: 中国海油、中国石油、中国神华.

| ID | Query | Document evidence intent | Evidence | Area | Expected answer fields | Forbidden failures |
|---|---|---|---|---|---|---|
| OIL-01 | 中国海油油气净产量、净证实储量和储量寿命分别是多少？ | `metric_value` | `table_row` | reserves and production | production, proved reserves, reserve life, unit | barrels/BOE/cubic feet mixed without label |
| OIL-02 | 中国海油资本开支是多少？主要投向哪些环节？ | `metric_value` | `table_row` + `section_text` | capex, business review | total capex, split/use if disclosed | planned capex used as actual without label |
| PTR-01 | 中国石油油气当量产量、天然气产量和炼油加工量分别是多少？ | `metric_value` | `table_row` | operating statistics | output, gas, refining throughput, units | future target used as historical actual |
| COAL-01 | 中国神华商品煤产量、煤炭销售量和平均售价是多少？ | `metric_value` | `table_row` | coal production/sales table | production, sales, ASP, YoY | self-produced coal confused with total sales |
| COAL-02 | 中国神华一体化运营模式包括哪些环节？它如何影响抗周期能力？ | `business_narrative` | `section_text` | business overview, MD&A | coal, power, railway, port, shipping, chemical; synergies | says integration fully eliminates commodity risk |

### Hydropower And Regulated Power

Representative company: 长江电力.

| ID | Query | Document evidence intent | Evidence | Area | Expected answer fields | Forbidden failures |
|---|---|---|---|---|---|---|
| HYDRO-01 | 长江电力发电量、上网电量、售电量和上网电价分别是多少？ | `metric_value` | `table_row` | electricity volume/tariff table | generation, on-grid volume, sales volume, tariff | 万千瓦时 and 亿千瓦时 confused |
| HYDRO-02 | 长江电力业绩增长与来水和发电量有什么关系？ | `metric_attribution` | `section_text` + `table_row` | hydrology, business review | inflow/generation explanation, profit linkage if stated | tariff rise invented when not disclosed |
| HYDRO-03 | 公司如何描述抽蓄、新能源和储能布局？ | `business_narrative` | `section_text` | strategy, business review | pumped storage, water-wind-solar integration, project examples | project mentions converted into revenue contribution |

### Premium Consumer And Advertising Media

Representative companies: 贵州茅台、分众传媒.

| ID | Query | Document evidence intent | Evidence | Area | Expected answer fields | Forbidden failures |
|---|---|---|---|---|---|---|
| BAIJIU-01 | 贵州茅台茅台酒和系列酒收入分别是多少？哪个增长更快？ | `metric_value` | `table_row` | product revenue | product revenue, YoY, share | approximate narrative used when table has exact value |
| BAIJIU-02 | 贵州茅台直销和批发代理渠道收入结构如何变化？ | `metric_value` | `table_row` | channel revenue | direct revenue, wholesale revenue, share, YoY | ecommerce commentary treated as direct-sales number |
| MEDIA-01 | 分众传媒楼宇媒体收入占主营业务收入的比例是多少？ | `metric_value` | `table_row` | product/segment revenue | building media revenue, share, other media | product description used as numeric share |
| MEDIA-02 | 分众传媒生活圈媒体网络覆盖哪些城市和设备类型？ | `business_narrative` | `section_text` + `table_row` | business overview | city coverage, elevator TV/poster devices, overseas coverage | interim data mixed with annual without label |

### Insurance And Integrated Finance

Representative company: 中国平安.

| ID | Query | Document evidence intent | Evidence | Area | Expected answer fields | Forbidden failures |
|---|---|---|---|---|---|---|
| INS-01 | 中国平安归母营运利润、归母净利润和营业收入分别是多少？ | `metric_value` | `table_row` | financial highlights | operating profit, net profit, revenue, YoY | operating profit confused with IFRS operating income |
| INS-02 | 平安寿险及健康险新业务价值 NBV 增长了多少？主要渠道贡献如何？ | `metric_attribution` | `table_row` + `section_text` | life and health review | NBV, YoY, channel contribution | NBV confused with premium income |
| INS-03 | 平安产险综合成本率是多少？年报如何解释业务质量变化？ | `metric_attribution` | `table_row` + `section_text` | P&C review | combined ratio, YoY/pps, underwriting/risk explanation | loss ratio used as combined ratio |

### New Energy Vehicles And Batteries

Representative companies: 比亚迪、宁德时代.

| ID | Query | Document evidence intent | Evidence | Area | Expected answer fields | Forbidden failures |
|---|---|---|---|---|---|---|
| NEV-01 | 比亚迪汽车业务和手机部件及组装业务收入分别是多少？占比如何？ | `metric_value` | `table_row` | product/segment revenue | segment revenue, share, YoY | BYD Electronic standalone data treated as group segment |
| NEV-02 | 比亚迪新能源汽车销量是多少？销量增长如何影响收入？ | `metric_attribution` | `table_row` + `section_text` | sales volume, business review | sales volume, YoY, management attribution | production volume used when sales volume is asked |
| BAT-01 | 宁德时代动力电池和储能电池收入分别是多少？ | `metric_value` | `table_row` | product revenue | EV battery revenue, ESS revenue, share, YoY | H1 data used for annual query |
| BAT-02 | 宁德时代为什么出现收入下降但净利润增长？ | `metric_attribution` | `section_text` + `table_row` | MD&A, gross margin, raw materials | revenue change, profit change, margin/cost explanation | demand collapse invented without filing evidence |

### OTA, Ecommerce Platforms, Local Services, IP, And Sportswear

Representative companies: 携程集团、阿里巴巴、美团、泡泡玛特、安踏体育.

| ID | Query | Document evidence intent | Evidence | Area | Expected answer fields | Forbidden failures |
|---|---|---|---|---|---|---|
| OTA-01 | 携程住宿预订、交通票务、旅游度假和商旅收入分别是多少？ | `metric_value` | `table_row` | revenue by service | four service revenues, YoY, share | Q4 numbers mixed with full-year numbers |
| OTA-02 | 携程国际业务增长有哪些披露？ | `business_narrative` | `section_text` | business updates | international OTA, outbound/inbound indicators | global travel trend cited without company disclosure |
| BABA-01 | 阿里巴巴各业务分部收入分别是多少？ | `metric_value` | `table_row` | segment revenue | Taobao/Tmall, Cloud, AIDC, Cainiao, Local Services, etc. | segment reclassification ignored |
| BABA-02 | 阿里收入按类型如何构成？客户管理、云服务、物流服务分别是多少？ | `metric_value` | `table_row` | revenue by type notes | customer management, cloud, logistics, goods sales | segment revenue confused with revenue type |
| LOCAL-01 | 美团核心本地商业和新业务收入分别是多少？ | `metric_value` | `table_row` | segment revenue | core local commerce revenue, new initiatives revenue, total, YoY | old segment names used without mapping |
| LOCAL-02 | 美团新业务亏损为什么收窄？ | `metric_attribution` | `section_text` + `table_row` | MD&A, segment profit/loss | operating loss amount, management-stated changes | loss narrowing described as segment profit |
| IP-01 | 泡泡玛特自有产品和艺术家 IP 收入占比是多少？ | `metric_value` | `table_row` | revenue by product/IP | proprietary products, artist IPs, revenue, share | current Labubu news used for historical filing query |
| IP-02 | THE MONSTERS、MOLLY、SKULLPANDA 等 IP 收入排名如何？ | `metric_value` | `table_row` | IP revenue table | IP names, revenue, share, ranking | IP names translated/merged inconsistently |
| SPORTS-01 | 安踏体育 ANTA、FILA 和其他品牌收入分别是多少？ | `metric_value` | `table_row` | brand/segment revenue | brand revenue, YoY, operating margin if disclosed | Amer Sports JV treated as consolidated brand revenue |
| SPORTS-02 | 安踏电商收入占比和库存周转天数如何变化？ | `metric_value` | `table_row` | operating metrics | ecommerce contribution, inventory turnover days, YoY/change | inventory balance confused with turnover days |

## Future Capability Set

Mark these as `requires_future_capability` instead of forcing them into smoke v2:

- customer/supplier concentration across all companies;
- lease liabilities, contract liabilities, impairment tests, and other deep note tables;
- derived metrics not explicitly disclosed;
- multi-document or cross-company comparison;
- external market share or current news unless the filing cites it;
- investment advice, valuation, target price, or buy/sell decisions;
- NPV/WACC/ROIC economic-profit style analysis unless explicitly supported by document and external assumptions.

## Acceptance Criteria For v2 Implementation

Before turning this design into JSON and runner code, require:

1. Each selected company has one local annual report file in `data/raw`.
2. Each query references one document, one company, and one report period.
3. Each query has one primary evidence kind.
4. Each query has explicit forbidden failure modes.
5. The runner records route, `document_evidence_intent`, retrieved evidence kinds, citation pages, latency, and answer hygiene checks.
6. The first `smoke_v2` runner should not require perfect semantic answer grading; it should first establish retrieval and output hygiene.

## Current Implementation Status

截至 2026-05-01，`golden_queries_v2` 已经进入首轮可运行 smoke/eval 诊断阶段：

- 已生成首个 anchor-confirmed runnable manifest：`data/outputs/eval/golden_queries_v2_smoke.json`，当前包含 `14` 条人工确认或修正页码的 case。
- `expected_pages` 只允许来自 `human_confirmed_pages` / `human_corrected_pages`；candidate pages、Codex probe pages、BM25/hybrid 命中页都不能自动升格为 gold。
- live retrieval pilot 只跑当前 14 条 manifest case，不跑完整 answer synthesis；结果为 `6 passed / 8 failed`。
- BM25 / hybrid retrieval diagnosis 只是 page-hit-only 对照实验，用来判断 retrieval 排序问题；它不是正式系统接入，也不代表 full live pilot rescue。
- failure probe 已把剩余失败归因为四类：router intent mismatch、table extraction gap、rank/rerank issue、generic metric row dominance。

## Next Implementation Step

下一步不要扩大 manifest，也不要改 gold。按最小工程切片推进：

1. 先修 `HA-03` 的 router intent 判别问题，并重跑当前 14 条 live retrieval pilot。
2. 再选 `HYDRO-01` 加一个消费/制造代表 case，修 table_row 抽取、行业指标别名或表格识别缺口。
3. 单独处理 `BABA-01` 的 metric/segment-aware table-row rerank，降低通用财报行和附注页优先级。
4. 最后处理 `HA-02` 这类 gold page low-rank 问题，把 BM25 可解释的页级信号转成轻量 rerank/boost 验证。

在上述切片通过之前，不要把 BM25/hybrid 接入正式 `ChatQAService`，也不要进入 full answer-quality benchmark。
