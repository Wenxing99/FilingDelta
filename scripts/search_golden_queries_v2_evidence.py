from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import fitz

from filingdelta.core.config import REPO_ROOT


DEFAULT_MATRIX = Path("data/outputs/eval/golden_queries_v2_industry_evidence_matrix.json")
DEFAULT_OUTPUT = Path("data/outputs/eval/golden_queries_v2_evidence_search.json")
DEFAULT_TOP_PAGES = 5

PAGE_NUMBER_POLICY = "runtime_pymupdf_1_based_page_number"

_TRAD_TO_SIMP = str.maketrans(
    {
        "業": "业",
        "務": "务",
        "電": "电",
        "商": "商",
        "貿": "贸",
        "線": "线",
        "庫": "库",
        "存": "存",
        "貨": "货",
        "週": "周",
        "轉": "转",
        "數": "数",
        "據": "据",
        "經": "经",
        "營": "营",
        "虧": "亏",
        "損": "损",
        "窄": "窄",
        "變": "变",
        "動": "动",
        "長": "长",
        "潤": "润",
        "淨": "净",
        "歸": "归",
        "母": "母",
        "證": "证",
        "壽": "寿",
        "產": "产",
        "銷": "销",
        "售": "售",
        "額": "额",
        "價": "价",
        "費": "费",
        "險": "险",
        "際": "际",
        "國": "国",
        "內": "内",
        "類": "类",
        "別": "别",
        "佔": "占",
        "比": "比",
        "與": "与",
        "為": "为",
        "現": "现",
        "金": "金",
        "流": "流",
        "會": "会",
        "計": "计",
        "東": "东",
        "資": "资",
        "負": "负",
        "債": "债",
        "權": "权",
        "益": "益",
        "應": "应",
        "收": "收",
        "發": "发",
        "網": "网",
        "體": "体",
        "關": "关",
        "聯": "联",
        "總": "总",
        "約": "约",
        "場": "场",
        "風": "风",
        "訊": "讯",
        "雲": "云",
        "灣": "湾",
        "眾": "众",
        "傳": "传",
        "媒": "媒",
        "樓": "楼",
        "宇": "宇",
        "億": "亿",
        "萬": "万",
        "元": "元",
    }
)

EXTRA_QUERY_TERMS: dict[str, tuple[str, ...]] = {
    "HA-01": ("智能家居", "商业及工业解决方案", "商業及工業解決方案", "分部收入", "收入"),
    "HA-02": ("国内市场", "國內市場", "海外市场", "海外市場", "增长", "增長", "驱动", "overseas"),
    "HA-03": ("存货", "存貨", "库存", "庫存", "渠道", "周转", "週轉", "库存效率", "庫存效率"),
    "SHIP-01": ("集装箱航运", "集裝箱航運", "货运量", "貨運量", "TEU", "收入"),
    "SHIP-02": ("利润增长", "利潤增長", "运价", "運價", "货量", "貨量", "运力", "運力"),
    "SHIP-03": ("红海", "紅海", "绕航", "繞航", "供需", "运价", "運價"),
    "OIL-01": ("净产量", "淨產量", "证实储量", "證實儲量", "储量寿命", "儲量壽命", "油气", "油氣"),
    "OIL-02": ("资本开支", "資本開支", "资本支出", "資本支出", "勘探", "开发", "開發"),
    "COAL-01": ("商品煤", "产量", "產量", "销售量", "銷售量", "平均售价", "平均售價"),
    "COAL-02": ("一体化", "一體化", "煤炭", "电力", "電力", "铁路", "鐵路", "港口", "航运", "航運", "化工"),
    "BAIJIU-01": ("茅台酒", "系列酒", "收入", "增长", "增長"),
    "BAIJIU-02": ("直销", "直銷", "批发", "批發", "代理", "渠道", "收入"),
    "INS-01": ("归母营运利润", "歸母營運利潤", "归母净利润", "歸母淨利潤", "营业收入", "營業收入"),
    "INS-02": ("新业务价值", "新業務價值", "NBV", "寿险", "壽險", "健康险", "健康險", "渠道"),
    "INS-03": ("综合成本率", "綜合成本率", "产险", "產險", "业务质量", "業務質量"),
    "NEV-01": ("汽车业务", "汽車業務", "手机部件", "手機部件", "组装", "組裝", "收入"),
    "NEV-02": ("新能源汽车", "新能源汽車", "销量", "銷量", "收入"),
    "BAT-01": ("动力电池", "動力電池", "储能电池", "儲能電池", "收入"),
    "BAT-02": ("收入下降", "净利润增长", "淨利潤增長", "毛利率", "原材料"),
    "OTA-01": (
        "accommodation reservation",
        "transportation ticketing",
        "packaged-tour",
        "corporate travel",
    ),
    "OTA-02": ("international", "outbound", "inbound", "global"),
    "BABA-01": ("淘天", "云智能", "雲智能", "菜鸟", "菜鳥", "本地生活", "收入"),
    "BABA-02": ("客户管理", "客戶管理", "云服务", "雲服務", "物流服务", "物流服務", "收入"),
    "LOCAL-01": ("核心本地商业", "核心本地商業", "本地商业", "本地商業", "新业务", "新業務", "收入"),
    "LOCAL-02": ("新业务", "新業務", "亏损", "虧損", "收窄", "经营亏损", "經營虧損", "operating loss"),
    "IP-01": ("自有产品", "自有產品", "艺术家", "藝術家", "IP", "收入", "占比"),
    "IP-02": ("THE MONSTERS", "MOLLY", "SKULLPANDA", "收入", "排名"),
    "SPORTS-01": ("ANTA", "FILA", "其他品牌", "收入"),
    "SPORTS-02": (
        "电商",
        "電商",
        "电子商务",
        "電子商務",
        "电子商贸",
        "電子商貿",
        "线上",
        "線上",
        "库存周转天数",
        "庫存周轉天數",
        "存货周转",
        "存貨周轉",
        "inventory turnover",
    ),
}

FIELD_TERMS: dict[str, tuple[str, ...]] = {
    "revenue": ("收入", "revenue"),
    "yoy": ("同比", "增长", "增長", "year-on-year"),
    "share": ("占比", "佔比", "比例", "share"),
    "unit": ("人民币", "人民幣", "元", "million", "billion"),
    "segment_revenue": ("分部收入", "分部", "收入"),
    "channel_explanation": ("渠道", "效率", "channel"),
    "inventory_value_or_turnover": ("存货", "存貨", "库存", "庫存", "周转", "週轉"),
    "operating_loss": ("经营亏损", "經營虧損", "亏损", "虧損", "operating loss"),
    "management_stated_changes": ("收窄", "改善", "优化", "優化"),
    "ecommerce_contribution": ("电商", "電商", "电子商务", "電子商務", "电子商贸", "電子商貿"),
    "inventory_turnover_days": ("库存周转天数", "庫存周轉天數", "存货周转", "存貨周轉"),
    "core_local_commerce_revenue": ("核心本地商业", "核心本地商業", "收入"),
    "new_initiatives_revenue": ("新业务", "新業務", "收入"),
}

SECTION_TERMS = (
    "管理层讨论",
    "管理層討論",
    "业务回顾",
    "業務回顧",
    "管理层讨论与分析",
    "管理層討論與分析",
    "经营情况讨论",
    "經營情況討論",
    "风险",
    "風險",
    "展望",
    "策略",
    "战略",
    "業務",
    "business review",
    "management discussion",
    "risk",
    "outlook",
)

CODEX_DEEP_PROBE_TARGETS: set[tuple[str, str]] = {
    ("海尔智家", "HA-03"),
    ("美团", "LOCAL-02"),
    ("中远海控", "SHIP-01"),
    ("分众传媒", "MEDIA-01"),
    ("泡泡玛特", "IP-01"),
    ("阿里巴巴", "BABA-01"),
}

DEEP_PROBE_TERMS: dict[str, tuple[str, ...]] = {
    "HA-03": (
        "存货",
        "存貨",
        "库存",
        "庫存",
        "渠道",
        "周转",
        "週轉",
        "渠道效率",
        "库存周转",
        "庫存週轉",
        "呆滞",
        "呆滯",
        "一盘货",
        "一盤貨",
        "动销",
        "動銷",
    ),
    "LOCAL-02": (
        "新业务",
        "新業務",
        "亏损",
        "虧損",
        "收窄",
        "经营亏损",
        "經營虧損",
        "分部亏损",
        "分部虧損",
        "operating loss",
        "segment loss",
        "loss narrowed",
    ),
    "SHIP-01": (
        "集装箱航运",
        "集裝箱航運",
        "集装箱航运业务",
        "集裝箱航運業務",
        "货运量",
        "貨運量",
        "TEU",
        "航线",
        "航線",
        "收入",
    ),
    "MEDIA-01": (
        "楼宇媒体",
        "樓宇媒體",
        "主营业务收入",
        "主營業務收入",
        "营业收入",
        "營業收入",
        "收入构成",
        "收入構成",
        "占比",
        "佔比",
    ),
    "IP-01": (
        "Proprietary products",
        "artist IPs",
        "自有产品",
        "自有產品",
        "艺术家",
        "藝術家",
        "Revenue by IPs",
        "收入",
        "占比",
        "佔比",
    ),
    "BABA-01": (
        "淘天",
        "云智能",
        "雲智能",
        "菜鸟",
        "菜鳥",
        "本地生活",
        "国际数字商业",
        "國際數字商業",
        "分部收入",
        "收入",
    ),
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self._parts)


@dataclass(frozen=True)
class PageSearchHit:
    page: int
    score: int
    confidence: str
    matched_terms: tuple[str, ...]
    forbidden_matched_terms: tuple[str, ...]
    snippets: tuple[str, ...]
    feature_scores: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "score": self.score,
            "confidence": self.confidence,
            "matched_terms": list(self.matched_terms),
            "forbidden_matched_terms": list(self.forbidden_matched_terms),
            "snippets": list(self.snippets),
            "feature_scores": self.feature_scores,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search raw filing text for golden_queries_v2 evidence anchor candidates."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-pages", type=int, default=DEFAULT_TOP_PAGES)
    args = parser.parse_args(argv)

    if args.top_pages < 1:
        raise SystemExit("--top-pages must be >= 1.")

    report = build_evidence_search_report(
        matrix_path=_resolve(args.matrix),
        top_pages=args.top_pages,
    )
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]
    print(
        "evidence_search "
        f"cases={summary['cases_total']} "
        f"high={summary['auto_anchor_high_confidence']} "
        f"low={summary['auto_anchor_low_confidence']} "
        f"manual={summary['needs_manual_probe']} "
        f"blocked={summary['blocked_missing_raw']} "
        f"output={output_path}"
    )
    return 0


def build_evidence_search_report(*, matrix_path: Path, top_pages: int) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    page_cache: dict[str, list[tuple[int, str]]] = {}
    deep_page_cache: dict[str, list[tuple[int, str]]] = {}

    for row in matrix["rows"]:
        if row["manifest_readiness"] == "blocked_missing_raw":
            cases.append(_blocked_case(row))
            continue

        local_path = row.get("local_path")
        if not local_path:
            cases.append(_manual_case(row, error="missing_local_path"))
            continue

        source_path = _resolve(Path(local_path))
        cache_key = str(source_path)
        if cache_key not in page_cache:
            try:
                page_cache[cache_key] = _load_pages(source_path)
            except (OSError, RuntimeError, ValueError) as exc:
                cases.append(_manual_case(row, error=f"{type(exc).__name__}: {exc}"))
                continue

        hits = _search_pages(row=row, pages=page_cache[cache_key], top_pages=top_pages)
        codex_probe_hits: list[PageSearchHit] = []
        codex_probe_error: str | None = None
        if _should_run_codex_deep_probe(row):
            try:
                if cache_key not in deep_page_cache:
                    deep_page_cache[cache_key] = _load_deep_pages(source_path)
                codex_probe_hits = _deep_probe_pages(
                    row=row,
                    pages=deep_page_cache[cache_key],
                    top_pages=top_pages,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                codex_probe_error = f"{type(exc).__name__}: {exc}"
        cases.append(
            _searched_case(
                row=row,
                hits=hits,
                codex_probe_hits=codex_probe_hits,
                codex_probe_error=codex_probe_error,
            )
        )

    summary = _summarize(cases)
    return {
        "schema_version": "golden_queries_v2_evidence_search.v1",
        "generated_at": date.today().isoformat(),
        "matrix_path": _display_path(matrix_path),
        "page_number_policy": PAGE_NUMBER_POLICY,
        "summary": summary,
        "cases": cases,
    }


def _blocked_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "query_id": row["query_id"],
        "company": row["company"],
        "industry": row["industry"],
        "status": "blocked_missing_raw",
        "review_priority": "must_review",
        "evidence_search_score": 0,
        "candidate_pages": [],
        "candidate_snippets": [],
        "matched_terms": [],
        "forbidden_matched_terms": [],
        "page_hits": [],
        "page_number_policy": PAGE_NUMBER_POLICY,
        **_empty_codex_probe_payload(status="not_requested"),
        "anchor_notes": "当前缺 raw filing；继续 defer。",
    }


def _manual_case(row: dict[str, Any], *, error: str) -> dict[str, Any]:
    codex_probe_payload = (
        {
            **_empty_codex_probe_payload(status="codex_probe_error"),
            "codex_probe_terms": _deep_probe_terms(row),
            "codex_probe_notes": f"Codex deep probe 无法执行：{error}",
        }
        if _should_run_codex_deep_probe(row)
        else _empty_codex_probe_payload(status="not_requested")
    )
    return {
        "case_id": row["case_id"],
        "query_id": row["query_id"],
        "company": row["company"],
        "industry": row["industry"],
        "status": "needs_manual_probe",
        "review_priority": "must_review",
        "evidence_search_score": 0,
        "candidate_pages": [],
        "candidate_snippets": [],
        "matched_terms": [],
        "forbidden_matched_terms": [],
        "page_hits": [],
        "page_number_policy": PAGE_NUMBER_POLICY,
        **codex_probe_payload,
        "anchor_notes": f"原文搜索无法完成：{error}",
    }


def _searched_case(
    row: dict[str, Any],
    hits: list[PageSearchHit],
    codex_probe_hits: list[PageSearchHit],
    codex_probe_error: str | None,
) -> dict[str, Any]:
    effective_hits = hits or codex_probe_hits
    if not effective_hits:
        status = "needs_manual_probe"
        notes = "原文搜索没有找到可用候选页。"
    elif not hits and codex_probe_hits:
        status = _auto_anchor_status(row, codex_probe_hits[0])
        notes = "候选页来自 Codex deep probe；仍需 evidence-location pass 确认 expected_pages。"
    else:
        status = _auto_anchor_status(row, hits[0])
        notes = "候选页来自原文搜索；仍需 evidence-location pass 确认 expected_pages。"

    top_hit = effective_hits[0] if effective_hits else None
    codex_probe_payload = _codex_probe_payload(
        row=row,
        hits=codex_probe_hits,
        error=codex_probe_error,
    )
    return {
        "case_id": row["case_id"],
        "query_id": row["query_id"],
        "company": row["company"],
        "industry": row["industry"],
        "status": status,
        "review_priority": _initial_review_priority(status),
        "evidence_search_score": top_hit.score if top_hit else 0,
        "candidate_pages": [hit.page for hit in effective_hits],
        "candidate_snippets": [
            snippet for hit in effective_hits[:2] for snippet in hit.snippets[:1]
        ],
        "matched_terms": sorted({term for hit in effective_hits for term in hit.matched_terms}),
        "forbidden_matched_terms": sorted(
            {term for hit in effective_hits for term in hit.forbidden_matched_terms}
        ),
        "page_hits": [hit.to_json() for hit in effective_hits],
        "page_number_policy": PAGE_NUMBER_POLICY,
        **codex_probe_payload,
        "anchor_notes": notes,
    }


def _search_pages(
    *,
    row: dict[str, Any],
    pages: list[tuple[int, str]],
    top_pages: int,
) -> list[PageSearchHit]:
    return _search_pages_with_terms(
        row=row,
        pages=pages,
        terms=_search_terms(row),
        top_pages=top_pages,
    )


def _deep_probe_pages(
    *,
    row: dict[str, Any],
    pages: list[tuple[int, str]],
    top_pages: int,
) -> list[PageSearchHit]:
    return _search_pages_with_terms(
        row=row,
        pages=pages,
        terms=_deep_probe_terms(row),
        top_pages=top_pages,
    )


def _search_pages_with_terms(
    *,
    row: dict[str, Any],
    pages: list[tuple[int, str]],
    terms: list[str],
    top_pages: int,
) -> list[PageSearchHit]:
    forbidden_terms = _forbidden_terms(row)
    hits: list[PageSearchHit] = []
    if not terms:
        return hits
    for page_number, text in pages:
        feature_scores = _feature_scores(row=row, text=text, terms=terms)
        if feature_scores["term_score"] == 0:
            continue
        score = sum(feature_scores.values())
        matched_terms = _matched_terms(text, terms)
        forbidden_matched_terms = _matched_terms(text, forbidden_terms)
        snippets = _snippets(text=text, matched_terms=matched_terms)
        hits.append(
            PageSearchHit(
                page=page_number,
                score=score,
                confidence=_confidence(score),
                matched_terms=tuple(matched_terms),
                forbidden_matched_terms=tuple(forbidden_matched_terms),
                snippets=tuple(snippets),
                feature_scores=feature_scores,
            )
        )

    hits.sort(key=lambda hit: (-hit.score, -len(hit.matched_terms), hit.page))
    return hits[:top_pages]


def _should_run_codex_deep_probe(row: dict[str, Any]) -> bool:
    return (str(row.get("company")), str(row.get("query_id"))) in CODEX_DEEP_PROBE_TARGETS


def _deep_probe_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    terms.extend(DEEP_PROBE_TERMS.get(row["query_id"], ()))
    terms.extend(_search_terms(row))
    return _dedupe_terms(terms)


def _codex_probe_payload(
    *,
    row: dict[str, Any],
    hits: list[PageSearchHit],
    error: str | None,
) -> dict[str, Any]:
    if not _should_run_codex_deep_probe(row):
        return _empty_codex_probe_payload(status="not_requested")
    terms = _deep_probe_terms(row)
    if error is not None:
        return {
            **_empty_codex_probe_payload(status="codex_probe_error"),
            "codex_probe_terms": terms,
            "codex_probe_notes": f"Codex deep probe 失败：{error}",
        }
    if not hits:
        return {
            **_empty_codex_probe_payload(status="codex_probe_no_hit_after_deep_search"),
            "codex_probe_terms": terms,
            "codex_probe_notes": "Codex deep probe 未找到可用候选页；保留为非用户复核项。",
        }
    return {
        "codex_probe_status": "codex_probe_candidate_found",
        "codex_probe_pages": [hit.page for hit in hits],
        "codex_probe_snippets": [snippet for hit in hits[:2] for snippet in hit.snippets[:1]],
        "codex_probe_matched_terms": sorted({term for hit in hits for term in hit.matched_terms}),
        "codex_probe_score": hits[0].score,
        "codex_probe_terms": terms,
        "codex_probe_notes": "Codex deep probe 找到候选页；仍不能直接升格为 expected_pages。",
    }


def _empty_codex_probe_payload(*, status: str) -> dict[str, Any]:
    return {
        "codex_probe_status": status,
        "codex_probe_pages": [],
        "codex_probe_snippets": [],
        "codex_probe_matched_terms": [],
        "codex_probe_score": 0,
        "codex_probe_terms": [],
        "codex_probe_notes": "",
    }


def _feature_scores(*, row: dict[str, Any], text: str, terms: list[str]) -> dict[str, int]:
    matched = _matched_terms(text, terms)
    term_score = min(40, len(matched) * 5)
    numeric_score = _numeric_score(text, row)
    table_score = _table_score(text) if _expects_table(row) else 0
    section_score = _section_score(text) if _expects_section(row) else 0
    exact_query_score = 10 if _query_signal(row["query"], text) else 0
    return {
        "term_score": term_score,
        "numeric_score": numeric_score,
        "table_score": table_score,
        "section_score": section_score,
        "exact_query_score": exact_query_score,
    }


def _auto_anchor_status(row: dict[str, Any], hit: PageSearchHit) -> str:
    has_required_shape = True
    if _expects_table(row):
        has_required_shape = has_required_shape and hit.feature_scores["table_score"] > 0
    if _expects_section(row):
        has_required_shape = has_required_shape and (
            hit.feature_scores["section_score"] > 0 or hit.feature_scores["term_score"] >= 20
        )

    if hit.score >= 32 and len(hit.matched_terms) >= 2 and has_required_shape:
        return "auto_anchor_high_confidence"
    if hit.score > 0:
        return "auto_anchor_low_confidence"
    return "needs_manual_probe"


def _initial_review_priority(status: str) -> str:
    if status == "blocked_missing_raw":
        return "must_review"
    if status == "needs_manual_probe":
        return "codex_probe"
    return "later_anchor_confirmation"


def _summarize(cases: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "cases_total": len(cases),
        "auto_anchor_high_confidence": sum(
            case["status"] == "auto_anchor_high_confidence" for case in cases
        ),
        "auto_anchor_low_confidence": sum(
            case["status"] == "auto_anchor_low_confidence" for case in cases
        ),
        "needs_manual_probe": sum(case["status"] == "needs_manual_probe" for case in cases),
        "blocked_missing_raw": sum(case["status"] == "blocked_missing_raw" for case in cases),
    }


def _search_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    terms.extend(EXTRA_QUERY_TERMS.get(row["query_id"], ()))
    terms.extend(_query_terms(row["query"]))
    terms.extend(_area_terms(row.get("area", "")))
    for field_id in row.get("expected_answer_field_ids", []):
        terms.extend(FIELD_TERMS.get(field_id, ()))
        terms.extend(_field_id_terms(field_id))
    return _dedupe_terms(terms)


def _forbidden_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for failure in row.get("forbidden_failure_modes", []):
        terms.extend(_query_terms(failure))
    return _dedupe_terms(terms)


def _query_terms(text: str) -> list[str]:
    normalized = re.sub(r"[，。！？、；：:,.?;()\[\]（）/]+", " ", text)
    normalized = re.sub(
        r"(分别|是多少|如何|哪些|什么|是否|公司|本期|年报|披露|主要|为什么|以及|和|与|及)",
        " ",
        normalized,
    )
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9&./ -]{1,}", normalized)
    terms: list[str] = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", cleaned) and len(cleaned) > 8:
            terms.extend(_cjk_windows(cleaned))
        else:
            terms.append(cleaned)
    return terms


def _area_terms(area: str) -> list[str]:
    return [term for term in re.split(r"[,/ ]+", area) if len(term) >= 3]


def _field_id_terms(field_id: str) -> list[str]:
    return [part for part in field_id.split("_") if len(part) >= 3]


def _cjk_windows(text: str) -> list[str]:
    windows: list[str] = []
    for size in (4, 5, 6):
        for index in range(0, max(0, len(text) - size + 1)):
            token = text[index : index + size]
            if not _mostly_stop_chars(token):
                windows.append(token)
    return windows


def _mostly_stop_chars(token: str) -> bool:
    stop_chars = set("的是多少如何哪些什么以及分别公司本期年报")
    return sum(char in stop_chars for char in token) >= max(2, len(token) - 1)


def _dedupe_terms(terms: list[str]) -> list[str]:
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


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    normalized_text = _normalize(text)
    return [term for term in terms if _normalize(term) in normalized_text]


def _query_signal(query: str, text: str) -> bool:
    query_terms = _query_terms(query)
    matched = _matched_terms(text, query_terms)
    return len(matched) >= 2


def _numeric_score(text: str, row: dict[str, Any]) -> int:
    if row["expected_document_evidence_intent"] == "business_narrative":
        cap = 4
    else:
        cap = 12
    numeric_count = len(re.findall(r"\d+(?:[.,]\d+)?\s*(?:%|个百分点|百万元|亿元|元|TEU|吨|次)?", text))
    return min(cap, numeric_count // 2)


def _table_score(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_like = 0
    for line in lines:
        if len(re.findall(r"\d+(?:[.,]\d+)?", line)) >= 2 and re.search(
            r"[\u4e00-\u9fffA-Za-z]", line
        ):
            table_like += 1
    return min(16, table_like * 2)


def _section_score(text: str) -> int:
    normalized_text = _normalize(text)
    hits = sum(1 for term in SECTION_TERMS if _normalize(term) in normalized_text)
    return min(12, hits * 4)


def _expects_table(row: dict[str, Any]) -> bool:
    return row["primary_evidence_kind"] == "table_row" or "table_row" in row.get(
        "secondary_evidence_kinds", []
    )


def _expects_section(row: dict[str, Any]) -> bool:
    return row["primary_evidence_kind"] == "section_text" or "section_text" in row.get(
        "secondary_evidence_kinds", []
    )


def _snippets(*, text: str, matched_terms: list[str]) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    snippets: list[str] = []
    normalized_terms = [(_normalize(term), term) for term in matched_terms]
    for index, line in enumerate(lines):
        normalized_line = _normalize(line)
        if any(term_key in normalized_line for term_key, _term in normalized_terms):
            start = max(0, index - 1)
            end = min(len(lines), index + 2)
            snippet = " / ".join(lines[start:end])
            snippets.append(_truncate(snippet, max_chars=260))
        if len(snippets) >= 2:
            break
    return snippets


def _confidence(score: int) -> str:
    if score >= 32:
        return "high"
    if score >= 14:
        return "medium"
    return "low"


def _truncate(text: str, *, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 1]}…"


def _load_pages(source_path: Path) -> list[tuple[int, str]]:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_pages(source_path)
    if suffix in {".htm", ".html"}:
        return _load_html_pages(source_path)
    raise ValueError(f"Unsupported source suffix: {suffix}")


def _load_deep_pages(source_path: Path) -> list[tuple[int, str]]:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_deep_pages(source_path)
    if suffix in {".htm", ".html"}:
        return _load_html_pages(source_path)
    raise ValueError(f"Unsupported source suffix: {suffix}")


def _load_pdf_pages(source_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with _suppress_mupdf_messages():
        with fitz.open(source_path) as document:
            for index, page in enumerate(document, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append((index, text))
    return pages


def _load_pdf_deep_pages(source_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with _suppress_mupdf_messages():
        with fitz.open(source_path) as document:
            for index, page in enumerate(document, start=1):
                parts = _dedupe_text_parts(
                    [
                        _page_text(page, "text"),
                        _page_text(page, "blocks"),
                        _page_text(page, "words"),
                    ]
                )
                text = "\n".join(parts)
                if text.strip():
                    pages.append((index, text))
    return pages


def _page_text(page: fitz.Page, mode: str) -> str:
    try:
        payload = page.get_text(mode)
    except (RuntimeError, ValueError):
        return ""
    if mode == "blocks" and isinstance(payload, list):
        return "\n".join(str(block[4]) for block in payload if len(block) > 4)
    if mode == "words" and isinstance(payload, list):
        return " ".join(str(word[4]) for word in payload if len(word) > 4)
    return str(payload or "")


def _dedupe_text_parts(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        compact = re.sub(r"\s+", " ", part).strip()
        if not compact:
            continue
        key = compact[:500]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return deduped


def _load_html_pages(source_path: Path) -> list[tuple[int, str]]:
    parser = _HTMLTextExtractor()
    parser.feed(source_path.read_text(encoding="utf-8", errors="ignore"))
    text = parser.text
    return [(1, text)] if text else []


def _normalize(text: str) -> str:
    translated = text.translate(_TRAD_TO_SIMP)
    return re.sub(r"\s+", "", translated).casefold()


@contextmanager
def _suppress_mupdf_messages():
    display_errors = bool(fitz.TOOLS.mupdf_display_errors())
    display_warnings = bool(fitz.TOOLS.mupdf_display_warnings())
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    try:
        yield
    finally:
        fitz.TOOLS.mupdf_display_errors(display_errors)
        fitz.TOOLS.mupdf_display_warnings(display_warnings)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
