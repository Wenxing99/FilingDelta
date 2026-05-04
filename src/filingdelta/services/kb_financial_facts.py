from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from uuid import uuid4

from filingdelta.core.config import REPO_ROOT
from filingdelta.financial_facts.catalog import CANONICAL_METRICS
from filingdelta.financial_facts.query import FinancialFactsQueryService, FinancialFactTopKResult
from filingdelta.schemas.chat import ChatAnswer, ChatAnswerSection


KB_FINANCIAL_FACTS_RETRIEVAL_MODE = "kb_financial_facts"
SUPPORTED_FACT_YEAR = 2025
DEFAULT_FACT_DB_PATH = REPO_ROOT / "data" / "indexes" / "financial_facts.sqlite"


@dataclass(frozen=True)
class KbMetricRankQuestion:
    metric_id: str | None
    fiscal_year: int | None
    limit: int | None
    recognized: bool
    unsupported_reason: str | None = None


class KbFinancialFactsChatService:
    def __init__(
        self,
        query_service: FinancialFactsQueryService | None = None,
    ) -> None:
        self._query_service = query_service or FinancialFactsQueryService(DEFAULT_FACT_DB_PATH)

    def answer_if_supported(
        self,
        *,
        document_id: str,
        session_id: str | None,
        question: str,
    ) -> ChatAnswer | None:
        parsed = parse_kb_metric_rank_question(question)
        if not parsed.recognized:
            return None
        active_session_id = (session_id or f"{document_id}-{uuid4()}").strip()
        if parsed.unsupported_reason is not None:
            return _unsupported_answer(
                document_id=document_id,
                session_id=active_session_id,
                question=question,
                reason=parsed.unsupported_reason,
            )
        assert parsed.metric_id is not None
        assert parsed.fiscal_year is not None
        assert parsed.limit is not None
        result = self._query_service.top_metric_by_year(
            metric_id=parsed.metric_id,
            fiscal_year=parsed.fiscal_year,
            limit=parsed.limit,
        )
        return _answer_from_result(
            document_id=document_id,
            session_id=active_session_id,
            question=question,
            result=result,
        )


def parse_kb_metric_rank_question(question: str) -> KbMetricRankQuestion:
    normalized = _normalize_question(question)
    metric_id = _detect_metric_id(normalized)
    if not _looks_like_metric_rank_question(normalized, metric_id=metric_id):
        return KbMetricRankQuestion(metric_id=None, fiscal_year=None, limit=None, recognized=False)

    fiscal_year = _detect_fiscal_year(normalized)
    limit = _detect_limit(normalized)
    if limit is None and metric_id is not None:
        limit = _detect_implicit_top1_limit(normalized)
    if metric_id is None:
        return KbMetricRankQuestion(
            metric_id=None,
            fiscal_year=fiscal_year,
            limit=limit,
            recognized=True,
            unsupported_reason="当前结构化事实库只支持营业收入、归母净利润、总资产和总负债的 TopK 查询。",
        )
    if fiscal_year is None:
        return KbMetricRankQuestion(
            metric_id=metric_id,
            fiscal_year=None,
            limit=limit,
            recognized=True,
            unsupported_reason="请明确查询年份；当前最小事实库只支持 2025 年。",
        )
    if fiscal_year != SUPPORTED_FACT_YEAR:
        return KbMetricRankQuestion(
            metric_id=metric_id,
            fiscal_year=fiscal_year,
            limit=limit,
            recognized=True,
            unsupported_reason=f"当前最小事实库只支持 {SUPPORTED_FACT_YEAR} 年，不支持 {fiscal_year} 年。",
        )
    if limit is None:
        return KbMetricRankQuestion(
            metric_id=metric_id,
            fiscal_year=fiscal_year,
            limit=None,
            recognized=True,
            unsupported_reason="请明确 TopK 数量，例如“三家公司”或“Top3”。",
        )
    return KbMetricRankQuestion(
        metric_id=metric_id,
        fiscal_year=fiscal_year,
        limit=limit,
        recognized=True,
    )


def _answer_from_result(
    *,
    document_id: str,
    session_id: str,
    question: str,
    result: FinancialFactTopKResult,
) -> ChatAnswer:
    if result.status in {"unsupported", "unavailable"} or not result.facts:
        return ChatAnswer(
            document_id=document_id,
            session_id=session_id,
            question=question,
            answer=_result_unavailable_text(result),
            route="unsupported",
            sections=[
                ChatAnswerSection(
                    section_type="analysis_and_limits",
                    title="查询边界",
                    items=result.notes,
                )
            ],
            citations=[],
            retrieval_mode=KB_FINANCIAL_FACTS_RETRIEVAL_MODE,
        )

    metric_label = CANONICAL_METRICS[result.metric_id].label
    lines = [
        f"{result.fiscal_year} 年当前结构化事实库中，{metric_label} Top {result.limit} 如下：",
        "",
        "| 排名 | 公司 | 数值 | 来源 |",
        "|---:|---|---:|---|",
    ]
    for rank, fact in enumerate(result.facts, start=1):
        company = fact.company_name or fact.document_id
        value = _format_normalized_value(fact.normalized_value, fact.normalized_unit)
        source = _format_fact_source(fact)
        lines.append(f"| {rank} | {company} | {value} | {source} |")
    lines.extend(["", _coverage_text(result)])
    if result.status == "partial":
        lines.append("注意：当前 KB 中可验证样本不足请求数量，因此这是部分结果。")

    section_items = [
        f"candidate_count={result.summary.candidate_count}",
        f"verified_annual_candidates={result.summary.verified_annual_candidates}",
        f"after_citation_filter={result.summary.after_citation_filter}",
        f"after_company_dedupe={result.summary.after_company_dedupe}",
        "跨公司页码和 quote 已在表格中列出；本轮不生成可点击 citation chips，避免误跳到当前文档 viewer。",
    ]
    return ChatAnswer(
        document_id=document_id,
        session_id=session_id,
        question=question,
        answer="\n".join(lines),
        route="document_only",
        sections=[
            ChatAnswerSection(
                section_type="analysis_and_limits",
                title="覆盖范围与边界",
                items=section_items,
            )
        ],
        citations=[],
        retrieval_mode=KB_FINANCIAL_FACTS_RETRIEVAL_MODE,
    )


def _unsupported_answer(
    *,
    document_id: str,
    session_id: str,
    question: str,
    reason: str,
) -> ChatAnswer:
    return ChatAnswer(
        document_id=document_id,
        session_id=session_id,
        question=question,
        answer=(
            "这个问题被识别为跨公司结构化财务指标排序，但当前最小事实库还不能可靠回答。\n\n"
            f"原因：{reason}\n\n"
            "为了避免用普通 RAG 相似度回答可排序财务指标，本问题不会回退到文档检索。"
        ),
        route="unsupported",
        sections=[
            ChatAnswerSection(
                section_type="analysis_and_limits",
                title="查询边界",
                items=[reason, "可排序财务指标必须走结构化事实库，不用向量相似度排序。"],
            )
        ],
        citations=[],
        retrieval_mode=KB_FINANCIAL_FACTS_RETRIEVAL_MODE,
    )


def _result_unavailable_text(result: FinancialFactTopKResult) -> str:
    metric_label = CANONICAL_METRICS.get(result.metric_id)
    label = metric_label.label if metric_label else result.metric_id
    reasons = "；".join(result.notes) if result.notes else "当前事实库没有足够 verified 数据。"
    return (
        f"这个问题需要查询结构化事实库中的 {label} Top {result.limit}，"
        "但当前 KB 还不能可靠返回结果。\n\n"
        f"原因：{reasons}\n\n"
        "为了避免用普通 RAG 相似度回答可排序财务指标，本问题不会回退到文档检索。"
    )


def _coverage_text(result: FinancialFactTopKResult) -> str:
    return (
        "覆盖口径：仅使用当前 SQLite `financial_facts` 中 "
        "verified、年度、带 page/quote citation、且单位可比的事实；"
        f"候选 {result.summary.candidate_count} 条，"
        f"去重后 {result.summary.after_company_dedupe} 家，"
        f"返回 {result.summary.returned_rows} 行。"
    )


def _format_normalized_value(value: float | None, normalized_unit: str | None) -> str:
    if value is None:
        return "N/A"
    if normalized_unit == "CNY":
        return f"{value / 100_000_000:.2f} 亿元"
    if normalized_unit:
        return f"{value:,.0f} {normalized_unit}"
    return f"{value:,.0f}"


def _format_fact_source(fact) -> str:
    page = f"第 {fact.evidence_page} 页" if fact.evidence_page is not None else "页码缺失"
    quote = _format_evidence_quote(fact.evidence_quote)
    return f"{fact.document_id}，{page}：“{quote}”"


def _format_evidence_quote(quote: str) -> str:
    return re.sub(r"\s+", " ", quote.strip()).replace("|", "\\|")


def _looks_like_metric_rank_question(normalized: str, *, metric_id: str | None) -> bool:
    if not _has_any_rank_signal(normalized):
        return False
    if _has_kb_scope(normalized) or _has_cross_company_scope(normalized):
        return True
    if not _has_unambiguous_topk_signal(normalized):
        return False
    return metric_id is not None or _has_unsupported_exact_metric_term(normalized)


def _detect_metric_id(normalized: str) -> str | None:
    for metric_id, aliases in _SUPPORTED_METRIC_ALIASES:
        if any(alias in normalized for alias in aliases):
            return metric_id
    return None


def _detect_fiscal_year(normalized: str) -> int | None:
    match = re.search(r"(20\d{2})年?", normalized)
    if match:
        return int(match.group(1))
    return None


def _detect_limit(normalized: str) -> int | None:
    patterns = (
        r"top\s*(\d+)",
        r"前\s*(\d+)",
        r"哪\s*(\d+)\s*家",
        r"(\d+)\s*家",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    chinese_match = re.search(r"(?:哪|前)?([一二两三四五六七八九十])家", normalized)
    if chinese_match:
        return _CHINESE_NUMBER_MAP.get(chinese_match.group(1))
    chinese_rank_match = re.search(r"前([一二两三四五六七八九十])(?:名|位)?", normalized)
    if chinese_rank_match:
        return _CHINESE_NUMBER_MAP.get(chinese_rank_match.group(1))
    return None


def _detect_implicit_top1_limit(normalized: str) -> int | None:
    if not (_has_kb_scope(normalized) or _has_cross_company_scope(normalized)):
        return None
    if any(term in normalized for term in _IMPLICIT_TOP1_RANK_TERMS):
        return 1
    return None


def _normalize_question(question: str) -> str:
    return "".join(unicodedata.normalize("NFKC", question).casefold().split())


def _has_any_rank_signal(normalized: str) -> bool:
    if any(term in normalized for term in ("最高", "最大", "最多", "top", "topk", "排名", "排行")):
        return True
    return bool(re.search(r"前(?:\d+|[一二两三四五六七八九十])", normalized))


def _has_unambiguous_topk_signal(normalized: str) -> bool:
    if "topk" in normalized or re.search(r"top\d+", normalized, flags=re.IGNORECASE):
        return True
    if re.search(r"前(?:\d+|[一二两三四五六七八九十])(?:名|位)?", normalized):
        return True
    return bool(re.search(r"[一二两三四五六七八九十\d]+家(?:公司|企业)?", normalized))


def _has_cross_company_scope(normalized: str) -> bool:
    return any(term in normalized for term in _CROSS_COMPANY_SCOPE_TERMS)


def _has_kb_scope(normalized: str) -> bool:
    return any(term in normalized for term in _KB_SCOPE_TERMS)


def _has_unsupported_exact_metric_term(normalized: str) -> bool:
    return any(term in normalized for term in _UNSUPPORTED_EXACT_METRIC_TERMS)


def _build_supported_metric_aliases() -> tuple[tuple[str, tuple[str, ...]], ...]:
    extra_aliases = {
        "revenue": ("revenue",),
        "net_profit_attributable": (
            "net profit attributable",
            "attributable net profit",
        ),
        "total_assets": ("total assets",),
        "total_liabilities": ("total liabilities",),
    }
    priority = (
        "net_profit_attributable",
        "total_liabilities",
        "total_assets",
        "revenue",
    )
    result: list[tuple[str, tuple[str, ...]]] = []
    for metric_id in priority:
        metric = CANONICAL_METRICS[metric_id]
        raw_aliases = (
            metric.metric_id,
            metric.label,
            *metric.aliases,
            *extra_aliases.get(metric_id, ()),
        )
        normalized_aliases = sorted(
            {
                _normalize_question(alias)
                for alias in raw_aliases
                if _normalize_question(alias)
            },
            key=len,
            reverse=True,
        )
        result.append((metric_id, tuple(normalized_aliases)))
    return tuple(result)


_CHINESE_NUMBER_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

_KB_SCOPE_TERMS = tuple(
    _normalize_question(term)
    for term in (
        "当前 KB",
        "当前KB",
        "KB",
        "current KB",
        "current-KB",
        "current knowledge base",
        "事实库",
        "事實庫",
        "结构化事实库",
        "結構化事實庫",
        "fact store",
        "fact database",
        "fact db",
        "knowledge base",
    )
)

_CROSS_COMPANY_SCOPE_TERMS = tuple(
    _normalize_question(term)
    for term in (
        "哪家公司",
        "哪些公司",
        "哪几家公司",
        "哪三家公司",
        "哪家企业",
        "哪些企业",
        "哪几家企业",
        "哪三家企业",
        "哪个公司",
        "哪些个公司",
        "几家公司",
        "三家公司",
        "3家公司",
        "几家企业",
        "三家企业",
        "3家企业",
        "公司排名",
        "企业排名",
        "which companies",
        "which company",
        "top companies",
        "top company",
    )
)

_UNSUPPORTED_EXACT_METRIC_TERMS = tuple(
    _normalize_question(term)
    for term in (
        "经营现金流",
        "经营活动现金流",
        "经营活动产生的现金流量净额",
        "经营活动现金流净额",
        "operating cash flow",
        "cash flow from operations",
        "net cash generated from operating activities",
        "roe",
        "roae",
        "gross margin",
        "毛利率",
        "资本开支",
        "capex",
    )
)

_IMPLICIT_TOP1_RANK_TERMS = tuple(
    _normalize_question(term)
    for term in (
        "最高",
        "最大",
        "最多",
        "highest",
        "largest",
        "biggest",
    )
)

_SUPPORTED_METRIC_ALIASES = _build_supported_metric_aliases()
