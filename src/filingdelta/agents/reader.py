from __future__ import annotations

from llama_index.llms.openai import OpenAI

from filingdelta.core.config import Settings, get_settings
from filingdelta.ingestion.page_locators import CandidatePageLocator
from filingdelta.prompts.reader import READER_SUMMARY_PROMPT
from filingdelta.schemas.filing import FilingChunk, ParsedFiling
from filingdelta.schemas.workflow import (
    ReaderDraftResult,
    SummaryDraftPoint,
    SummaryDraftSection,
)


_SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "财务表现摘要": (
        "financial highlights",
        "financial summary",
        "主要财务数据",
        "财务摘要",
        "营业收入",
        "收入",
        "净利润",
        "毛利",
        "每股盈利",
        "profit attributable",
        "operating profit",
    ),
    "股息 / 分红": (
        "dividend",
        "股息",
        "分红",
        "派息",
        "现金股息",
        "末期股息",
        "每股现金分红",
    ),
    "经营数据 / 关键指标": (
        "总资产",
        "客户存款",
        "贷款和垫款",
        "经营活动产生的现金流量净额",
        "月活跃账户数",
        "订阅会员数",
        "不良贷款率",
        "拨备覆盖率",
        "active accounts",
        "arpu",
        "revenue growth",
    ),
    "战略与展望": (
        "strategy",
        "strategic",
        "outlook",
        "plan",
        "目标",
        "战略",
        "将继续",
        "计划",
        "展望",
        "愿景",
        "best value",
    ),
    "股东情况": (
        "shareholder",
        "股东",
        "普通股股东总数",
        "a股股东总数",
        "h股股东总数",
    ),
    "风险与资产质量": (
        "不良贷款",
        "贷款损失准备",
        "拨备覆盖率",
        "风险",
        "资产质量",
        "npl",
        "impairment",
        "allowance",
    ),
    "业务回顾": (
        "business review",
        "业务回顾",
        "广告",
        "游戏",
        "视频号",
        "云业务",
        "金融科技",
        "wechat",
        "qq",
        "mini program",
        "cloud",
    ),
    "用户与产品数据": (
        "月活跃账户数",
        "月活跃",
        "subscription",
        "会员数",
        "wechat",
        "qq",
        "视频号",
        "小游戏",
        "mini games",
        "video accounts",
        "users",
    ),
    "可持续发展": (
        "esg",
        "可持续",
        "公益",
        "碳中和",
        "慈善",
        "绿色电力",
        "sustainability",
        "climate",
    ),
    "公司治理": (
        "corporate governance",
        "governance",
        "董事会",
        "监事会",
        "审计",
        "委员会",
        "董事",
        "治理",
    ),
}

_SECTION_TAXONOMY = tuple(_SECTION_KEYWORDS.keys())
_SECTION_ORDER = {title: index for index, title in enumerate(_SECTION_TAXONOMY)}


class ReaderAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = OpenAI(
            model=self._settings.filingdelta_llm_model,
            temperature=0,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
            strict=True,
        )

    async def read(
        self,
        parsed_filing: ParsedFiling,
        chunks: list[FilingChunk],
    ) -> ReaderDraftResult:
        page_numbers = _select_summary_pages(parsed_filing)
        page_context = _build_page_context(parsed_filing, page_numbers)
        chunk_count = len(chunks)

        result = await self._llm.astructured_predict(
            ReaderDraftResult,
            READER_SUMMARY_PROMPT,
            company_name=parsed_filing.document.company_name,
            ticker=parsed_filing.document.ticker or "",
            market=parsed_filing.document.market.value,
            doc_type=parsed_filing.document.doc_type.value,
            fiscal_period=parsed_filing.document.fiscal_period or "",
            section_taxonomy="\n".join(f"- {section_title}" for section_title in _SECTION_TAXONOMY),
            page_numbers=", ".join(str(page_number) for page_number in page_numbers) or "none",
            page_context=(
                f"[Document metadata]\n"
                f"- total_pages: {parsed_filing.document.total_pages}\n"
                f"- chunk_count: {chunk_count}\n\n"
                f"{page_context}"
            ),
        )
        result.overview = _normalize_overview(result.overview)
        result.sections = _normalize_sections(result.sections)
        return result


def _select_summary_pages(parsed_filing: ParsedFiling) -> list[int]:
    locator = CandidatePageLocator()
    selection = locator.locate(parsed_filing)

    page_numbers = [
        page.page_number for page in parsed_filing.pages[: min(10, len(parsed_filing.pages))]
    ]
    page_numbers.extend(selection.shared_pages)
    page_numbers.extend(selection.pages_for("revenue")[:3])
    page_numbers.extend(selection.pages_for("net_profit")[:3])

    for keywords in _SECTION_KEYWORDS.values():
        page_numbers.extend(_match_section_pages(parsed_filing, keywords, limit=2))

    deduped = _dedupe_preserve_order(page_numbers)
    return deduped[:14]


def _match_section_pages(
    parsed_filing: ParsedFiling,
    keywords: tuple[str, ...],
    *,
    limit: int,
) -> list[int]:
    matched_pages: list[int] = []
    normalized_keywords = [_normalize_for_match(keyword) for keyword in keywords]

    for page in parsed_filing.pages:
        page_text = _normalize_for_match(_page_text(page))
        if any(keyword and keyword in page_text for keyword in normalized_keywords):
            matched_pages.append(page.page_number)
            if len(matched_pages) >= limit:
                break

    return matched_pages


def _build_page_context(parsed_filing: ParsedFiling, page_numbers: list[int]) -> str:
    page_lookup = {page.page_number: page for page in parsed_filing.pages}
    parts: list[str] = []
    for page_number in page_numbers:
        page = page_lookup.get(page_number)
        if not page:
            continue
        page_text = _page_text(page).strip()
        if not page_text:
            continue
        parts.append(f"[Page {page_number}]\n{_truncate_text(page_text)}")
    return "\n\n".join(parts)


def _normalize_overview(overview: SummaryDraftPoint | None) -> SummaryDraftPoint | None:
    if overview is None:
        return None
    if not _normalize_text_key(overview.text):
        return None
    return overview


def _normalize_sections(sections: list[SummaryDraftSection]) -> list[SummaryDraftSection]:
    merged: dict[str, list[SummaryDraftPoint]] = {}

    for section in sections:
        normalized_title = _normalize_section_title(section.title)
        if normalized_title not in _SECTION_ORDER:
            continue
        merged.setdefault(normalized_title, []).extend(section.points)

    normalized_sections: list[SummaryDraftSection] = []
    for title in _SECTION_TAXONOMY:
        points = _dedupe_summary_points(merged.get(title, []))[:5]
        if not points:
            continue
        normalized_sections.append(SummaryDraftSection(title=title, points=points))

    return normalized_sections


def _dedupe_summary_points(points: list[SummaryDraftPoint]) -> list[SummaryDraftPoint]:
    seen: set[str] = set()
    deduped: list[SummaryDraftPoint] = []
    for point in points:
        text_key = _normalize_text_key(point.text)
        if not text_key or text_key in seen:
            continue
        seen.add(text_key)
        deduped.append(point)
    return deduped


def _normalize_section_title(title: str) -> str:
    title_key = _normalize_text_key(title)
    for candidate in _SECTION_TAXONOMY:
        if title_key == _normalize_text_key(candidate):
            return candidate
    return title.strip()


def _page_text(page) -> str:
    return page.markdown or page.text


def _truncate_text(text: str, limit: int = 2600) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _normalize_for_match(text: str) -> str:
    return "".join(text.lower().split())


def _dedupe_preserve_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _normalize_text_key(text: str) -> str:
    return " ".join(text.lower().split())
