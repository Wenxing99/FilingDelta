from __future__ import annotations

import re

from filingdelta.schemas.fact_extraction import CandidatePageSelection
from filingdelta.schemas.filing import ParsedFiling, ParsedPage


_FIELD_NAMES = (
    "company_name",
    "fiscal_period",
    "unit",
    "revenue",
    "net_profit",
)

_FRONT_PAGE_PRIORITIES = {
    "company_name": 3,
    "fiscal_period": 4,
    "unit": 4,
    "revenue": 2,
    "net_profit": 2,
}

_SHARED_KEYWORDS = (
    "financial highlights",
    "financial summary",
    "summary of financial",
    "results announcement",
    "results release",
    "quarterly results",
    "interim results",
    "annual results",
    "主要财务数据",
    "财务摘要",
    "业绩摘要",
    "业绩公告",
    "财务概要",
    "condensed consolidated",
)

_FIELD_KEYWORDS = {
    "fiscal_period": (
        "year ended",
        "for the quarter ended",
        "for the nine months ended",
        "for the six months ended",
        "annual report",
        "interim report",
        "quarterly report",
        "截至",
        "年度报告",
        "半年度报告",
        "季度报告",
        "止年度",
    ),
    "unit": (
        "unit:",
        "currency:",
        "rmb",
        "cny",
        "hkd",
        "usd",
        "million",
        "thousand",
        "人民币",
        "百万元",
        "百萬元",
        "万元",
        "億",
        "亿元",
    ),
    "revenue": (
        "revenue",
        "revenues",
        "total revenues",
        "营业收入",
        "收入",
        "營業收入",
        "總收入",
    ),
    "net_profit": (
        "net profit",
        "net income",
        "profit attributable",
        "profit attributable to",
        "净利润",
        "淨利潤",
        "归属于",
        "應佔盈利",
        "本公司权益持有人应占",
        "本公司權益持有人應佔",
    ),
}


class CandidatePageLocator:
    def locate(self, parsed_filing: ParsedFiling) -> CandidatePageSelection:
        shared_pages = _collect_shared_pages(parsed_filing)
        field_pages = {
            field_name: _collect_field_pages(parsed_filing, field_name)
            for field_name in _FIELD_NAMES
        }
        return CandidatePageSelection(shared_pages=shared_pages, field_pages=field_pages)


def _collect_shared_pages(parsed_filing: ParsedFiling) -> list[int]:
    pages = parsed_filing.pages
    shared = [page.page_number for page in pages[: min(2, len(pages))]]
    shared.extend(_match_keyword_pages(parsed_filing, _SHARED_KEYWORDS, limit=4))
    return _dedupe_preserve_order(shared)


def _collect_field_pages(parsed_filing: ParsedFiling, field_name: str) -> list[int]:
    pages = parsed_filing.pages
    priorities = [page.page_number for page in pages[: _FRONT_PAGE_PRIORITIES.get(field_name, 0)]]
    keywords = _FIELD_KEYWORDS.get(field_name, ())
    priorities.extend(_match_keyword_pages(parsed_filing, keywords, limit=5))
    return _dedupe_preserve_order(priorities)


def _match_keyword_pages(
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


def _page_text(page: ParsedPage) -> str:
    return page.markdown or page.text


def _normalize_for_match(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    return normalized.lower()


def _dedupe_preserve_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
