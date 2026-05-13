from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import unicodedata

from filingdelta.schemas.fact_extraction import CandidatePageSelection
from filingdelta.schemas.filing import ParsedFiling, ParsedPage


_FIELD_NAMES = (
    "company_name",
    "fiscal_period",
    "unit",
    "revenue",
    "net_profit",
    "total_assets",
    "total_liabilities",
    "roe",
)

_METRIC_FIELD_NAMES = (
    "revenue",
    "net_profit",
    "total_assets",
    "total_liabilities",
    "roe",
)

_BALANCE_SHEET_FIELDS = {"total_assets", "total_liabilities"}

_FRONT_PAGE_PRIORITIES = {
    "company_name": 3,
    "fiscal_period": 4,
    "unit": 4,
    "revenue": 2,
    "net_profit": 2,
    "total_assets": 2,
    "total_liabilities": 2,
    "roe": 2,
}

_SHARED_KEYWORDS = (
    "financial highlights",
    "financial summary",
    "summary of financial",
    "selected financial",
    "results announcement",
    "results release",
    "quarterly results",
    "interim results",
    "annual results",
    "condensed consolidated",
    "\u4e3b\u8981\u8d22\u52a1\u6570\u636e",
    "\u4e3b\u8981\u8ca1\u52d9\u6578\u64da",
    "\u8d22\u52a1\u6458\u8981",
    "\u8ca1\u52d9\u6458\u8981",
    "\u4e1a\u7ee9\u6458\u8981",
    "\u696d\u7e3e\u6458\u8981",
    "\u4e1a\u7ee9\u516c\u544a",
    "\u696d\u7e3e\u516c\u544a",
    "\u8d22\u52a1\u6982\u8981",
    "\u8ca1\u52d9\u6982\u8981",
)

_FIELD_KEYWORDS = {
    "fiscal_period": (
        "year ended",
        "for the quarter ended",
        "for the nine months ended",
        "for the six months ended",
        "fiscal year",
        "annual report",
        "interim report",
        "quarterly report",
        "\u622a\u81f3",
        "\u5e74\u5ea6\u62a5\u544a",
        "\u5e74\u5ea6\u5831\u544a",
        "\u534a\u5e74\u5ea6\u62a5\u544a",
        "\u534a\u5e74\u5ea6\u5831\u544a",
        "\u5b63\u5ea6\u62a5\u544a",
        "\u5b63\u5ea6\u5831\u544a",
        "\u6b62\u5e74\u5ea6",
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
        "\u4eba\u6c11\u5e01",
        "\u4eba\u6c11\u5e63",
        "\u767e\u4e07\u5143",
        "\u767e\u842c\u5143",
        "\u4e07\u5143",
        "\u5104",
        "\u4ebf\u5143",
    ),
    "revenue": (
        "revenue",
        "revenues",
        "total revenues",
        "\u8425\u4e1a\u6536\u5165",
        "\u71df\u696d\u6536\u5165",
        "\u6536\u5165",
        "\u603b\u6536\u5165",
        "\u7e3d\u6536\u5165",
    ),
    "net_profit": (
        "net profit",
        "net income",
        "profit attributable",
        "profit attributable to",
        "net income attributable",
        "\u51c0\u5229\u6da6",
        "\u6de8\u5229\u6f64",
        "\u5f52\u5c5e\u4e8e",
        "\u6b78\u5c6c\u65bc",
        "\u5e94\u5360\u76c8\u5229",
        "\u61c9\u4f54\u76c8\u5229",
        "\u672c\u516c\u53f8\u6743\u76ca\u6301\u6709\u4eba\u5e94\u5360",
        "\u672c\u516c\u53f8\u6b0a\u76ca\u6301\u6709\u4eba\u61c9\u4f54",
    ),
    "total_assets": (
        "total assets",
        "assets total",
        "\u603b\u8d44\u4ea7",
        "\u7e3d\u8cc7\u7522",
        "\u8d44\u4ea7\u603b\u8ba1",
        "\u8cc7\u7522\u7e3d\u8a08",
        "\u8d44\u4ea7\u5408\u8ba1",
        "\u8cc7\u7522\u5408\u8a08",
    ),
    "total_liabilities": (
        "total liabilities",
        "liabilities total",
        "\u603b\u8d1f\u503a",
        "\u7e3d\u8ca0\u50b5",
        "\u8d1f\u503a\u5408\u8ba1",
        "\u8ca0\u50b5\u5408\u8a08",
        "\u8d1f\u503a\u603b\u8ba1",
        "\u8ca0\u50b5\u7e3d\u8a08",
    ),
    "roe": (
        "return on equity",
        "weighted average return on equity",
        "roe",
        "roae",
        "\u51c0\u8d44\u4ea7\u6536\u76ca\u7387",
        "\u6de8\u8cc7\u7522\u6536\u76ca\u7387",
        "\u52a0\u6743\u5e73\u5747\u51c0\u8d44\u4ea7\u6536\u76ca\u7387",
        "\u52a0\u6b0a\u5e73\u5747\u6de8\u8cc7\u7522\u6536\u76ca\u7387",
        "\u6263\u9664\u975e\u7ecf\u5e38\u6027\u635f\u76ca\u540e",
        "\u6263\u9664\u975e\u7d93\u5e38\u6027\u640d\u76ca\u5f8c",
        "\u666e\u901a\u80a1\u80a1\u4e1c",
        "\u666e\u901a\u80a1\u80a1\u6771",
        "\u5e74\u5316",
    ),
}

_UNIT_HINT_KEYWORDS = (
    "rmb",
    "cny",
    "hkd",
    "usd",
    "million",
    "thousand",
    "\u4eba\u6c11\u5e01",
    "\u4eba\u6c11\u5e63",
    "\u767e\u4e07\u5143",
    "\u767e\u842c\u5143",
    "\u4e07\u5143",
    "\u5104",
    "\u4ebf\u5143",
)

_PERIOD_HINT_KEYWORDS = (
    "year ended",
    "quarter ended",
    "six months ended",
    "nine months ended",
    "fiscal year",
    "annual report",
    "interim report",
    "quarterly report",
    "\u622a\u81f3",
    "\u5e74\u5ea6\u62a5\u544a",
    "\u5e74\u5ea6\u5831\u544a",
    "\u534a\u5e74\u5ea6\u62a5\u544a",
    "\u534a\u5e74\u5ea6\u5831\u544a",
    "\u5b63\u5ea6\u62a5\u544a",
    "\u5b63\u5ea6\u5831\u544a",
)

_FINANCIAL_SUMMARY_CONTEXT_KEYWORDS = (
    "financial highlights",
    "financial summary",
    "selected financial",
    "key financial",
    "\u4e3b\u8981\u8d22\u52a1",
    "\u4e3b\u8981\u8ca1\u52d9",
    "\u8d22\u52a1\u6458\u8981",
    "\u8ca1\u52d9\u6458\u8981",
)

_OPERATING_RESULTS_CONTEXT_KEYWORDS = (
    "operating results",
    "consolidated results",
    "results of operations",
    "\u7ecf\u8425\u4e1a\u7ee9",
    "\u7d93\u71df\u696d\u7e3e",
)

_FINANCIAL_METRIC_CONTEXT_KEYWORDS = (
    *_FINANCIAL_SUMMARY_CONTEXT_KEYWORDS,
    *_OPERATING_RESULTS_CONTEXT_KEYWORDS,
)

_BALANCE_SHEET_CONTEXT_KEYWORDS = (
    "balance sheet",
    "statement of financial position",
    "statements of financial position",
    "\u8d44\u4ea7\u8d1f\u503a\u8868",
    "\u8cc7\u7522\u8ca0\u50b5\u8868",
    "\u8d22\u52a1\u72b6\u51b5\u8868",
    "\u8ca1\u52d9\u72c0\u6cc1\u8868",
)


@dataclass(frozen=True)
class LocatorDebugEntry:
    page_number: int
    score: int
    matched_terms: tuple[str, ...]
    snippet: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class LocatorTraceResult:
    selection: CandidatePageSelection
    shared_debug: list[LocatorDebugEntry]
    field_debug: dict[str, list[LocatorDebugEntry]]


@dataclass(frozen=True)
class _PageMatchContext:
    page: ParsedPage
    raw_text: str
    normalized_text: str
    normalized_to_raw: tuple[int, ...]


class CandidatePageLocator:
    def locate(self, parsed_filing: ParsedFiling) -> CandidatePageSelection:
        return self.locate_with_trace(parsed_filing).selection

    def locate_with_trace(self, parsed_filing: ParsedFiling) -> LocatorTraceResult:
        contexts = [_build_page_match_context(page) for page in parsed_filing.pages]

        shared_debug = _rank_debug_entries(
            _score_shared_page(context, _first_page_numbers(parsed_filing, limit=2))
            for context in contexts
        )
        shared_pages = _dedupe_preserve_order(
            [
                *_first_page_numbers(parsed_filing, limit=2),
                *[entry.page_number for entry in shared_debug[:4]],
            ]
        )
        headline_summary_pages = set(_first_page_numbers(parsed_filing, limit=30))

        field_debug: dict[str, list[LocatorDebugEntry]] = {}
        field_pages: dict[str, list[int]] = {}
        for field_name in _FIELD_NAMES:
            front_pages = _first_page_numbers(
                parsed_filing,
                limit=_FRONT_PAGE_PRIORITIES.get(field_name, 0),
            )
            entries = _rank_debug_entries(
                _score_field_page(
                    context,
                    field_name,
                    front_pages,
                    headline_summary_pages,
                )
                for context in contexts
            )
            field_debug[field_name] = entries
            field_pages[field_name] = _dedupe_preserve_order(
                [*front_pages, *[entry.page_number for entry in entries[:5]]]
            )

        selection = CandidatePageSelection(
            shared_pages=shared_pages,
            field_pages=field_pages,
        )
        return LocatorTraceResult(
            selection=selection,
            shared_debug=shared_debug,
            field_debug=field_debug,
        )


def select_summary_pages(
    parsed_filing: ParsedFiling,
    *,
    section_keyword_groups: Iterable[tuple[str, ...]] = (),
) -> list[int]:
    selection = CandidatePageLocator().locate(parsed_filing)

    page_numbers = _first_page_numbers(parsed_filing, limit=10)
    page_numbers.extend(selection.shared_pages)
    page_numbers.extend(selection.pages_for("revenue")[:3])
    page_numbers.extend(selection.pages_for("net_profit")[:3])

    for keywords in section_keyword_groups:
        page_numbers.extend(_match_keyword_pages(parsed_filing, keywords, limit=2))

    return _dedupe_preserve_order(page_numbers)[:14]


def _score_shared_page(
    context: _PageMatchContext,
    front_pages: list[int],
) -> LocatorDebugEntry | None:
    score = 0
    matched_terms: list[str] = []
    reasons: list[str] = []

    if context.page.page_number in front_pages:
        score += 3
        reasons.append("front_page_boost")

    shared_terms = _matched_terms(context, _SHARED_KEYWORDS)
    if shared_terms:
        score += 8 * len(shared_terms)
        matched_terms.extend(shared_terms)
        reasons.append("shared_summary_keywords")

    financial_context_terms = _matched_terms(context, _FINANCIAL_METRIC_CONTEXT_KEYWORDS)
    if financial_context_terms:
        score += 3
        matched_terms.extend(financial_context_terms)
        reasons.append("financial_metric_context")

    unit_terms = _matched_terms(context, _UNIT_HINT_KEYWORDS)
    if unit_terms:
        score += 1
        matched_terms.extend(unit_terms[:2])
        reasons.append("unit_hints")

    period_terms = _matched_terms(context, _PERIOD_HINT_KEYWORDS)
    if period_terms:
        score += 1
        matched_terms.extend(period_terms[:2])
        reasons.append("period_hints")

    if score <= 0:
        return None
    return _build_debug_entry(context, score, matched_terms, reasons)


def _score_field_page(
    context: _PageMatchContext,
    field_name: str,
    front_pages: list[int],
    headline_summary_pages: set[int],
) -> LocatorDebugEntry | None:
    score = 0
    matched_terms: list[str] = []
    reasons: list[str] = []
    is_front_page = context.page.page_number in front_pages

    if is_front_page:
        score += 2
        reasons.append("front_page_boost")

    field_terms = _matched_terms(context, _keywords_for_field(field_name))
    if field_terms:
        score += 10 * len(field_terms)
        matched_terms.extend(field_terms)
        reasons.append("field_keywords")

    financial_summary_context_terms: list[str] = []
    operating_results_context_terms: list[str] = []
    paired_terms: list[str] = []
    unit_terms: list[str] = []
    period_terms: list[str] = []

    if field_name in _METRIC_FIELD_NAMES and field_terms:
        financial_summary_context_terms = _matched_terms(
            context,
            _FINANCIAL_SUMMARY_CONTEXT_KEYWORDS,
        )
        operating_results_context_terms = _matched_terms(
            context,
            _OPERATING_RESULTS_CONTEXT_KEYWORDS,
        )
        financial_context_terms = _dedupe_preserve_order(
            [*financial_summary_context_terms, *operating_results_context_terms]
        )
        if financial_context_terms:
            score += 4
            matched_terms.extend(financial_context_terms[:2])
            reasons.append("financial_metric_context")

    if field_name in _BALANCE_SHEET_FIELDS and field_terms:
        balance_context_terms = _matched_terms(context, _BALANCE_SHEET_CONTEXT_KEYWORDS)
        if balance_context_terms:
            score += 5
            matched_terms.extend(balance_context_terms[:2])
            reasons.append("balance_sheet_context")

        paired_field = (
            "total_liabilities" if field_name == "total_assets" else "total_assets"
        )
        paired_terms = _matched_terms(context, _FIELD_KEYWORDS[paired_field])
        if paired_terms:
            score += 3
            matched_terms.extend(paired_terms[:2])
            reasons.append("balance_sheet_paired_terms")

    if field_name in _METRIC_FIELD_NAMES and field_terms:
        unit_terms = _matched_terms(context, _UNIT_HINT_KEYWORDS)
        if unit_terms:
            score += 2
            matched_terms.extend(unit_terms[:2])
            reasons.append("unit_hints")

        period_terms = _matched_terms(context, _PERIOD_HINT_KEYWORDS)
        if period_terms:
            score += 1
            matched_terms.extend(period_terms[:2])
            reasons.append("period_hints")

        has_unit_or_period_hint = bool(unit_terms or period_terms)
        has_headline_context = bool(
            financial_summary_context_terms
            or operating_results_context_terms
            or paired_terms
        )
        if (
            context.page.page_number in headline_summary_pages
            and has_unit_or_period_hint
            and has_headline_context
        ):
            score += 8
            reasons.append("headline_summary_context")

    if field_name == "unit":
        unit_terms = _matched_terms(context, _UNIT_HINT_KEYWORDS)
        if unit_terms:
            score += 4
            matched_terms.extend(unit_terms[:2])
            reasons.append("unit_hints")

    if field_name == "fiscal_period":
        period_terms = _matched_terms(context, _PERIOD_HINT_KEYWORDS)
        if period_terms:
            score += 4
            matched_terms.extend(period_terms[:2])
            reasons.append("period_hints")

    if score <= 0:
        return None

    if not is_front_page and not field_terms and field_name != "unit":
        return None
    if not is_front_page and not field_terms and field_name == "unit":
        return None

    return _build_debug_entry(context, score, matched_terms, reasons)


def _keywords_for_field(field_name: str) -> tuple[str, ...]:
    if field_name == "company_name":
        return ()
    return _FIELD_KEYWORDS.get(field_name, ())


def _rank_debug_entries(
    entries: Iterable[LocatorDebugEntry | None],
) -> list[LocatorDebugEntry]:
    ranked = [entry for entry in entries if entry is not None and entry.score > 0]
    ranked.sort(key=lambda entry: (-entry.score, entry.page_number))
    return ranked


def _build_debug_entry(
    context: _PageMatchContext,
    score: int,
    matched_terms: list[str],
    reasons: list[str],
) -> LocatorDebugEntry:
    deduped_terms = tuple(_dedupe_preserve_order(matched_terms))
    deduped_reasons = _dedupe_preserve_order(reasons)
    snippet, snippet_reason = _snippet_for_terms(context, deduped_terms)
    if snippet_reason:
        deduped_reasons.append(snippet_reason)

    return LocatorDebugEntry(
        page_number=context.page.page_number,
        score=score,
        matched_terms=deduped_terms,
        snippet=snippet,
        reasons=tuple(deduped_reasons),
    )


def _snippet_for_terms(
    context: _PageMatchContext,
    matched_terms: tuple[str, ...],
    *,
    window: int = 90,
) -> tuple[str, str | None]:
    for term in matched_terms:
        normalized_term = _normalize_for_match(term)
        if not normalized_term:
            continue
        match_index = context.normalized_text.find(normalized_term)
        if match_index < 0 or not context.normalized_to_raw:
            continue
        raw_start = context.normalized_to_raw[match_index]
        raw_end_index = min(
            match_index + len(normalized_term) - 1,
            len(context.normalized_to_raw) - 1,
        )
        raw_end = context.normalized_to_raw[raw_end_index] + 1
        snippet_start = max(0, raw_start - window)
        snippet_end = min(len(context.raw_text), raw_end + window)
        return _clean_snippet(context.raw_text[snippet_start:snippet_end]), None

    return _clean_snippet(context.raw_text[: window * 2]), "snippet_fallback"


def _matched_terms(
    context: _PageMatchContext,
    terms: Iterable[str],
) -> list[str]:
    matched: list[str] = []
    for term in terms:
        normalized_term = _normalize_for_match(term)
        if normalized_term and normalized_term in context.normalized_text:
            matched.append(term)
    return _dedupe_preserve_order(matched)


def _match_keyword_pages(
    parsed_filing: ParsedFiling,
    keywords: tuple[str, ...],
    *,
    limit: int,
) -> list[int]:
    matched_pages: list[int] = []
    for page in parsed_filing.pages:
        context = _build_page_match_context(page)
        if _matched_terms(context, keywords):
            matched_pages.append(page.page_number)
            if len(matched_pages) >= limit:
                break
    return matched_pages


def _first_page_numbers(parsed_filing: ParsedFiling, *, limit: int) -> list[int]:
    if limit <= 0:
        return []
    return [
        page.page_number
        for page in parsed_filing.pages[: min(limit, len(parsed_filing.pages))]
    ]


def _build_page_match_context(page: ParsedPage) -> _PageMatchContext:
    raw_text = _page_text(page)
    normalized_text, normalized_to_raw = _normalize_with_mapping(raw_text)
    return _PageMatchContext(
        page=page,
        raw_text=raw_text,
        normalized_text=normalized_text,
        normalized_to_raw=tuple(normalized_to_raw),
    )


def _page_text(page: ParsedPage) -> str:
    return page.markdown or page.text


def _normalize_for_match(text: str) -> str:
    normalized, _ = _normalize_with_mapping(text)
    return normalized


def _normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    raw_indexes: list[int] = []

    for raw_index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char).casefold()
        for normalized_char in normalized:
            if normalized_char.isspace():
                continue
            normalized_chars.append(normalized_char)
            raw_indexes.append(raw_index)

    return "".join(normalized_chars), raw_indexes


def _clean_snippet(text: str) -> str:
    return " ".join(text.split())


def _dedupe_preserve_order[T](items: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    deduped: list[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
