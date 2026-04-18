from __future__ import annotations

import re
from typing import Iterable

from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import Citation, ParsedFiling, ParsedPage


_HEADLINE_FACT_FIELDS = (
    "company_name",
    "fiscal_period",
    "unit",
    "revenue",
    "net_profit",
)


def enrich_headline_metric_citations(
    parsed_filing: ParsedFiling,
    facts: HeadlineMetricFacts,
) -> HeadlineMetricFacts:
    enriched_facts = facts.model_copy(deep=True)

    for field_name in _HEADLINE_FACT_FIELDS:
        fact_field = getattr(enriched_facts, field_name)
        if fact_field.citations or fact_field.value is None:
            continue

        citations = _find_citations(
            field_name=field_name,
            parsed_filing=parsed_filing,
            candidates=_build_search_candidates(field_name, fact_field.value),
            max_results=1,
        )
        if citations:
            fact_field.citations.extend(citations)

    return enriched_facts


def _build_search_candidates(field_name: str, value: str | float | int) -> list[str]:
    if field_name == "fiscal_period" and isinstance(value, str):
        return _build_fiscal_period_candidates(value)

    if field_name == "unit" and isinstance(value, str):
        return _build_unit_candidates(value)

    if isinstance(value, str):
        candidate = value.strip()
        return [candidate] if candidate else []

    candidates: list[str] = []
    numeric_text = str(value).strip()
    if numeric_text.endswith(".0"):
        numeric_text = numeric_text[:-2]

    if numeric_text:
        candidates.append(numeric_text)

    try:
        integer_value = int(float(value))
        candidates.append(f"{integer_value:,}")
    except (TypeError, ValueError):
        pass

    return _dedupe_preserve_order(candidates)


def _build_unit_candidates(value: str) -> list[str]:
    candidate = value.strip()
    if not candidate:
        return []

    variants = [candidate]
    normalized_value = candidate.lower()

    if "人民幣百萬元" in candidate or "人民币百万元" in candidate:
        variants.extend(
            [
                "（人民幣百萬元，另有指明者除外）",
                "（人民币百万元，另有指明者除外）",
                "人民幣百萬元",
                "人民币百万元",
            ]
        )

    if "rmb" in normalized_value:
        variants.extend(["RMB", "RMB million", "Renminbi"])

    return _dedupe_preserve_order(variants)


def _build_fiscal_period_candidates(value: str) -> list[str]:
    candidate = value.strip()
    if not candidate:
        return []

    variants = [candidate]
    english_match = re.search(
        r"(?:(?:fiscal\s+)?)year ended ([A-Za-z]+) (\d{1,2}), (\d{4})",
        candidate,
        re.IGNORECASE,
    )

    if english_match:
        month_name, day_text, year_text = english_match.groups()
        month_number = _month_name_to_number(month_name)
        if month_number is not None:
            day_number = int(day_text)
            year_digits_cn = _digits_to_cn(year_text)
            month_cn = _number_to_cn(month_number)
            day_cn = _number_to_cn(day_number)
            variants.extend(
                [
                    f"截至{year_digits_cn}年{month_cn}月{day_cn}日止年度",
                    f"截至{year_text}年{month_number}月{day_number}日止年度",
                    f"{year_digits_cn}年{month_cn}月{day_cn}日",
                    f"{year_text}年{month_number}月{day_number}日",
                ]
            )
    elif "annual report" in candidate.lower():
        variants.append(re.sub("annual report", "年度报告", candidate, flags=re.IGNORECASE))

    return _dedupe_preserve_order(variants)


def _find_citations(
    *,
    field_name: str,
    parsed_filing: ParsedFiling,
    candidates: Iterable[str],
    max_results: int,
) -> list[Citation]:
    normalized_candidates = [
        (candidate, _normalize_for_match(candidate))
        for candidate in candidates
        if candidate and _normalize_for_match(candidate)
    ]
    if not normalized_candidates:
        return []

    citations: list[Citation] = []

    for page in _iter_pages_for_field(parsed_filing, field_name):
        page_text = page.markdown or page.text
        page_lines = list(_iter_non_empty_lines(page_text))

        for window_text in _iter_search_windows(page_lines, max_window_size=3):
            normalized_window = _normalize_for_match(window_text)
            if not normalized_window:
                continue

            for _, normalized_candidate in normalized_candidates:
                if normalized_candidate and normalized_candidate in normalized_window:
                    citations.append(
                        Citation(
                            document_id=parsed_filing.document.document_id,
                            source_path=parsed_filing.document.source_path,
                            page_number=page.page_number,
                            quote=_shorten_quote(window_text),
                        )
                    )
                    break

            if len(citations) >= max_results:
                return citations

    return citations


def _iter_pages_for_field(parsed_filing: ParsedFiling, field_name: str) -> Iterable[ParsedPage]:
    priority_count = {
        "company_name": 3,
        "fiscal_period": 4,
        "unit": 6,
        "revenue": 8,
        "net_profit": 8,
    }.get(field_name, 0)

    pages = parsed_filing.pages
    if priority_count <= 0 or len(pages) <= priority_count:
        yield from pages
        return

    yield from pages[:priority_count]
    yield from pages[priority_count:]


def _iter_non_empty_lines(text: str) -> Iterable[str]:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            yield line


def _iter_search_windows(lines: list[str], max_window_size: int) -> Iterable[str]:
    for index, line in enumerate(lines):
        yield line

        for window_size in range(2, max_window_size + 1):
            end_index = index + window_size
            if end_index > len(lines):
                break
            yield " ".join(lines[index:end_index])


def _normalize_for_match(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    normalized = normalized.lower()
    normalized = normalized.replace(",", "").replace("，", "")
    normalized = normalized.replace("（", "(").replace("）", ")")
    return normalized


def _shorten_quote(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _month_name_to_number(month_name: str) -> int | None:
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    return months.get(month_name.lower())


def _digits_to_cn(text: str) -> str:
    mapping = {
        "0": "零",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }
    return "".join(mapping.get(char, char) for char in text)


def _number_to_cn(number: int) -> str:
    if number <= 0:
        return str(number)

    numerals = {
        0: "零",
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    if number <= 10:
        return numerals[number]
    if number < 20:
        return f"十{numerals[number % 10]}" if number % 10 else "十"
    if number < 100:
        tens, ones = divmod(number, 10)
        if ones == 0:
            return f"{numerals[tens]}十"
        return f"{numerals[tens]}十{numerals[ones]}"
    return str(number)
