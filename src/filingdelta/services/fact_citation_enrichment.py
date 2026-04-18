from __future__ import annotations

import re
from typing import Iterable

from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import Citation, ParsedFiling


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
            parsed_filing=parsed_filing,
            candidates=_build_search_candidates(fact_field.value),
            max_results=1,
        )
        if citations:
            fact_field.citations.extend(citations)

    return enriched_facts


def _build_search_candidates(value: str | float | int) -> list[str]:
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


def _find_citations(
    *,
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

    for page in parsed_filing.pages:
        page_text = page.markdown or page.text
        for line in _iter_non_empty_lines(page_text):
            normalized_line = _normalize_for_match(line)
            if not normalized_line:
                continue

            for _, normalized_candidate in normalized_candidates:
                if normalized_candidate in normalized_line:
                    citations.append(
                        Citation(
                            document_id=parsed_filing.document.document_id,
                            source_path=parsed_filing.document.source_path,
                            page_number=page.page_number,
                            quote=_shorten_quote(line),
                        )
                    )
                    break

            if len(citations) >= max_results:
                return citations

    return citations


def _iter_non_empty_lines(text: str) -> Iterable[str]:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            yield line


def _normalize_for_match(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    normalized = normalized.replace(",", "").replace("，", "")
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
