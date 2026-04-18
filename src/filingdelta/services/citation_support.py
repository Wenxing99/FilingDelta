from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from filingdelta.schemas.filing import Citation, ParsedFiling, ParsedPage


def build_citation_from_evidence(
    parsed_filing: ParsedFiling,
    *,
    evidence_page: int | None,
    evidence_quote: str | None,
) -> Citation | None:
    if not evidence_page or not evidence_quote:
        return None

    page = next(
        (candidate for candidate in parsed_filing.pages if candidate.page_number == evidence_page),
        None,
    )
    if page is None:
        return None

    quote = match_quote_on_page(page, evidence_quote)
    if not quote:
        return None

    return Citation(
        document_id=parsed_filing.document.document_id,
        source_path=parsed_filing.document.source_path,
        page_number=page.page_number,
        quote=quote,
    )


def match_quote_on_page(page: ParsedPage, evidence_quote: str) -> str | None:
    page_text = page.markdown or page.text
    page_lines = list(iter_non_empty_lines(page_text))

    for candidate_quote in _build_quote_candidates(evidence_quote):
        normalized_quote = normalize_for_match(candidate_quote)
        if not normalized_quote or len(normalized_quote) < 6:
            continue

        normalized_page = normalize_for_match(page_text)
        if normalized_quote in normalized_page:
            return shorten_quote(candidate_quote)

        for window_text in iter_search_windows(page_lines, max_window_size=3):
            normalized_window = normalize_for_match(window_text)
            if not normalized_window:
                continue
            if normalized_quote in normalized_window:
                return shorten_quote(window_text)
            if (
                normalized_window in normalized_quote
                and len(normalized_window) >= max(12, len(normalized_quote) // 2)
            ):
                return shorten_quote(window_text)

    fragment_match = _match_fragmented_table_quote(page_lines, evidence_quote)
    if fragment_match:
        return fragment_match

    return None


def _build_quote_candidates(evidence_quote: str) -> list[str]:
    candidates = [evidence_quote.strip()]
    ellipsis_fragments = re.split(r"(?:\.{3,}|…+)", evidence_quote)
    candidates.extend(fragment.strip() for fragment in ellipsis_fragments if fragment.strip())
    candidates = [candidate for candidate in candidates if candidate]
    candidates.sort(key=len, reverse=True)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _match_fragmented_table_quote(page_lines: list[str], evidence_quote: str) -> str | None:
    fragments = _build_table_fragments(evidence_quote)
    if len(fragments) < 2:
        return None

    matched_windows: list[str] = []
    for fragment in fragments:
        for window_text in iter_search_windows(page_lines, max_window_size=3):
            normalized_window = normalize_for_match(window_text)
            if not normalized_window:
                continue
            if fragment in normalized_window:
                matched_windows.append(shorten_quote(window_text))
                break
            if (
                normalized_window in fragment
                and len(normalized_window) >= max(12, len(fragment) // 2)
            ):
                matched_windows.append(shorten_quote(window_text))
                break

    if len(matched_windows) < 2:
        return None

    deduped_windows: list[str] = []
    seen: set[str] = set()
    for window in matched_windows:
        if window in seen:
            continue
        seen.add(window)
        deduped_windows.append(window)
    return shorten_quote(" ".join(deduped_windows))


def _build_table_fragments(evidence_quote: str) -> list[str]:
    fragments = [
        normalize_for_match(fragment)
        for fragment in re.split(r"[;；。]", evidence_quote)
        if fragment.strip()
    ]
    return [fragment for fragment in fragments if len(fragment) >= 10]


def iter_non_empty_lines(text: str) -> Iterable[str]:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            yield line


def iter_search_windows(lines: list[str], max_window_size: int) -> Iterable[str]:
    for index, line in enumerate(lines):
        yield line

        for window_size in range(2, max_window_size + 1):
            end_index = index + window_size
            if end_index > len(lines):
                break
            yield " ".join(lines[index:end_index])


def normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"[,\.;:，。；：()\[\]{}<>％%/\\\-—_]", "", normalized)
    return normalized


def shorten_quote(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."
