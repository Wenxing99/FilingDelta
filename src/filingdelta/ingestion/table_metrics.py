from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import fitz

from filingdelta.schemas.fact_extraction import (
    HeadlineMetricsStructuredExtraction,
    NumericFactEvidence,
    TextFactEvidence,
)
from filingdelta.schemas.filing import FilingDocType, FilingSource, ParsedFiling, ParsedPage


_METRIC_FIELDS = ("revenue", "net_profit", "total_assets", "total_liabilities", "roe")
_BALANCE_SHEET_FIELDS = {"total_assets", "total_liabilities"}

_TABLE_CONTEXT_KEYWORDS = (
    "主要会计数据",
    "主要會計數據",
    "主要财务指标",
    "主要財務指標",
    "会计数据和财务指标",
    "會計數據和財務指標",
    "财务比率",
    "財務比率",
    "经营业绩",
    "經營業績",
    "经营概要",
    "經營概要",
    "业绩摘要",
    "業績摘要",
    "financial highlights",
    "financial summary",
    "selected financial",
    "management discussion",
    "balance sheet",
    "statement of financial position",
    "statements of financial position",
    "资产负债表",
    "資產負債表",
    "财务状况表",
    "財務狀況表",
)

_UNIT_PATTERNS = (
    r"(人民币百万元[^;\n，。)]*)",
    r"(人民幣百萬元[^;\n，。)]*)",
    r"(人民币千元[^;\n，。)]*)",
    r"(人民幣千元[^;\n，。)]*)",
    r"(人民币万元[^;\n，。)]*)",
    r"(人民幣萬元[^;\n，。)]*)",
    r"(人民币亿元[^;\n，。)]*)",
    r"(人民幣億元[^;\n，。)]*)",
    r"(港币百万元[^;\n，。)]*)",
    r"(港幣百萬元[^;\n，。)]*)",
    r"(港元百万元[^;\n，。)]*)",
    r"(港元百萬元[^;\n，。)]*)",
    r"(RMB\s+million)",
    r"(HKD\s+million)",
    r"(USD\s+million)",
    r"(million\s+RMB)",
    r"(million\s+HKD)",
    r"(million\s+USD)",
)


@dataclass(frozen=True)
class _RowValue:
    value: float
    line_index: int
    raw_text: str


@dataclass(frozen=True)
class _MetricCandidate:
    field_name: str
    value: float
    page_number: int
    label: str
    selected_text: str
    unit: str | None
    score: int
    source: str


@dataclass(frozen=True)
class _LabelMatch:
    label: str
    end_index: int
    score: int


@dataclass(frozen=True)
class TableHeadlineMetricsResult:
    structured: HeadlineMetricsStructuredExtraction
    has_table_signal: bool = False


def extract_table_headline_metrics(
    *,
    source: FilingSource,
    parsed_filing: ParsedFiling,
    selection: object,
) -> TableHeadlineMetricsResult:
    if source.source_path.suffix.lower() != ".pdf":
        return TableHeadlineMetricsResult(
            structured=HeadlineMetricsStructuredExtraction(),
            has_table_signal=False,
        )

    candidate_pages = _candidate_pages(parsed_filing, selection)
    candidates: list[_MetricCandidate] = []

    for page in _pages_by_number(parsed_filing, candidate_pages):
        candidates.extend(_extract_line_table_candidates(page=page, source=source))

    structured = HeadlineMetricsStructuredExtraction()
    best_by_field = {
        field_name: _select_best_candidate(candidates, field_name)
        for field_name in _METRIC_FIELDS
    }

    for field_name, candidate in best_by_field.items():
        if candidate is None:
            continue
        setattr(structured, field_name, _candidate_to_numeric_evidence(candidate))

    unit = _select_unit_from_candidates(candidates)
    if unit:
        structured.unit = TextFactEvidence(value=unit, confidence=0.92)

    return TableHeadlineMetricsResult(
        structured=structured,
        has_table_signal=bool(candidates) or _has_table_signal(parsed_filing, candidate_pages),
    )


def _candidate_pages(parsed_filing: ParsedFiling, selection: object) -> list[int]:
    pages: list[int] = []

    for method_name in ("all_pages",):
        method = getattr(selection, method_name, None)
        if callable(method):
            pages.extend(int(page_number) for page_number in method())

    pages_for = getattr(selection, "pages_for", None)
    if callable(pages_for):
        for field_name in (
            "revenue",
            "net_profit",
            "total_assets",
            "total_liabilities",
            "roe",
            "unit",
        ):
            pages.extend(int(page_number) for page_number in pages_for(field_name))

    scored_pages: list[tuple[int, int]] = []
    for page in parsed_filing.pages:
        score = _score_table_page(page)
        if score > 0:
            scored_pages.append((score, page.page_number))
    scored_pages.sort(key=lambda item: (-item[0], item[1]))
    pages.extend(page_number for _, page_number in scored_pages[:10])

    first_pages = [page.page_number for page in parsed_filing.pages[: min(3, len(parsed_filing.pages))]]
    pages.extend(first_pages)

    return _dedupe_preserve_order(pages)[:16]


def _score_table_page(page: ParsedPage) -> int:
    text = _normalize_for_match(page.markdown or page.text)
    score = 0
    for keyword in _TABLE_CONTEXT_KEYWORDS:
        if _normalize_for_match(keyword) in text:
            score += 4
    for keyword in ("营业收入", "營業收入", "归属于", "歸屬於", "净资产收益率", "淨資產收益率", "ROE", "ROAE"):
        if _normalize_for_match(keyword) in text:
            score += 2
    if _label_contains_any_variant(
        text,
        (
            "总资产",
            "總資產",
            "资产总计",
            "資產總計",
            "总负债",
            "總負債",
            "负债合计",
            "負債合計",
            "资产负债表",
            "資產負債表",
        ),
    ):
        score += 4
    if any(
        token in text
        for token in (
            "totalassets",
            "totalliabilities",
            "balancesheet",
            "statementoffinancialposition",
        )
    ):
        score += 4
    return score


def _pages_by_number(parsed_filing: ParsedFiling, page_numbers: list[int]) -> list[ParsedPage]:
    lookup = {page.page_number: page for page in parsed_filing.pages}
    return [lookup[page_number] for page_number in page_numbers if page_number in lookup]


def _extract_line_table_candidates(
    *,
    page: ParsedPage,
    source: FilingSource,
) -> list[_MetricCandidate]:
    lines = _clean_lines(page.markdown or page.text)
    if not lines:
        return []

    candidates: list[_MetricCandidate] = []
    seen: set[tuple[str, int, int]] = set()

    for index, _line in enumerate(lines):
        for field_name in _METRIC_FIELDS:
            if _should_skip_metric_context(source, lines, index, field_name):
                continue
            label_match = _match_metric_label(lines, index, field_name)
            if label_match is None:
                continue
            if field_name == "net_profit" and _near_previous_non_ifrs_label(lines, index):
                continue
            seen_key = (field_name, page.page_number, index)
            if seen_key in seen:
                continue
            seen.add(seen_key)

            row_values = _collect_row_values(lines, label_match.end_index)
            selected_value = _select_row_value(
                values=row_values,
                source=source,
                context_lines=lines[max(0, index - 80) : label_match.end_index + 1],
            )
            if selected_value is None:
                continue
            if not _value_is_compatible(field_name, selected_value.value):
                continue

            unit = _find_unit_near(lines, index)
            candidates.append(
                _MetricCandidate(
                    field_name=field_name,
                    value=selected_value.value,
                    page_number=page.page_number,
                    label=label_match.label,
                    selected_text=selected_value.raw_text,
                    unit=unit,
                    score=label_match.score
                    + _score_context(lines, index)
                    + _score_selected_value(source, row_values, selected_value),
                    source="line_table",
                )
            )

    return candidates


def _extract_pymupdf_table_candidates(
    *,
    source_path: Path,
    parsed_filing: ParsedFiling,
    source: FilingSource,
    page_numbers: list[int],
) -> list[_MetricCandidate]:
    if source_path.suffix.lower() != ".pdf" or not source_path.exists():
        return []

    candidates: list[_MetricCandidate] = []
    valid_page_numbers = {
        page.page_number for page in parsed_filing.pages if page.page_number in page_numbers
    }
    if not valid_page_numbers:
        return candidates

    try:
        document = fitz.open(source_path)
    except (OSError, RuntimeError, ValueError):
        return candidates

    with document:
        for page_number in sorted(valid_page_numbers):
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(document):
                continue
            page = document[page_index]
            for strategy in ("text", "lines"):
                try:
                    table_finder = page.find_tables(strategy=strategy)
                except (RuntimeError, ValueError):
                    continue
                for table in table_finder.tables:
                    candidates.extend(
                        _extract_matrix_candidates(
                            matrix=table.extract(),
                            page_number=page_number,
                            source=source,
                            strategy=strategy,
                        )
                    )
    return candidates


def _extract_matrix_candidates(
    *,
    matrix: list[list[str | None]],
    page_number: int,
    source: FilingSource,
    strategy: str,
) -> list[_MetricCandidate]:
    candidates: list[_MetricCandidate] = []
    unit = _find_unit_in_text(" ".join(_cell_to_text(cell) for row in matrix[:8] for cell in row))

    for row_index, raw_row in enumerate(matrix):
        row = [_cell_to_text(cell) for cell in raw_row]
        if not any(row):
            continue

        row_text = " ".join(cell for cell in row if cell)
        if len(row_text) > 120:
            continue
        for field_name in _METRIC_FIELDS:
            label_score = _score_metric_label(field_name, row_text)
            if label_score <= 0:
                continue
            values = _row_values_from_cells(row, row_index)
            selected_value = _select_row_value(
                values=values,
                source=source,
                context_lines=_matrix_context(matrix, row_index),
            )
            if selected_value is None or not _value_is_compatible(field_name, selected_value.value):
                continue
            candidates.append(
                _MetricCandidate(
                    field_name=field_name,
                    value=selected_value.value,
                    page_number=page_number,
                    label=_strip_value_like_suffix(row_text),
                    selected_text=selected_value.raw_text,
                    unit=unit,
                    score=label_score + 8 + _score_selected_value(source, values, selected_value),
                    source=f"pymupdf_{strategy}",
                )
            )

    return candidates


def _match_metric_label(
    lines: list[str],
    start_index: int,
    field_name: str,
) -> _LabelMatch | None:
    if _line_is_numeric_like(lines[start_index]):
        return None

    best: _LabelMatch | None = None
    parts: list[str] = []
    for offset in range(4):
        index = start_index + offset
        if index >= len(lines):
            break

        line = lines[index]
        if offset > 0 and _line_is_numeric_like(line):
            break
        if offset > 0 and _is_obvious_table_header(line):
            break

        parts.append(line)
        label = " ".join(parts)
        score = _score_metric_label(field_name, label)
        if score > 0 and (best is None or score >= best.score):
            best = _LabelMatch(label=label, end_index=index, score=score)

    return best


def _score_metric_label(field_name: str, label: str) -> int:
    if len(label) > 90:
        return 0
    if re.search(r"[。；;]", label) or any(token in label for token in ("报告期内", "實現", "实现")):
        return 0

    normalized = _normalize_for_match(label)
    if not normalized:
        return 0

    if field_name == "revenue":
        if any(
            token in normalized
            for token in (
                "占营业收入",
                "佔營業收入",
                "revenuepercentage",
                "营业收入利息收入",
                "營業收入利息收入",
            )
        ):
            return 0
        if "营业收入" in normalized or "營業收入" in normalized:
            return 70
        if normalized in {"收入", "收益"}:
            return 64
        if "总收入" in normalized or "總收入" in normalized:
            return 55
        if re.search(r"\brevenues?\b", normalized):
            return 55
        return 0

    if field_name == "net_profit":
        if any(token in normalized for token in ("非国际", "非國際", "nonifrs", "nongaap")):
            return 0
        if any(token in normalized for token in ("非经常性", "非經常性", "nonrecurring")):
            return 35
        if (
            ("归属于" in normalized or "歸屬於" in normalized)
            and ("股东" in normalized or "股東" in normalized)
            and ("净利润" in normalized or "淨利潤" in normalized)
        ):
            return 78
        if (
            ("权益持有人" in normalized or "權益持有人" in normalized)
            and ("应占" in normalized or "應佔" in normalized)
            and ("盈利" in normalized or "利润" in normalized or "利潤" in normalized)
        ):
            return 78
        if "profitattributable" in normalized or "netincomeattributable" in normalized:
            return 70
        return 0

    if field_name == "total_assets":
        if _label_contains_any_variant(
            normalized,
            (
                "总资产",
                "總資產",
                "资产总计",
                "資產總計",
                "资产合计",
                "資產合計",
            ),
        ):
            return 78
        if any(token in normalized for token in ("totalassets", "assetstotal")):
            return 78
        return 0

    if field_name == "total_liabilities":
        if any(
            token in normalized
            for token in (
                "totalliabilitiesandequity",
                "totalliabilitiesandshareholdersequity",
                "liabilitiesandequity",
            )
        ):
            return 0
        if _label_contains_any_variant(
            normalized,
            (
                "总负债",
                "總負債",
                "负债合计",
                "負債合計",
                "负债总计",
                "負債總計",
            ),
        ):
            return 78
        if any(token in normalized for token in ("totalliabilities", "liabilitiestotal")):
            return 78
        return 0

    if field_name == "roe":
        if any(
            token in normalized
            for token in (
                "roaa",
                "returnonassets",
                "总资产收益率",
                "總資產收益率",
                "平均总资产收益率",
                "平均總資產收益率",
            )
        ):
            return 0
        if not any(
            token in normalized
            for token in (
                "roe",
                "roae",
                "returnonequity",
                "净资产收益率",
                "淨資產收益率",
            )
        ):
            return 0

        score = 60
        if "普通股" in normalized or "commonshare" in normalized:
            score += 10
        if "加权平均" in normalized or "加權平均" in normalized or "weightedaverage" in normalized:
            score += 8
        if "年化" in normalized or "annualized" in normalized:
            score += 6
        # The UI label is generic ROE, so non-deducted ROE remains slightly preferred.
        if "非经常性" in normalized or "非經常性" in normalized or "nonrecurring" in normalized:
            score -= 3
        return score

    return 0


def _collect_row_values(lines: list[str], label_end_index: int) -> list[_RowValue]:
    values: list[_RowValue] = []
    for index in range(label_end_index + 1, min(len(lines), label_end_index + 13)):
        line = lines[index]
        if _is_footnote_marker(line):
            continue
        if values and _looks_like_next_row_label(line):
            break
        line_values = _extract_numeric_values(line)
        if line_values:
            values.extend(
                _RowValue(value=value, line_index=index, raw_text=line)
                for value in line_values
            )
            continue
        if values and not _is_ignorable_after_values(line):
            break
    return values


def _row_values_from_cells(row: list[str], row_index: int) -> list[_RowValue]:
    values: list[_RowValue] = []
    for cell in row:
        if not cell:
            continue
        for value in _extract_numeric_values(cell):
            values.append(_RowValue(value=value, line_index=row_index, raw_text=cell))
    return values


def _select_row_value(
    *,
    values: list[_RowValue],
    source: FilingSource,
    context_lines: list[str],
) -> _RowValue | None:
    if not values:
        return None

    if _is_interim_context(source, context_lines):
        if len(values) >= 4 and _looks_like_interim_alternating_row(values):
            return values[2]
        if len(values) >= 2:
            return values[-1]

    if source.doc_type == FilingDocType.ANNUAL_REPORT:
        selected = _select_annual_value_by_target_year(
            values=values,
            source=source,
            context_lines=context_lines,
        )
        if selected is not None:
            return selected
        if _context_has_multiple_year_headers(context_lines):
            return None

    return values[0]


def _select_annual_value_by_target_year(
    *,
    values: list[_RowValue],
    source: FilingSource,
    context_lines: list[str],
) -> _RowValue | None:
    target_year = _target_year_from_source(source)
    if target_year is None:
        return None

    year_headers = _extract_year_headers(context_lines)
    if not year_headers:
        return None
    if len(year_headers) > len(values):
        year_headers = year_headers[-len(values) :]

    for index, year in enumerate(year_headers):
        if year == target_year and index < len(values):
            return values[index]
    return None


def _target_year_from_source(source: FilingSource) -> int | None:
    for text in (
        source.fiscal_period or "",
        source.source_path.name,
        source.source_path.stem,
    ):
        year = _extract_first_year(text)
        if year is not None:
            return year
    return None


def _extract_first_year(text: str) -> int | None:
    match = re.search(r"(20\d{2})", text)
    if match:
        return int(match.group(1))

    chinese_match = re.search(r"([二兩两零〇○]{2,4}[一二三四五六七八九十零〇○]{0,2})年", text)
    if not chinese_match:
        return None
    return _parse_chinese_year(chinese_match.group(1))


def _extract_year_headers(context_lines: list[str]) -> list[int]:
    years: list[int] = []
    for line in context_lines:
        years.extend(_extract_years_from_line(line))
    return years


def _extract_years_from_line(line: str) -> list[int]:
    years: list[int] = []
    normalized = unicodedata.normalize("NFKC", line)

    for match in re.finditer(r"20\d{2}\s*年?", normalized):
        year = _extract_first_year(match.group(0))
        if year is not None:
            years.append(year)

    for match in re.finditer(r"([二兩两零〇○]{2,4}[一二三四五六七八九十零〇○]{0,2})年", normalized):
        year = _parse_chinese_year(match.group(1))
        if year is not None:
            years.append(year)

    return years


def _parse_chinese_year(text: str) -> int | None:
    mapping = {
        "零": "0",
        "〇": "0",
        "○": "0",
        "一": "1",
        "二": "2",
        "兩": "2",
        "两": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
    }
    digits = "".join(mapping.get(char, "") for char in text)
    if len(digits) != 4:
        return None
    year = int(digits)
    if 1900 <= year <= 2100:
        return year
    return None


def _context_has_multiple_year_headers(context_lines: list[str]) -> bool:
    return len(_extract_year_headers(context_lines)) >= 2


def _looks_like_interim_alternating_row(values: list[_RowValue]) -> bool:
    if len(values) < 4:
        return False
    return abs(values[1].value) < 200 and abs(values[3].value) < 200


def _is_interim_context(source: FilingSource, context_lines: list[str]) -> bool:
    context = _normalize_for_match(" ".join(context_lines) + " " + (source.fiscal_period or ""))
    if source.doc_type == FilingDocType.INTERIM_REPORT:
        return True
    return any(token in context for token in ("1-9月", "1-6月", "1-3月", "q1", "q2", "q3"))


def _value_is_compatible(field_name: str, value: float) -> bool:
    if field_name == "roe":
        return 0 < value < 100
    if field_name in {"revenue", "net_profit", "total_assets", "total_liabilities"}:
        return abs(value) >= 1
    return True


def _score_selected_value(
    source: FilingSource,
    values: list[_RowValue],
    selected_value: _RowValue,
) -> int:
    if not _is_interim_context(source, []):
        return 0
    if len(values) >= 2 and selected_value == values[-1]:
        return 4
    return 0


def _score_context(lines: list[str], label_index: int) -> int:
    context = _normalize_for_match(" ".join(lines[max(0, label_index - 60) : label_index + 1]))
    score = 0
    if any(
        token in context
        for token in (
            "主要会计数据",
            "主要會計數據",
            "主要财务数据",
            "主要財務數據",
            "主要财务指标",
            "主要財務指標",
            "会计数据和财务指标",
            "會計數據和財務指標",
        )
    ):
        score += 16
    elif any(_normalize_for_match(keyword) in context for keyword in _TABLE_CONTEXT_KEYWORDS):
        score += 6
    if any(token in context for token in ("人民币百万元", "人民幣百萬元", "rmbmillion")):
        score += 4
    return score


def _should_skip_metric_context(
    source: FilingSource,
    lines: list[str],
    label_index: int,
    field_name: str,
) -> bool:
    context = _normalize_for_match(" ".join(lines[max(0, label_index - 60) : label_index + 1]))
    if source.doc_type == FilingDocType.ANNUAL_REPORT and any(
        token in context
        for token in (
            "按季度披露",
            "季度披露",
            "第四季度",
            "三个月",
            "三個月",
            "环比",
            "環比",
        )
    ):
        return True

    if field_name in _BALANCE_SHEET_FIELDS and any(
        token in context
        for token in (
            "èµ„äº§è´Ÿå€ºè¡¨",
            "è³‡ç”¢è² å‚µè¡¨",
            "balancesheet",
            "statementoffinancialposition",
        )
    ):
        return False

    if any(
        token in context
        for token in (
            "财务报表",
            "財務報表",
            "合并利润表",
            "合併利潤表",
            "综合损益表",
            "綜合損益表",
            "资产负债表",
            "資產負債表",
            "现金流量表",
            "現金流量表",
        )
    ) and not any(
        token in context
        for token in (
            "主要会计数据",
            "主要會計數據",
            "主要财务数据",
            "主要財務數據",
            "主要财务指标",
            "主要財務指標",
        )
    ):
        return True

    return False


def _near_previous_non_ifrs_label(lines: list[str], label_index: int) -> bool:
    context = _normalize_for_match(" ".join(lines[max(0, label_index - 3) : label_index + 1]))
    return any(token in context for token in ("非国际", "非國際", "nonifrs", "nongaap"))


def _candidate_to_numeric_evidence(candidate: _MetricCandidate) -> NumericFactEvidence:
    return NumericFactEvidence(
        value=candidate.value,
        evidence_page=candidate.page_number,
        evidence_quote=_build_evidence_quote(candidate),
        confidence=min(0.99, 0.86 + min(candidate.score, 30) / 300),
    )


def _build_evidence_quote(candidate: _MetricCandidate) -> str:
    fragments: list[str] = []
    if candidate.unit:
        fragments.append(candidate.unit)
    fragments.append(f"{candidate.label} {candidate.selected_text}".strip())
    return "; ".join(_dedupe_preserve_order(fragments))


def _select_best_candidate(
    candidates: list[_MetricCandidate],
    field_name: str,
) -> _MetricCandidate | None:
    field_candidates = [candidate for candidate in candidates if candidate.field_name == field_name]
    if not field_candidates:
        return None
    return max(
        field_candidates,
        key=lambda candidate: (candidate.score, -len(candidate.label), -candidate.page_number),
    )


def _select_unit_from_candidates(candidates: list[_MetricCandidate]) -> str | None:
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if candidate.unit:
            return candidate.unit
    return None


def _has_table_signal(parsed_filing: ParsedFiling, page_numbers: list[int]) -> bool:
    for page in _pages_by_number(parsed_filing, page_numbers):
        text = _normalize_for_match(page.markdown or page.text)
        if any(_normalize_for_match(keyword) in text for keyword in _TABLE_CONTEXT_KEYWORDS):
            return True
    return False


def _find_unit_near(lines: list[str], label_index: int) -> str | None:
    start = max(0, label_index - 35)
    for index in range(label_index, start - 1, -1):
        unit = _find_unit_in_text(lines[index])
        if unit:
            return unit
    return None


def _find_unit_in_text(text: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", text)
    for pattern in _UNIT_PATTERNS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" （()，,")
    return None


def _matrix_context(matrix: list[list[str | None]], row_index: int) -> list[str]:
    start = max(0, row_index - 8)
    context: list[str] = []
    for row in matrix[start : row_index + 1]:
        context.append(" ".join(_cell_to_text(cell) for cell in row if cell))
    return context


def _extract_numeric_values(text: str) -> list[float]:
    cleaned = unicodedata.normalize("NFKC", text).strip()
    if not cleaned:
        return []

    for marker in ("下降", "上升", "减少", "增加", "decrease", "increase"):
        marker_index = cleaned.lower().find(marker.lower())
        if marker_index == 0:
            return []
        if marker_index > 0:
            cleaned = cleaned[:marker_index]
            break

    if not cleaned.strip():
        return []
    if re.fullmatch(r"\(?\d+\)?", cleaned) and len(cleaned.strip("()")) <= 2:
        return []

    values: list[float] = []
    for match in re.finditer(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", cleaned):
        token = match.group(0)
        if _token_looks_like_date_fragment(cleaned, token):
            continue
        try:
            values.append(_parse_numeric_token(token))
        except ValueError:
            continue
    return values


def _parse_numeric_token(token: str) -> float:
    negative = token.startswith("(") and token.endswith(")")
    normalized = token.strip("()").replace(",", "")
    value = float(normalized)
    return -value if negative else value


def _token_looks_like_date_fragment(line: str, token: str) -> bool:
    if "年" not in line and "月" not in line and "-" not in line:
        return False
    compact_line = _normalize_for_match(line)
    compact_token = _normalize_for_match(token)
    if compact_token and re.search(rf"{re.escape(compact_token)}(?:年|月|日)", compact_line):
        return True
    return bool(re.fullmatch(r"\d{1,2}", token) and "-" in line)


def _line_is_numeric_like(line: str) -> bool:
    if _is_footnote_marker(line):
        return True
    return bool(_extract_numeric_values(line)) and not re.search(r"[\u4e00-\u9fffA-Za-z]", line)


def _is_footnote_marker(line: str) -> bool:
    return bool(re.fullmatch(r"[\(（]?\d{1,2}[\)）]?", line.strip()))


def _looks_like_next_row_label(line: str) -> bool:
    if _line_is_numeric_like(line):
        return False
    if _is_footnote_marker(line):
        return False
    if _is_change_line(line):
        return False
    if _is_obvious_table_header(line):
        return True
    if any(_score_metric_label(field_name, line) > 0 for field_name in _METRIC_FIELDS):
        return True
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]", line)) and not _extract_numeric_values(line)


def _is_ignorable_after_values(line: str) -> bool:
    normalized = _normalize_for_match(line)
    return not normalized or normalized in {"-", "—", "注", "note", "notes"} or _is_change_line(line)


def _is_change_line(line: str) -> bool:
    normalized = unicodedata.normalize("NFKC", line).strip().lower()
    return normalized.startswith(("下降", "上升", "减少", "增加", "decrease", "increase"))


def _is_obvious_table_header(line: str) -> bool:
    normalized = _normalize_for_match(line)
    return any(
        token in normalized
        for token in (
            "2025年",
            "2024年",
            "2023年",
            "同比增减",
            "比上年",
            "上年末",
            "报告期",
            "財務比率",
            "财务比率",
        )
    )


def _strip_value_like_suffix(text: str) -> str:
    return re.sub(r"\s+\(?-?\d[\d,]*(?:\.\d+)?\)?.*$", "", text).strip() or text


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = unicodedata.normalize("NFKC", raw_line).strip()
        if not line:
            continue
        lines.append(line)
    return lines


def _cell_to_text(value: str | None) -> str:
    return unicodedata.normalize("NFKC", value or "").strip()


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"[,\.;:，。；：()\[\]{}<>％%/\\\-—_]", "", normalized)
    return normalized


def _label_contains_any_variant(normalized_label: str, labels: tuple[str, ...]) -> bool:
    return any(
        _normalize_for_match(variant) in normalized_label
        for label in labels
        for variant in _label_variants(label)
    )


def _label_variants(label: str) -> tuple[str, ...]:
    variants = [label]
    try:
        variants.append(label.encode("utf-8").decode("cp1252", errors="ignore"))
    except UnicodeError:
        pass
    return tuple(_dedupe_preserve_order(variants))


def _dedupe_preserve_order(items: list[str] | list[int]) -> list:
    seen: set[object] = set()
    deduped: list[object] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
