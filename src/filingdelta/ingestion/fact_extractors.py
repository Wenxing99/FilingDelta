from __future__ import annotations

import re
from typing import Protocol

from llama_cloud import LlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, ExtractedFieldMetadata
from llama_index.llms.openai import OpenAI

from filingdelta.core.config import Settings, get_settings
from filingdelta.ingestion.page_locators import CandidatePageLocator
from filingdelta.prompts.fact_extraction import HEADLINE_METRICS_EXTRACTION_PROMPT
from filingdelta.schemas.fact_extraction import (
    HeadlineMetricsStructuredExtraction,
    NumericFactEvidence,
    TextFactEvidence,
)
from filingdelta.schemas.facts import (
    ExtractedFactField,
    HeadlineMetricFacts,
    HeadlineMetricsExtractionSchema,
)
from filingdelta.schemas.filing import Citation, FilingSource, ParsedFiling


class FilingFactExtractor(Protocol):
    def extract(self, source: FilingSource, parsed_filing: ParsedFiling) -> HeadlineMetricFacts: ...


class StructuredFactExtractor:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._locator = CandidatePageLocator()
        self._llm = OpenAI(
            model=self._settings.filingdelta_llm_model,
            temperature=0,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
            strict=True,
        )

    def extract(self, source: FilingSource, parsed_filing: ParsedFiling) -> HeadlineMetricFacts:
        selection = self._locator.locate(parsed_filing)
        page_context = _build_page_context(parsed_filing, selection.all_pages())

        structured = self._llm.structured_predict(
            HeadlineMetricsStructuredExtraction,
            HEADLINE_METRICS_EXTRACTION_PROMPT,
            company_name=source.company_name,
            ticker=source.ticker or "",
            market=source.market.value,
            doc_type=source.doc_type.value,
            fiscal_period=source.fiscal_period or "",
            company_name_pages=_format_pages(selection.pages_for("company_name")),
            fiscal_period_pages=_format_pages(selection.pages_for("fiscal_period")),
            unit_pages=_format_pages(selection.pages_for("unit")),
            revenue_pages=_format_pages(selection.pages_for("revenue")),
            net_profit_pages=_format_pages(selection.pages_for("net_profit")),
            roe_pages=_format_pages(selection.pages_for("roe")),
            page_context=page_context,
        )
        structured = _refine_structured_extraction(
            source=source,
            parsed_filing=parsed_filing,
            selection=selection,
            structured=structured,
        )

        document_id = parsed_filing.document.document_id
        source_path = parsed_filing.document.source_path
        return HeadlineMetricFacts(
            document_id=document_id,
            source_path=source_path,
            company_name=_build_structured_fact_field(structured.company_name),
            fiscal_period=_build_structured_fact_field(structured.fiscal_period),
            unit=_build_structured_fact_field(structured.unit),
            revenue=_build_structured_fact_field(structured.revenue),
            net_profit=_build_structured_fact_field(structured.net_profit),
            roe=_build_structured_fact_field(structured.roe),
        )


class LlamaExtractFactExtractor:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = LlamaCloud(**self._settings.llama_cloud_client_kwargs())

    def extract(self, source: FilingSource, parsed_filing: ParsedFiling) -> HeadlineMetricFacts:
        with source.source_path.open("rb") as file_handle:
            uploaded_file = self._client.files.create(file=file_handle, purpose="extract")

        job = self._client.extract.run(
            file_input=uploaded_file.id,
            configuration={
                "data_schema": HeadlineMetricsExtractionSchema.model_json_schema(),
                "cite_sources": True,
                "confidence_scores": True,
                "extract_version": "latest",
                "extraction_target": "per_doc",
                "parse_tier": self._settings.filingdelta_llama_parse_tier,
                "tier": self._settings.filingdelta_llama_extract_tier,
            },
            verbose=True,
        )
        job_with_metadata = self._client.extract.get(
            job.id,
            expand=["extract_metadata"],
        )

        extracted = ExtractedData.from_extract_job(
            job_with_metadata,
            HeadlineMetricsExtractionSchema,
            file_id=uploaded_file.id,
            file_name=source.source_path.name,
        )
        document_id = parsed_filing.document.document_id

        return HeadlineMetricFacts(
            document_id=document_id,
            source_path=parsed_filing.document.source_path,
            company_name=_build_fact_field(
                document_id=document_id,
                source_path=parsed_filing.document.source_path,
                value=extracted.data.company_name,
                metadata=extracted.field_metadata.get("company_name"),
            ),
            fiscal_period=_build_fact_field(
                document_id=document_id,
                source_path=parsed_filing.document.source_path,
                value=extracted.data.fiscal_period,
                metadata=extracted.field_metadata.get("fiscal_period"),
            ),
            unit=_build_fact_field(
                document_id=document_id,
                source_path=parsed_filing.document.source_path,
                value=extracted.data.unit,
                metadata=extracted.field_metadata.get("unit"),
            ),
            revenue=_build_fact_field(
                document_id=document_id,
                source_path=parsed_filing.document.source_path,
                value=extracted.data.revenue,
                metadata=extracted.field_metadata.get("revenue"),
            ),
            net_profit=_build_fact_field(
                document_id=document_id,
                source_path=parsed_filing.document.source_path,
                value=extracted.data.net_profit,
                metadata=extracted.field_metadata.get("net_profit"),
            ),
            roe=_build_fact_field(
                document_id=document_id,
                source_path=parsed_filing.document.source_path,
                value=extracted.data.roe,
                metadata=extracted.field_metadata.get("roe"),
            ),
        )


def get_filing_fact_extractor(settings: Settings | None = None) -> FilingFactExtractor:
    resolved_settings = settings or get_settings()
    if resolved_settings.filingdelta_extract_provider == "llama_extract":
        return LlamaExtractFactExtractor(settings=resolved_settings)
    return StructuredFactExtractor(settings=resolved_settings)


def _build_fact_field(
    *,
    document_id: str,
    source_path: object,
    value: str | float | int | None,
    metadata: object,
) -> ExtractedFactField:
    field_metadata = metadata if isinstance(metadata, ExtractedFieldMetadata) else None
    citations: list[Citation] = []

    if field_metadata and field_metadata.citation:
        for citation in field_metadata.citation:
            citations.append(
                Citation(
                    document_id=document_id,
                    source_path=source_path,
                    page_number=citation.page,
                    quote=citation.matching_text or "",
                )
            )

    return ExtractedFactField(
        value=value,
        reasoning=field_metadata.reasoning if field_metadata else None,
        confidence=field_metadata.confidence if field_metadata else None,
        citations=citations,
    )


def _build_structured_fact_field(
    evidence: TextFactEvidence | NumericFactEvidence,
) -> ExtractedFactField:
    return ExtractedFactField(
        value=evidence.value,
        confidence=evidence.confidence,
        evidence_page=evidence.evidence_page,
        evidence_quote=evidence.evidence_quote,
    )


def _build_page_context(parsed_filing: ParsedFiling, page_numbers: list[int]) -> str:
    if not page_numbers:
        page_numbers = [page.page_number for page in parsed_filing.pages[: min(4, len(parsed_filing.pages))]]

    parts: list[str] = []
    page_lookup = {page.page_number: page for page in parsed_filing.pages}
    for page_number in page_numbers:
        page = page_lookup.get(page_number)
        if not page:
            continue
        page_text = (page.markdown or page.text).strip()
        if not page_text:
            continue
        parts.append(f"[Page {page_number}]\n{_truncate_text(page_text)}")

    return "\n\n".join(parts)


def _truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _format_pages(page_numbers: list[int]) -> str:
    return ", ".join(str(page_number) for page_number in page_numbers) or "none"


def _refine_structured_extraction(
    *,
    source: FilingSource,
    parsed_filing: ParsedFiling,
    selection: object,
    structured: HeadlineMetricsStructuredExtraction,
) -> HeadlineMetricsStructuredExtraction:
    refined_period = _refine_fiscal_period(
        source=source,
        parsed_filing=parsed_filing,
        selection=selection,
        current=structured.fiscal_period,
    )
    if refined_period is not None:
        structured.fiscal_period = refined_period
    refined_net_profit = _refine_net_profit(
        parsed_filing=parsed_filing,
        selection=selection,
        current=structured.net_profit,
    )
    if refined_net_profit is not None:
        structured.net_profit = refined_net_profit
    structured.roe = _refine_roe(
        parsed_filing=parsed_filing,
        selection=selection,
        current=structured.roe,
    )
    return structured


def _refine_fiscal_period(
    *,
    source: FilingSource,
    parsed_filing: ParsedFiling,
    selection: object,
    current: TextFactEvidence,
) -> TextFactEvidence | None:
    candidate_pages = getattr(selection, "pages_for")("fiscal_period")
    page_lookup = {page.page_number: page for page in parsed_filing.pages}
    standardized_label = _build_standardized_fiscal_period_label(source)

    if standardized_label and source.fiscal_period:
        normalized_hint = _normalize_for_match(source.fiscal_period)
        for page_number in candidate_pages:
            page = page_lookup.get(page_number)
            if not page:
                continue
            if normalized_hint in _normalize_for_match(page.markdown or page.text):
                return TextFactEvidence(
                    value=standardized_label,
                    evidence_page=page_number,
                    evidence_quote=source.fiscal_period,
                    confidence=0.99,
                )

    period_patterns = (
        r"\d{4}\s*年度报告摘要",
        r"\d{4}\s*年度报告",
        r"\d{4}\s*年半年度报告",
        r"\d{4}\s*年(?:第)?一季度报告",
        r"\d{4}\s*年(?:第)?二季度报告",
        r"\d{4}\s*年(?:第)?三季度报告",
        r"\d{4}\s*年(?:第)?四季度报告",
        r"截至[^。\n]{0,50}?止年度[^。\n]{0,30}?(?:业绩公布|業績公佈)",
        r"(?:Fourth Quarter\s+\d{4}\s+and\s+)?Fiscal Year\s+\d{4}[^\n]{0,80}?Results",
    )

    best_match: tuple[int, str] | None = None
    for page_number in candidate_pages:
        page = page_lookup.get(page_number)
        if not page:
            continue
        page_text = page.markdown or page.text
        for pattern in period_patterns:
            for match in re.finditer(pattern, page_text, flags=re.IGNORECASE):
                quote = match.group(0).strip()
                if not quote:
                    continue
                if best_match is None or len(quote) > len(best_match[1]):
                    best_match = (page_number, quote)

    if best_match is None:
        if standardized_label:
            return TextFactEvidence(
                value=standardized_label,
                evidence_page=current.evidence_page,
                evidence_quote=current.evidence_quote or source.fiscal_period,
                confidence=current.confidence or 0.9,
            )
        return None

    page_number, quote = best_match
    return TextFactEvidence(
        value=standardized_label or quote,
        evidence_page=page_number,
        evidence_quote=quote,
        confidence=0.95,
    )


def _is_generic_fiscal_period(value: str | None) -> bool:
    if not value:
        return True
    candidate = value.strip()
    return bool(re.fullmatch(r"\d{4}(?:年|年度)?", candidate))


def _refine_net_profit(
    *,
    parsed_filing: ParsedFiling,
    selection: object,
    current: NumericFactEvidence,
) -> NumericFactEvidence | None:
    candidate_pages = getattr(selection, "pages_for")("net_profit")
    page_lookup = {page.page_number: page for page in parsed_filing.pages}

    for page_number in candidate_pages:
        page = page_lookup.get(page_number)
        if not page:
            continue
        page_text = page.markdown or page.text
        for window_text in _iter_text_windows(page_text):
            extracted = _extract_attributable_net_profit(window_text)
            if extracted is None:
                continue
            value, quote = extracted
            if current.value is not None and _numeric_equal(current.value, value):
                return NumericFactEvidence(
                    value=current.value,
                    evidence_page=page_number,
                    evidence_quote=quote,
                    confidence=max(current.confidence or 0.0, 0.98),
                )
            return NumericFactEvidence(
                value=value,
                evidence_page=page_number,
                evidence_quote=quote,
                confidence=max(current.confidence or 0.0, 0.98),
            )

    return None


def _iter_text_windows(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    windows: list[str] = []
    for index, line in enumerate(lines):
        windows.append(line)
        if index + 1 < len(lines):
            windows.append(f"{line} {lines[index + 1]}")
    return windows


def _iter_roe_windows(text: str) -> list[str]:
    normalized_text = _normalize_whitespace(text)
    if not normalized_text:
        return []

    windows: list[str] = []
    for anchor in (
        "roae",
        "return on equity",
        "净资产收益率",
        "淨資產收益率",
        "加权平均净资产收益率",
        "加權平均淨資產收益率",
    ):
        for match in re.finditer(re.escape(anchor), normalized_text, flags=re.IGNORECASE):
            start = max(0, match.start() - 40)
            end = min(len(normalized_text), match.end() + 160)
            windows.append(normalized_text[start:end].strip())

    if not windows:
        return [normalized_text]

    windows.append(normalized_text)
    return _dedupe_preserve_text_order(windows)


def _extract_attributable_net_profit(text: str) -> tuple[float, str] | None:
    excluded_markers = (
        "扣除非经常性损益后",
        "扣除非经常性损益後",
        "excluding non-recurring",
        "excluding nonrecurring",
    )
    patterns = (
        r"(归属于本行股东的净利润[^\n]{0,80})",
        r"(归属于母公司股东的净利润[^\n]{0,80})",
        r"(归属于本公司股东的净利润[^\n]{0,80})",
        r"(归属于普通股股东的净利润[^\n]{0,80})",
        r"(归属于本行普通股股东的净利润[^\n]{0,80})",
        r"(本公司权益持有人应占盈利[^\n]{0,80})",
        r"(本公司权益持有人應佔盈利[^\n]{0,80})",
        r"(profit attributable to [^\n]{0,80})",
        r"(net income attributable to [^\n]{0,80})",
    )

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        quote = match.group(1).strip()
        prefix = text[max(0, match.start() - 24) : match.start()]
        lower_quote = quote.lower()
        lower_prefix = prefix.lower()
        if any(
            marker in quote
            or marker in lower_quote
            or marker in prefix
            or marker in lower_prefix
            for marker in excluded_markers
        ):
            continue
        value_match = re.search(r"-?\d[\d,]*(?:\.\d+)?", quote)
        if not value_match:
            continue
        try:
            return float(value_match.group(0).replace(",", "")), quote
        except ValueError:
            continue
    return None


def _numeric_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) < 1e-9


def _refine_roe(
    *,
    parsed_filing: ParsedFiling,
    selection: object,
    current: NumericFactEvidence,
) -> NumericFactEvidence:
    current_match = _extract_preferred_roe(current.evidence_quote or "")
    if current_match is not None:
        value, quote, priority = current_match
        if current.value is not None and _numeric_equal(current.value, value):
            return NumericFactEvidence(
                value=current.value,
                evidence_page=current.evidence_page,
                evidence_quote=current.evidence_quote or quote,
                confidence=max(current.confidence or 0.0, min(0.99, 0.9 + priority * 0.02)),
            )

    candidate_pages = getattr(selection, "pages_for")("roe")
    page_lookup = {page.page_number: page for page in parsed_filing.pages}
    best_match: tuple[int, float, str, int] | None = None

    for page_number in candidate_pages:
        page = page_lookup.get(page_number)
        if not page:
            continue
        page_text = page.markdown or page.text
        for window_text in _iter_roe_windows(page_text):
            extracted = _extract_preferred_roe(window_text)
            if extracted is None:
                continue
            value, quote, priority = extracted
            if best_match is None or priority > best_match[3]:
                best_match = (page_number, value, quote, priority)

    if best_match is None:
        return NumericFactEvidence()

    page_number, value, quote, priority = best_match
    resolved_value = current.value if _numeric_equal(current.value, value) else value
    return NumericFactEvidence(
        value=resolved_value,
        evidence_page=page_number,
        evidence_quote=quote,
        confidence=max(current.confidence or 0.0, min(0.99, 0.9 + priority * 0.02)),
    )


def _extract_preferred_roe(text: str) -> tuple[float, str, int] | None:
    normalized_text = _normalize_whitespace(text)
    if not normalized_text:
        return None

    has_roe_anchor = _contains_any(
        normalized_text,
        (
            "roae",
            "roe",
            "return on equity",
            "weighted average return on equity",
            "净资产收益率",
            "淨資產收益率",
            "加权平均净资产收益率",
            "加權平均淨資產收益率",
        ),
    )
    if not has_roe_anchor:
        return None

    if _contains_any(normalized_text, ("roaa", "return on assets", "总资产收益率", "總資產收益率")):
        paired_match = _extract_paired_roaa_roae(normalized_text)
        if paired_match is not None:
            return paired_match

    priority_markers = (
        (
            4,
            (
                "年化",
                ("扣除非经常性损益后", "扣除非經常性損益後"),
                ("普通股股东", "普通股股東"),
                ("加权平均净资产收益率", "加權平均淨資產收益率"),
            ),
        ),
        (
            3,
            (
                ("扣除非经常性损益后", "扣除非經常性損益後"),
                ("普通股股东", "普通股股東"),
                ("加权平均净资产收益率", "加權平均淨資產收益率"),
            ),
        ),
        (
            2,
            (
                ("普通股股东", "普通股股東"),
                ("加权平均净资产收益率", "加權平均淨資產收益率"),
            ),
        ),
        (
            1,
            (
                ("加权平均净资产收益率", "加權平均淨資產收益率", "净资产收益率", "淨資產收益率"),
                ("return on equity", "weighted average return on equity", "roe"),
            ),
        ),
    )

    detected_priority = 0
    for priority, groups in priority_markers:
        if all(_contains_any(normalized_text, group) for group in groups):
            detected_priority = priority
            break

    if detected_priority <= 0:
        for priority, groups in priority_markers:
            if any(_contains_any(normalized_text, group) for group in groups):
                detected_priority = priority
                break

    if detected_priority <= 0:
        return None

    explicit_patterns = (
        r"(?P<quote>(?:归属于[^。]{0,60}?普通股股东[^。]{0,40}?净资产收益率(?:\(ROAE\))?|归属于[^。]{0,60}?普通股股東[^。]{0,40}?淨資產收益率(?:\(ROAE\))?|加权平均净资产收益率(?:\(ROAE\))?|加權平均淨資產收益率(?:\(ROAE\))?|净资产收益率(?:\(ROAE\))?|淨資產收益率(?:\(ROAE\))?|weighted average return on equity|return on equity|ROAE|ROE)[^%]{0,120}?(?P<value>-?\d[\d,]*(?:\.\d+)?)\s*(?:%|％))",
    )

    for pattern in explicit_patterns:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE)
        if not match:
            continue
        quote = match.group("quote").strip()
        if _contains_any(quote, ("同比增长", "同比下降", "year-on-year", "yoy")) and not _contains_any(
            quote,
            ("分别为", "分別為", "roae", "roe", "净资产收益率", "淨資產收益率"),
        ):
            continue
        if _contains_any(quote, ("roaa", "return on assets", "总资产收益率", "總資產收益率")) and not _contains_any(
            quote,
            ("roae", "roe", "净资产收益率", "淨資產收益率", "return on equity"),
        ):
            continue
        try:
            value = float(match.group("value").replace(",", ""))
        except ValueError:
            continue
        return value, quote, detected_priority

    return None


def _extract_paired_roaa_roae(text: str) -> tuple[float, str, int] | None:
    normalized_text = _normalize_whitespace(text)
    if "分别" not in normalized_text and "分別" not in normalized_text:
        return None
    if not _contains_any(normalized_text, ("roaa", "return on assets", "总资产收益率", "總資產收益率")):
        return None
    if not _contains_any(normalized_text, ("roae", "return on equity", "净资产收益率", "淨資產收益率")):
        return None

    pair_match = re.search(
        r"分\s*(?:别|別)\s*为\s*(-?\d[\d,]*(?:\.\d+)?)\s*(?:%|％)\s*(?:和|及)\s*(-?\d[\d,]*(?:\.\d+)?)\s*(?:%|％)",
        normalized_text,
        flags=re.IGNORECASE,
    )
    if not pair_match:
        return None

    roaa_position = _first_keyword_position(normalized_text, ("roaa", "return on assets", "总资产收益率", "總資產收益率"))
    roae_position = _first_keyword_position(normalized_text, ("roae", "return on equity", "净资产收益率", "淨資產收益率"))
    if roaa_position is None or roae_position is None:
        return None

    paired_values = [pair_match.group(1), pair_match.group(2)]
    selected_index = 1 if roae_position > roaa_position else 0

    try:
        value = float(paired_values[selected_index].replace(",", ""))
    except ValueError:
        return None

    return value, pair_match.group(0).strip(), 4


def _contains_any(text: str, candidates: tuple[str, ...] | str) -> bool:
    if isinstance(candidates, str):
        return candidates.lower() in text.lower()
    lowered = text.lower()
    return any(candidate.lower() in lowered for candidate in candidates)


def _normalize_for_match(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    return normalized.lower()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_standardized_fiscal_period_label(source: FilingSource) -> str | None:
    raw_period = source.fiscal_period or ""
    year_match = re.search(r"(20\d{2})", raw_period)
    year = year_match.group(1) if year_match else None
    market_suffix = ""
    if source.market.value == "a_share":
        market_suffix = "（A股）"
    elif source.market.value == "h_share":
        market_suffix = "（H股）"

    if source.doc_type.value == "annual_report" and year:
        return f"{year}年度报告{market_suffix}"
    if source.doc_type.value == "interim_report" and year:
        quarter_label = _infer_standardized_quarter_label(raw_period, year)
        if quarter_label:
            return f"{quarter_label}{market_suffix}"
        interim_label = _infer_standardized_interim_label(raw_period, year)
        if interim_label:
            return f"{interim_label}{market_suffix}"
    if source.doc_type.value == "earnings_release" and year:
        return f"{year}业绩公告{market_suffix}"
    return None


def _infer_standardized_quarter_label(raw_period: str, year: str) -> str | None:
    quarter_match = re.search(r"Q([1-4])", raw_period, flags=re.IGNORECASE)
    chinese_match = re.search(r"第([一二三四1-4])季度", raw_period)
    plain_match = re.search(r"([1-4])季度", raw_period)
    token = (
        quarter_match.group(1)
        if quarter_match
        else chinese_match.group(1)
        if chinese_match
        else plain_match.group(1)
        if plain_match
        else None
    )
    if token is None:
        return None
    return f"{year}年{_normalize_quarter_token(token)}季度报告"


def _infer_standardized_interim_label(raw_period: str, year: str) -> str | None:
    if re.search(r"(H1|半年|半年度)", raw_period, flags=re.IGNORECASE):
        return f"{year}年半年度报告"
    if re.search(r"中期", raw_period, flags=re.IGNORECASE):
        return f"{year}年中期报告"
    return None


def _normalize_quarter_token(token: str) -> str:
    if token in {"1", "一"}:
        return "第一"
    if token in {"2", "二"}:
        return "第二"
    if token in {"3", "三"}:
        return "第三"
    if token in {"4", "四"}:
        return "第四"
    return token


def _first_keyword_position(text: str, candidates: tuple[str, ...]) -> int | None:
    lowered = text.lower()
    positions = [lowered.find(candidate.lower()) for candidate in candidates if lowered.find(candidate.lower()) >= 0]
    if not positions:
        return None
    return min(positions)


def _dedupe_preserve_text_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
