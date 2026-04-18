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
    if _is_generic_fiscal_period(structured.fiscal_period.value):
        refined_period = _refine_fiscal_period(
            source=source,
            parsed_filing=parsed_filing,
            selection=selection,
        )
        if refined_period is not None:
            structured.fiscal_period = refined_period
    return structured


def _refine_fiscal_period(
    *,
    source: FilingSource,
    parsed_filing: ParsedFiling,
    selection: object,
) -> TextFactEvidence | None:
    candidate_pages = getattr(selection, "pages_for")("fiscal_period")
    page_lookup = {page.page_number: page for page in parsed_filing.pages}

    if source.fiscal_period:
        normalized_hint = _normalize_for_match(source.fiscal_period)
        for page_number in candidate_pages:
            page = page_lookup.get(page_number)
            if not page:
                continue
            if normalized_hint in _normalize_for_match(page.markdown or page.text):
                return TextFactEvidence(
                    value=source.fiscal_period,
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
        return None

    page_number, quote = best_match
    return TextFactEvidence(
        value=quote,
        evidence_page=page_number,
        evidence_quote=quote,
        confidence=0.95,
    )


def _is_generic_fiscal_period(value: str | None) -> bool:
    if not value:
        return True
    candidate = value.strip()
    return bool(re.fullmatch(r"\d{4}(?:年)?", candidate))


def _normalize_for_match(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    return normalized.lower()
