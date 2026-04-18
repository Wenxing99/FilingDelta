from __future__ import annotations

from typing import Protocol

from llama_cloud import LlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, ExtractedFieldMetadata

from filingdelta.core.config import Settings, get_settings
from filingdelta.schemas.facts import (
    ExtractedFactField,
    HeadlineMetricFacts,
    HeadlineMetricsExtractionSchema,
)
from filingdelta.schemas.filing import Citation, FilingSource


class FilingFactExtractor(Protocol):
    def extract(self, source: FilingSource) -> HeadlineMetricFacts: ...


class LlamaExtractFactExtractor:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = LlamaCloud(**self._settings.llama_cloud_client_kwargs())

    def extract(self, source: FilingSource) -> HeadlineMetricFacts:
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
        document_id = source.source_path.stem.lower().replace(" ", "_")

        return HeadlineMetricFacts(
            document_id=document_id,
            source_path=source.source_path.resolve(),
            company_name=_build_fact_field(
                document_id=document_id,
                source=source,
                value=extracted.data.company_name,
                metadata=extracted.field_metadata.get("company_name"),
            ),
            fiscal_period=_build_fact_field(
                document_id=document_id,
                source=source,
                value=extracted.data.fiscal_period,
                metadata=extracted.field_metadata.get("fiscal_period"),
            ),
            unit=_build_fact_field(
                document_id=document_id,
                source=source,
                value=extracted.data.unit,
                metadata=extracted.field_metadata.get("unit"),
            ),
            revenue=_build_fact_field(
                document_id=document_id,
                source=source,
                value=extracted.data.revenue,
                metadata=extracted.field_metadata.get("revenue"),
            ),
            net_profit=_build_fact_field(
                document_id=document_id,
                source=source,
                value=extracted.data.net_profit,
                metadata=extracted.field_metadata.get("net_profit"),
            ),
        )


def get_filing_fact_extractor(settings: Settings | None = None) -> FilingFactExtractor:
    return LlamaExtractFactExtractor(settings=settings)


def _build_fact_field(
    *,
    document_id: str,
    source: FilingSource,
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
                    source_path=source.source_path.resolve(),
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
