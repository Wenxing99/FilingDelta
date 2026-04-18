from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from filingdelta.schemas.benchmark import (
    BenchmarkFactFieldResult,
    SmallDocBenchmarkDocumentResult,
    SmallDocBenchmarkEntry,
    SmallDocBenchmarkManifest,
    SmallDocBenchmarkReport,
    SmallDocBenchmarkSummary,
)
from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.services.single_filing import SingleFilingProcessor, SingleFilingRunResult
from filingdelta.storage.paths import ensure_data_dirs


_HEADLINE_FACT_FIELDS = (
    "company_name",
    "fiscal_period",
    "unit",
    "revenue",
    "net_profit",
)


class SmallDocBenchmarkRunResult(BaseModel):
    report: SmallDocBenchmarkReport
    report_path: Path


class SmallDocBenchmarkProcessor:
    def __init__(self, processor: SingleFilingProcessor | None = None) -> None:
        self._processor = processor or SingleFilingProcessor()

    def run(self, manifest: SmallDocBenchmarkManifest) -> SmallDocBenchmarkRunResult:
        paths = ensure_data_dirs()
        documents: list[SmallDocBenchmarkDocumentResult] = []

        for entry in manifest.entries:
            documents.append(self._run_entry(entry))

        summary = _build_summary(documents)
        report = SmallDocBenchmarkReport(summary=summary, documents=documents)
        report_path = paths["outputs"] / "small_doc_benchmark.summary.json"
        _write_model_json(report_path, report)

        return SmallDocBenchmarkRunResult(report=report, report_path=report_path)

    def _run_entry(self, entry: SmallDocBenchmarkEntry) -> SmallDocBenchmarkDocumentResult:
        try:
            result = self._processor.run(entry.to_filing_source())
        except Exception as exc:
            return SmallDocBenchmarkDocumentResult(
                entry=entry,
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        return _build_document_result(entry, result)


def load_manifest(path: Path) -> SmallDocBenchmarkManifest:
    return SmallDocBenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _build_document_result(
    entry: SmallDocBenchmarkEntry,
    result: SingleFilingRunResult,
) -> SmallDocBenchmarkDocumentResult:
    fact_results = _build_fact_results(result.facts)
    populated_field_count = sum(1 for fact in fact_results if fact.value_present)
    cited_field_count = sum(1 for fact in fact_results if fact.value_present and fact.has_citation)
    citation_coverage = (
        cited_field_count / populated_field_count if populated_field_count else 0.0
    )

    return SmallDocBenchmarkDocumentResult(
        entry=entry,
        success=True,
        document_id=result.ingestion.parsed_filing.document.document_id,
        total_pages=len(result.ingestion.parsed_filing.pages),
        chunk_count=len(result.ingestion.chunks),
        populated_field_count=populated_field_count,
        cited_field_count=cited_field_count,
        citation_coverage=round(citation_coverage, 4),
        facts=fact_results,
        parsed_output=result.artifacts.parsed_path,
        facts_output=result.artifacts.facts_path,
    )


def _build_fact_results(facts: HeadlineMetricFacts) -> list[BenchmarkFactFieldResult]:
    fact_results: list[BenchmarkFactFieldResult] = []

    for field_name in _HEADLINE_FACT_FIELDS:
        fact = getattr(facts, field_name)
        value_present = fact.value is not None
        citation_count = len(fact.citations)
        fact_results.append(
            BenchmarkFactFieldResult(
                field_name=field_name,
                value=fact.value,
                value_present=value_present,
                citation_count=citation_count,
                has_citation=citation_count > 0,
                confidence=fact.confidence,
            )
        )

    return fact_results


def _build_summary(
    documents: list[SmallDocBenchmarkDocumentResult],
) -> SmallDocBenchmarkSummary:
    total_documents = len(documents)
    successful_documents = sum(1 for document in documents if document.success)
    failed_documents = total_documents - successful_documents

    successful_coverages = [
        document.citation_coverage for document in documents if document.success
    ]
    average_citation_coverage = (
        sum(successful_coverages) / len(successful_coverages) if successful_coverages else 0.0
    )

    return SmallDocBenchmarkSummary(
        total_documents=total_documents,
        successful_documents=successful_documents,
        failed_documents=failed_documents,
        average_citation_coverage=round(average_citation_coverage, 4),
    )


def _write_model_json(path: Path, model: BaseModel) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
