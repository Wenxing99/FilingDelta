from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from filingdelta.core.config import REPO_ROOT
from filingdelta.schemas.filing import FilingDocType, FilingSource, Market


class SmallDocBenchmarkEntry(BaseModel):
    source_path: Path
    company_name: str
    ticker: str | None = None
    market: Market = Market.OTHER
    doc_type: FilingDocType = FilingDocType.OTHER
    fiscal_period: str | None = None
    language: str = "zh"

    def to_filing_source(self) -> FilingSource:
        source_path = self.source_path
        if not source_path.is_absolute():
            source_path = (REPO_ROOT / source_path).resolve()

        return FilingSource(
            source_path=source_path,
            company_name=self.company_name,
            ticker=self.ticker,
            market=self.market,
            doc_type=self.doc_type,
            fiscal_period=self.fiscal_period,
            language=self.language,
        )


class SmallDocBenchmarkManifest(BaseModel):
    entries: list[SmallDocBenchmarkEntry] = Field(default_factory=list)


class BenchmarkFactFieldResult(BaseModel):
    field_name: str
    value: str | float | int | None = None
    value_present: bool = False
    citation_count: int = 0
    has_citation: bool = False
    confidence: float | None = None


class SmallDocBenchmarkDocumentResult(BaseModel):
    entry: SmallDocBenchmarkEntry
    success: bool
    document_id: str | None = None
    total_pages: int | None = None
    chunk_count: int | None = None
    populated_field_count: int = 0
    cited_field_count: int = 0
    citation_coverage: float = 0.0
    facts: list[BenchmarkFactFieldResult] = Field(default_factory=list)
    parsed_output: Path | None = None
    facts_output: Path | None = None
    error_type: str | None = None
    error_message: str | None = None


class SmallDocBenchmarkSummary(BaseModel):
    total_documents: int
    successful_documents: int
    failed_documents: int
    average_citation_coverage: float = 0.0


class SmallDocBenchmarkReport(BaseModel):
    summary: SmallDocBenchmarkSummary
    documents: list[SmallDocBenchmarkDocumentResult] = Field(default_factory=list)
