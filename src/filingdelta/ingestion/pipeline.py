from __future__ import annotations

from pydantic import BaseModel, Field

from filingdelta.core.config import Settings, get_settings
from filingdelta.ingestion.chunking import build_chunks
from filingdelta.ingestion.parsers import get_filing_parser
from filingdelta.schemas.filing import FilingChunk, FilingSource, ParsedFiling
from filingdelta.storage.paths import ensure_data_dirs


class IngestionResult(BaseModel):
    parsed_filing: ParsedFiling
    chunks: list[FilingChunk] = Field(default_factory=list)


class FilingIngestionPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._parser = get_filing_parser(self._settings)

    def run(self, source: FilingSource) -> IngestionResult:
        ensure_data_dirs()
        parsed_filing = self._parser.parse(source)
        chunks = build_chunks(parsed_filing)
        return IngestionResult(parsed_filing=parsed_filing, chunks=chunks)
