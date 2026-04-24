from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Market(str, Enum):
    A_SHARE = "a_share"
    H_SHARE = "h_share"
    ADR = "adr"
    OTHER = "other"


class FilingDocType(str, Enum):
    ANNUAL_REPORT = "annual_report"
    INTERIM_REPORT = "interim_report"
    EARNINGS_RELEASE = "earnings_release"
    EARNINGS_PREVIEW = "earnings_preview"
    RESPONSE_LETTER = "response_letter"
    FORM_20F = "20f"
    FORM_6K = "6k"
    OTHER = "other"


class ParserKind(str, Enum):
    LLAMA_PARSE = "llama_parse"
    PYMUPDF = "pymupdf"
    HTML_TAG = "html_tag"
    UNSTRUCTURED = "unstructured"
    FALLBACK = "fallback"


class EvidenceKind(str, Enum):
    PAGE_TEXT = "page_text"
    SECTION_TEXT = "section_text"
    TABLE_ROW = "table_row"


class FilingSource(BaseModel):
    source_path: Path
    company_name: str
    ticker: str | None = None
    market: Market = Market.OTHER
    doc_type: FilingDocType = FilingDocType.OTHER
    fiscal_period: str | None = None
    language: str = "zh"


class FilingDocument(BaseModel):
    document_id: str
    company_name: str
    ticker: str | None = None
    market: Market = Market.OTHER
    doc_type: FilingDocType = FilingDocType.OTHER
    fiscal_period: str | None = None
    language: str = "zh"
    source_path: Path
    parser_kind: ParserKind
    total_pages: int = 0


class ParsedPage(BaseModel):
    page_number: int
    text: str
    markdown: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ParsedFiling(BaseModel):
    document: FilingDocument
    pages: list[ParsedPage]


class ChunkMetadata(BaseModel):
    document_id: str
    company_name: str
    ticker: str | None = None
    market: Market = Market.OTHER
    doc_type: FilingDocType = FilingDocType.OTHER
    fiscal_period: str | None = None
    source_path: Path
    page_number: int
    chunk_index: int
    parser_kind: ParserKind


class FilingChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class EvidenceMetadata(BaseModel):
    document_id: str
    source_path: Path
    page_number: int
    page_end: int | None = None
    parser_kind: ParserKind
    chunk_kind: EvidenceKind
    section_title: str | None = None
    section_type: str | None = None
    table_id: str | None = None
    row_label: str | None = None
    metric_tags: list[str] = Field(default_factory=list)
    period_hint: str | None = None


class EvidenceUnit(BaseModel):
    evidence_id: str
    text: str
    metadata: EvidenceMetadata


class Citation(BaseModel):
    document_id: str
    source_path: Path
    page_number: int | None = None
    quote: str = ""


class SummaryItem(BaseModel):
    title: str
    summary: str
    citations: list[Citation] = Field(default_factory=list)
    needs_human_review: bool = False


class SummaryPoint(BaseModel):
    point_id: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    verification_status: Literal["verified", "review"] = "verified"
    needs_human_review: bool = False


class SummarySection(BaseModel):
    section_id: str
    title: str
    points: list[SummaryPoint] = Field(default_factory=list)
    needs_human_review: bool = False


class DiffItem(BaseModel):
    topic: str
    description: str
    document_a_citations: list[Citation] = Field(default_factory=list)
    document_b_citations: list[Citation] = Field(default_factory=list)
    needs_human_review: bool = False
