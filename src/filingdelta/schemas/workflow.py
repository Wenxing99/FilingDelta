from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import ParserKind, SummaryItem


class SummaryDraftItem(BaseModel):
    title: str
    summary: str
    evidence_page: int | None = Field(default=None)
    evidence_quote: str | None = Field(default=None)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ReaderDraftResult(BaseModel):
    items: list[SummaryDraftItem] = Field(default_factory=list)


class VerificationIssue(BaseModel):
    scope: Literal["summary", "facts"] = Field(default="summary")
    item_key: str
    message: str
    severity: Literal["warning", "review"] = Field(default="review")
    evidence_page: int | None = Field(default=None)
    evidence_quote: str | None = Field(default=None)


class VerificationResult(BaseModel):
    summary_items: list[SummaryItem] = Field(default_factory=list)
    issues: list[VerificationIssue] = Field(default_factory=list)
    needs_human_review: bool = False


class SingleFilingWorkflowResult(BaseModel):
    document_id: str
    source_path: Path
    parser_kind: ParserKind
    total_pages: int
    chunk_count: int
    reader_drafts: ReaderDraftResult = Field(default_factory=ReaderDraftResult)
    summary_items: list[SummaryItem] = Field(default_factory=list)
    headline_metrics: HeadlineMetricFacts
    verification_issues: list[VerificationIssue] = Field(default_factory=list)
    needs_human_review: bool = False
