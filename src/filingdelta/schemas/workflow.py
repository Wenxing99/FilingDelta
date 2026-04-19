from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import ParserKind, SummaryItem, SummarySection


class SummaryDraftPoint(BaseModel):
    text: str
    evidence_page: int | None = Field(default=None)
    evidence_quote: str | None = Field(default=None)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SummaryDraftSection(BaseModel):
    title: str
    points: list[SummaryDraftPoint] = Field(default_factory=list)


class ReaderDraftResult(BaseModel):
    overview: SummaryDraftPoint | None = None
    sections: list[SummaryDraftSection] = Field(default_factory=list)


class VerificationIssue(BaseModel):
    scope: Literal["summary", "facts"] = Field(default="summary")
    item_key: str
    item_label: str
    message: str
    severity: Literal["warning", "review"] = Field(default="review")
    review_reason: Literal["citation_pending", "numeric_pending", "summary_incomplete"] = Field(
        default="citation_pending"
    )
    user_visible_reason: str = "引用待确认"
    evidence_page: int | None = Field(default=None)
    evidence_quote: str | None = Field(default=None)


class ReviewStatusSummary(BaseModel):
    status: Literal["passed", "needs_confirmation", "failed"] = "passed"
    verified_count: int = 0
    pending_confirmation_count: int = 0
    failed_count: int = 0


class VerificationResult(BaseModel):
    overview: SummaryItem | None = None
    summary_sections: list[SummarySection] = Field(default_factory=list)
    summary_items: list[SummaryItem] = Field(default_factory=list)
    issues: list[VerificationIssue] = Field(default_factory=list)
    needs_human_review: bool = False
    review: ReviewStatusSummary = Field(default_factory=ReviewStatusSummary)


class SingleFilingWorkflowResult(BaseModel):
    document_id: str
    source_path: Path
    parser_kind: ParserKind
    total_pages: int
    chunk_count: int
    reader_drafts: ReaderDraftResult = Field(default_factory=ReaderDraftResult)
    overview: SummaryItem | None = None
    summary_sections: list[SummarySection] = Field(default_factory=list)
    summary_items: list[SummaryItem] = Field(default_factory=list)
    headline_metrics: HeadlineMetricFacts
    verification_issues: list[VerificationIssue] = Field(default_factory=list)
    needs_human_review: bool = False
    review: ReviewStatusSummary = Field(default_factory=ReviewStatusSummary)
