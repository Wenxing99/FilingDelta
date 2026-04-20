from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from filingdelta.schemas.chat import ChatAnswer
from filingdelta.schemas.workflow import SingleFilingWorkflowResult


class DemoDocument(BaseModel):
    document_id: str
    label: str
    company_name: str
    ticker: str | None = None
    market: str
    doc_type: str
    fiscal_period: str | None = None
    language: str = "zh"
    source_kind: Literal["pdf", "html", "other"]
    source_url: str


class DemoDocumentListResponse(BaseModel):
    documents: list[DemoDocument] = Field(default_factory=list)


class CreateDemoRunRequest(BaseModel):
    document_id: str


class DemoRunIssueActionRequest(BaseModel):
    item_key: str


class DemoRunFeedbackActionRequest(BaseModel):
    feedback_category: Literal["citation", "numeric", "summary"]


class DemoChatRequest(BaseModel):
    document_id: str
    session_id: str | None = Field(default=None, min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=1000)


class DemoRunStageTelemetry(BaseModel):
    orchestrate_ms: float | None = None
    reader_ms: float | None = None
    fact_extractor_ms: float | None = None
    verifier_ms: float | None = None
    total_ms: float | None = None


class DemoRunArtifactsTelemetry(BaseModel):
    total_pages: int | None = None
    chunk_count: int | None = None
    summary_sections_count: int | None = None
    summary_points_count: int | None = None
    verification_issues_count: int | None = None
    needs_human_review: bool | None = None


class DemoRunTelemetry(BaseModel):
    succeeded: bool = False
    stage_timings: DemoRunStageTelemetry = Field(default_factory=DemoRunStageTelemetry)
    artifacts: DemoRunArtifactsTelemetry = Field(default_factory=DemoRunArtifactsTelemetry)


class DemoRun(BaseModel):
    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"] = "queued"
    stage: Literal["queued", "orchestrate", "reader", "fact_extractor", "verifier", "done", "failed"] = (
        "queued"
    )
    stage_label: str
    stage_index: int = 0
    stage_count: int = 4
    progress_message: str
    document_id: str
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    result: SingleFilingWorkflowResult | None = None
    telemetry: DemoRunTelemetry = Field(default_factory=DemoRunTelemetry)


class DemoRunResponse(BaseModel):
    run: DemoRun


class DemoChatResponse(BaseModel):
    response: ChatAnswer
