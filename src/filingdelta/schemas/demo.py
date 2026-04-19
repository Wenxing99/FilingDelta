from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

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


class DemoRunResponse(BaseModel):
    run: DemoRun
