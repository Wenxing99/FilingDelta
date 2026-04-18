from __future__ import annotations

from llama_index.core.workflow import Event

from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.workflow import ReaderDraftResult


class WorkflowProgressEvent(Event):
    stage: str
    message: str


class PreparedFilingEvent(Event):
    document_id: str


class ReaderCompletedEvent(Event):
    reader_drafts: ReaderDraftResult


class FactExtractionCompletedEvent(Event):
    facts: HeadlineMetricFacts
