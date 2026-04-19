from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int | None = None
    source_path: Path
    text: str
    score: float | None = None
    retrieval_source: Literal["semantic", "keyword_fallback"] = "semantic"


class ChatRouteDecision(BaseModel):
    route: Literal["document_only", "concept_only", "mixed", "unsupported"]
    needs_external_background: bool = False
    needs_risk_reasoning: bool = False
    rationale: str = ""


class ChatPlan(BaseModel):
    analysis_mode: Literal["document_only", "concept_only", "mixed", "unsupported"]
    document_query: str | None = None
    external_query: str | None = None
    subquestions: list[str] = Field(default_factory=list)
    external_search_kind: Literal[
        "none",
        "concept",
        "background",
        "concept_and_background",
    ] = "none"


class ChatConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationSummary(BaseModel):
    summary_text: str = ""
    discussed_terms: list[str] = Field(default_factory=list)
    confirmed_facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ChatSessionState(BaseModel):
    session_id: str
    document_id: str
    recent_messages: list[ChatConversationMessage] = Field(default_factory=list)
    conversation_summary: ConversationSummary = Field(default_factory=ConversationSummary)


class ChatContextualization(BaseModel):
    standalone_question: str
    used_memory: bool = False
    resolved_references: list[str] = Field(default_factory=list)


class ChatSynthesisDraft(BaseModel):
    answer: str
    document_evidence: list[str] = Field(default_factory=list)
    external_evidence: list[str] = Field(default_factory=list)
    analysis_and_limits: list[str] = Field(default_factory=list)
    used_document_refs: list[str] = Field(default_factory=list)
    used_external_refs: list[str] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)
    used_external_citation_ids: list[str] = Field(default_factory=list)


class ChatCitation(BaseModel):
    citation_id: str
    source_type: Literal["document", "external"] = "document"
    page_number: int | None = None
    quote: str = ""
    url: str | None = None
    title: str | None = None
    snippet: str | None = None


class ExternalEvidenceResult(BaseModel):
    search_query: str
    search_kind: Literal["concept", "background", "concept_and_background"]
    answer_text: str
    citations: list[ChatCitation] = Field(default_factory=list)


class ChatAnswerSection(BaseModel):
    section_type: Literal["document_evidence", "external_evidence", "analysis_and_limits"]
    title: str
    items: list[str] = Field(default_factory=list)


class ChatAnswer(BaseModel):
    document_id: str
    session_id: str
    question: str
    answer: str
    route: Literal["document_only", "concept_only", "mixed", "unsupported"] = "document_only"
    sections: list[ChatAnswerSection] = Field(default_factory=list)
    citations: list[ChatCitation] = Field(default_factory=list)
    retrieval_mode: Literal[
        "semantic_with_filters",
        "semantic_with_keyword_fallback",
        "external_web_search",
        "external_search_unavailable",
        "mixed_document_external",
        "unsupported",
    ] = "semantic_with_filters"
