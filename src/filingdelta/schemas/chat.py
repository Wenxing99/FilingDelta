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


class ChatAnswerDraft(BaseModel):
    answer: str
    used_chunk_ids: list[str] = Field(default_factory=list)


class ChatCitation(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    source_path: Path
    page_number: int | None = None
    quote: str = ""
    score: float | None = None


class ChatAnswer(BaseModel):
    document_id: str
    question: str
    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_mode: Literal[
        "semantic_with_filters",
        "semantic_with_filters_and_keyword_fallback",
        "semantic_with_keyword_fallback",
    ] = "semantic_with_filters"
