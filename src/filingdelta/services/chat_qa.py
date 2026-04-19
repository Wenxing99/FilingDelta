from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from filingdelta.agents.answerer import AnswererAgent
from filingdelta.core.config import Settings, get_settings
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.retrieval.indexer import DocumentChunkIndexer, chunk_node_id
from filingdelta.retrieval.retriever import DocumentChunkRetriever
from filingdelta.schemas.chat import ChatAnswer, ChatCitation, RetrievedChunk
from filingdelta.schemas.filing import FilingChunk, FilingSource, ParsedFiling


_KEYWORD_FALLBACK_TERMS = (
    "股息",
    "分红",
    "派息",
    "营业收入",
    "收入",
    "净利润",
    "归属于本行股东的净利润",
    "总资产",
    "客户存款",
    "贷款和垫款",
    "不良贷款率",
    "拨备覆盖率",
    "股东",
    "战略",
    "展望",
    "风险",
    "资产质量",
    "业务回顾",
    "可持续",
    "esg",
    "revenue",
    "net profit",
    "dividend",
    "shareholder",
    "strategy",
    "risk",
    "cloud",
    "wechat",
    "qq",
)


@dataclass(slots=True)
class IndexedDocumentBundle:
    source: FilingSource
    parsed_filing: ParsedFiling
    chunks: list[FilingChunk]


class ChatQAService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._qdrant_client = QdrantClient(path=str(self._settings.qdrant_path))
        self._pipeline = FilingIngestionPipeline(settings=self._settings)
        self._indexer = DocumentChunkIndexer(settings=self._settings, client=self._qdrant_client)
        self._retriever = DocumentChunkRetriever(settings=self._settings, client=self._qdrant_client)
        self._answerer = AnswererAgent(settings=self._settings)
        self._bundles: dict[str, IndexedDocumentBundle] = {}
        self._ensure_lock = asyncio.Lock()

    async def ask(
        self,
        *,
        document_id: str,
        source: FilingSource,
        question: str,
    ) -> ChatAnswer:
        question_text = question.strip()
        if not question_text:
            raise ValueError("Question must not be empty.")

        bundle = await self._ensure_document_bundle(document_id=document_id, source=source)
        semantic_chunks = await asyncio.to_thread(
            self._retriever.retrieve,
            document_id=document_id,
            question=question_text,
        )
        retrieved_chunks, retrieval_mode = _merge_keyword_fallback(
            question=question_text,
            document_id=document_id,
            semantic_chunks=semantic_chunks,
            all_chunks=bundle.chunks,
        )

        answer_draft = await self._answerer.answer(
            question=question_text,
            document=bundle.parsed_filing.document,
            retrieved_chunks=retrieved_chunks,
        )

        citations = _assemble_chat_citations(
            used_chunk_ids=answer_draft.used_chunk_ids,
            retrieved_chunks=retrieved_chunks,
        )

        return ChatAnswer(
            document_id=document_id,
            question=question_text,
            answer=answer_draft.answer.strip(),
            citations=citations,
            used_chunk_ids=answer_draft.used_chunk_ids,
            retrieval_mode=retrieval_mode,
        )

    async def _ensure_document_bundle(
        self,
        *,
        document_id: str,
        source: FilingSource,
    ) -> IndexedDocumentBundle:
        existing = self._bundles.get(document_id)
        if existing is not None:
            return existing

        async with self._ensure_lock:
            current = self._bundles.get(document_id)
            if current is not None:
                return current

            ingestion_result = await asyncio.to_thread(self._pipeline.run, source)
            await asyncio.to_thread(
                self._indexer.index_document,
                document_id=document_id,
                chunks=ingestion_result.chunks,
            )
            bundle = IndexedDocumentBundle(
                source=source,
                parsed_filing=ingestion_result.parsed_filing,
                chunks=ingestion_result.chunks,
            )
            self._bundles[document_id] = bundle
            return bundle


_chat_qa_service: ChatQAService | None = None


def get_chat_qa_service() -> ChatQAService:
    global _chat_qa_service
    if _chat_qa_service is None:
        _chat_qa_service = ChatQAService()
    return _chat_qa_service


def _merge_keyword_fallback(
    *,
    question: str,
    document_id: str,
    semantic_chunks: list[RetrievedChunk],
    all_chunks: list[FilingChunk],
) -> tuple[list[RetrievedChunk], str]:
    keyword_terms = _extract_keyword_terms(question)

    if not keyword_terms:
        return semantic_chunks, "semantic_with_filters"
    if semantic_chunks:
        return semantic_chunks, "semantic_with_filters"

    keyword_matches: list[RetrievedChunk] = []
    for chunk in all_chunks:
        effective_chunk_id = chunk_node_id(chunk, document_id=document_id)
        normalized_text = _normalize_for_match(chunk.text)
        matched_terms = [term for term in keyword_terms if term in normalized_text]
        if not matched_terms:
            continue
        keyword_matches.append(
            RetrievedChunk(
                chunk_id=effective_chunk_id,
                document_id=document_id,
                page_number=chunk.metadata.page_number,
                source_path=Path(chunk.metadata.source_path),
                text=chunk.text,
                score=float(len(matched_terms)),
                retrieval_source="keyword_fallback",
            )
        )

    keyword_matches.sort(
        key=lambda chunk: (
            -(chunk.score or 0.0),
            chunk.page_number or 0,
        )
    )

    if not keyword_matches:
        return semantic_chunks, "semantic_with_filters"

    return keyword_matches[:2], "semantic_with_keyword_fallback"


def _extract_keyword_terms(question: str) -> list[str]:
    normalized_question = _normalize_for_match(question)
    matched_terms = [term for term in _KEYWORD_FALLBACK_TERMS if _normalize_for_match(term) in normalized_question]
    if matched_terms:
        return matched_terms

    english_terms = re.findall(r"[A-Za-z]{3,}", question.lower())
    return list(dict.fromkeys(english_terms[:3]))


def _normalize_for_match(text: str) -> str:
    return "".join(text.lower().split())


def _assemble_chat_citations(
    *,
    used_chunk_ids: list[str],
    retrieved_chunks: list[RetrievedChunk],
) -> list[ChatCitation]:
    retrieved_lookup = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    citations: list[ChatCitation] = []
    for index, chunk_id in enumerate(used_chunk_ids):
        chunk = retrieved_lookup.get(chunk_id)
        if chunk is None:
            continue
        citations.append(_chunk_to_chat_citation(index=index, chunk=chunk))
    return citations


def _chunk_to_chat_citation(*, index: int, chunk: RetrievedChunk) -> ChatCitation:
    return ChatCitation(
        citation_id=f"chat-citation-{index + 1}",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source_path=chunk.source_path,
        page_number=chunk.page_number,
        quote=_truncate_quote(chunk.text),
        score=chunk.score,
    )


def _truncate_quote(text: str, limit: int = 260) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3].rstrip()}..."
