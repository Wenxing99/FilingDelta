from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from llama_index.core.callbacks import CallbackManager
from qdrant_client import QdrantClient

from filingdelta.agents.answerer import AnswererAgent
from filingdelta.agents.chat_contextualizer import ChatContextualizerAgent
from filingdelta.agents.chat_memory_summarizer import ChatMemorySummarizerAgent
from filingdelta.agents.chat_planner import ChatPlannerAgent
from filingdelta.agents.chat_router import ChatRouterAgent
from filingdelta.core.config import Settings, get_settings
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.retrieval.indexer import DocumentChunkIndexer, chunk_node_id
from filingdelta.retrieval.retriever import DocumentChunkRetriever
from filingdelta.schemas.chat import (
    ChatAnswer,
    ChatAnswerSection,
    ChatCitation,
    ChatContextualization,
    ChatPlan,
    ChatRouteDecision,
    ChatSessionState,
    ExternalEvidenceResult,
    RetrievedChunk,
)
from filingdelta.schemas.filing import EvidenceKind, EvidenceUnit, FilingChunk, FilingSource, ParsedFiling
from filingdelta.services.chat_memory import ChatMemoryStore
from filingdelta.services.external_search import ExternalSearchError, ExternalSearchService
from filingdelta.services.chat_telemetry import ChatTelemetryRecorder


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
_SECTION_TEXT_PREFER_TERMS = (
    "如何",
    "为什么",
    "為何",
    "哪些",
    "原因",
    "归因",
    "歸因",
    "展望",
    "措施",
    "应对",
    "應對",
    "描述",
    "提到",
    "披露",
    "管控",
    "政策",
    "战略",
    "戰略",
    "转型",
    "轉型",
    "风险",
    "風險",
    "业务回顾",
    "業務回顧",
    "内容生态",
    "內容生態",
    "用户时长",
    "用戶時長",
    "视频号",
    "視頻號",
    "ai",
    "人工智能",
    "數智化",
    "数智化",
)
_PAGE_TEXT_PREFER_TERMS = (
    "多少",
    "几",
    "幾",
    "同比",
    "环比",
    "增幅",
    "增长率",
    "增長率",
    "下降",
    "金额",
    "金額",
    "单位",
    "單位",
    "每股",
    "资本开支",
    "資本開支",
    "营收",
    "营业收入",
    "營業收入",
    "净利润",
    "淨利潤",
    "roe",
    "roae",
    "roaa",
    "拨备覆盖率",
    "撥備覆蓋率",
    "不良贷款率",
    "不良貸款率",
    "资本充足率",
    "資本充足率",
)


@dataclass(slots=True)
class IndexedDocumentBundle:
    source: FilingSource
    parsed_filing: ParsedFiling
    chunks: list[FilingChunk]
    evidence_units: list[EvidenceUnit]


class ChatQAService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._qdrant_client = QdrantClient(path=str(self._settings.qdrant_path))
        self._pipeline = FilingIngestionPipeline(settings=self._settings)
        self._indexer = DocumentChunkIndexer(settings=self._settings, client=self._qdrant_client)
        self._retriever = DocumentChunkRetriever(settings=self._settings, client=self._qdrant_client)
        self._memory = ChatMemoryStore()
        self._contextualizer = ChatContextualizerAgent(settings=self._settings)
        self._memory_summarizer = ChatMemorySummarizerAgent(settings=self._settings)
        self._router = ChatRouterAgent(settings=self._settings)
        self._planner = ChatPlannerAgent(settings=self._settings)
        self._external_search = ExternalSearchService(settings=self._settings)
        self._answerer = AnswererAgent(settings=self._settings)
        self._bundles: dict[str, IndexedDocumentBundle] = {}
        self._ensure_lock = asyncio.Lock()

    async def ask(
        self,
        *,
        document_id: str,
        source: FilingSource,
        question: str,
        session_id: str | None = None,
    ) -> ChatAnswer:
        question_text = question.strip()
        if not question_text:
            raise ValueError("Question must not be empty.")
        active_session_id = (session_id or f"{document_id}-{uuid4()}").strip()
        if not active_session_id:
            raise ValueError("Session ID must not be empty.")

        telemetry = ChatTelemetryRecorder()
        bundle_exists = document_id in self._bundles
        if bundle_exists:
            bundle = await self._ensure_document_bundle(
                document_id=document_id,
                source=source,
                callback_manager=telemetry.callback_manager,
            )
        else:
            with telemetry.track("index_build_ms"):
                bundle = await self._ensure_document_bundle(
                    document_id=document_id,
                    source=source,
                    callback_manager=telemetry.callback_manager,
                )
        session_state = await self._memory.get_or_create(
            document_id=document_id,
            session_id=active_session_id,
        )
        try:
            with telemetry.track("contextualizer_ms"):
                contextualization = await self._contextualizer.contextualize(
                    question=question_text,
                    document=bundle.parsed_filing.document,
                    recent_messages=session_state.recent_messages,
                    conversation_summary=session_state.conversation_summary,
                    callback_manager=telemetry.callback_manager,
                )
        except Exception:
            contextualization = ChatContextualization(
                standalone_question=question_text,
                used_memory=False,
            )
        effective_question = _resolve_effective_question(
            original_question=question_text,
            contextualization=contextualization,
        )
        with telemetry.track("router_ms"):
            route_decision = await self._router.route(
                question=effective_question,
                document=bundle.parsed_filing.document,
                callback_manager=telemetry.callback_manager,
            )
        plan = await self._build_plan(
            question=effective_question,
            document=bundle.parsed_filing.document,
            route_decision=route_decision,
            telemetry=telemetry,
        )

        if route_decision.route == "unsupported":
            answer = _build_unsupported_answer(
                document_id=document_id,
                session_id=active_session_id,
                question=question_text,
            )
            await self._record_conversation_turn(
                session_state=session_state,
                document=bundle.parsed_filing.document,
                user_question=question_text,
                assistant_answer=answer.answer,
                telemetry=telemetry,
            )
            answer.telemetry = telemetry.build(route_type=route_decision.route, succeeded=True)
            return answer

        retrieved_chunks: list[RetrievedChunk] = []
        document_retrieval_mode: str | None = None
        if plan.document_query:
            with telemetry.track("document_retrieval_ms"):
                retrieval_strategy = _select_document_retrieval_strategy(plan.document_query)
                semantic_chunks, document_retrieval_mode = await asyncio.to_thread(
                    _retrieve_document_evidence,
                    retriever=self._retriever,
                    document_id=document_id,
                    question=plan.document_query,
                    callback_manager=telemetry.callback_manager,
                    strategy=retrieval_strategy,
                )
            if retrieval_strategy.primary_chunk_kind == EvidenceKind.PAGE_TEXT.value:
                retrieved_chunks, document_retrieval_mode = _merge_keyword_fallback(
                    question=plan.document_query,
                    document_id=document_id,
                    semantic_chunks=semantic_chunks,
                    all_chunks=bundle.chunks,
                    retrieval_mode=document_retrieval_mode,
                )
            else:
                retrieved_chunks = semantic_chunks
        telemetry.set_retrieval(
            document_top_k=6 if plan.document_query else 0,
            document_retrieved_chunks=len(retrieved_chunks),
        )

        external_result: ExternalEvidenceResult | None = None
        external_error: str | None = None
        if plan.external_query and plan.external_search_kind != "none":
            try:
                with telemetry.track("external_search_ms"):
                    external_result = await self._external_search.search(
                        question=plan.external_query,
                        search_kind=plan.external_search_kind,
                    )
                telemetry.add_web_search_usage(external_result.usage)
            except ExternalSearchError as error:
                external_error = str(error)
        telemetry.set_retrieval(
            external_sources_count=len(external_result.citations) if external_result else 0,
        )

        if route_decision.route == "concept_only" and external_result is None:
            answer = _build_external_failure_answer(
                document_id=document_id,
                session_id=active_session_id,
                question=question_text,
                error_message=external_error or "External search is unavailable.",
            )
            await self._record_conversation_turn(
                session_state=session_state,
                document=bundle.parsed_filing.document,
                user_question=question_text,
                assistant_answer=answer.answer,
                telemetry=telemetry,
            )
            answer.telemetry = telemetry.build(route_type=route_decision.route, succeeded=True)
            return answer

        with telemetry.track("answerer_ms"):
            answer_draft = await self._answerer.answer(
                question=question_text,
                standalone_question=effective_question,
                document=bundle.parsed_filing.document,
                route_decision=route_decision,
                plan=plan,
                retrieved_chunks=retrieved_chunks,
                external_citations=external_result.citations if external_result else [],
                external_summary=external_result.answer_text if external_result else "",
                callback_manager=telemetry.callback_manager,
            )
        answer_draft = _sanitize_synthesis_draft(answer_draft)

        sections = _build_answer_sections(
            route=route_decision.route,
            answer_draft=answer_draft,
            external_error=external_error,
        )
        citations = _assemble_chat_citations(
            used_chunk_ids=answer_draft.used_chunk_ids,
            retrieved_chunks=retrieved_chunks,
            used_external_citation_ids=answer_draft.used_external_citation_ids,
            external_citations=external_result.citations if external_result else [],
        )
        telemetry.set_retrieval(
            used_document_citations_count=len(
                [citation for citation in citations if citation.source_type == "document"]
            ),
            used_external_citations_count=len(
                [citation for citation in citations if citation.source_type == "external"]
            ),
        )

        answer = ChatAnswer(
            document_id=document_id,
            session_id=active_session_id,
            question=question_text,
            answer=answer_draft.answer.strip(),
            route=route_decision.route,
            sections=sections,
            citations=citations,
            retrieval_mode=_resolve_retrieval_mode(
                route=route_decision.route,
                document_retrieval_mode=document_retrieval_mode,
                has_external_result=external_result is not None,
            ),
        )
        await self._record_conversation_turn(
            session_state=session_state,
            document=bundle.parsed_filing.document,
            user_question=question_text,
            assistant_answer=answer.answer,
            telemetry=telemetry,
        )
        answer.telemetry = telemetry.build(route_type=route_decision.route, succeeded=True)
        return answer

    async def _ensure_document_bundle(
        self,
        *,
        document_id: str,
        source: FilingSource,
        callback_manager: CallbackManager | None = None,
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
                evidence_units=ingestion_result.evidence_units,
                callback_manager=callback_manager,
            )
            bundle = IndexedDocumentBundle(
                source=source,
                parsed_filing=ingestion_result.parsed_filing,
                chunks=ingestion_result.chunks,
                evidence_units=ingestion_result.evidence_units,
            )
            self._bundles[document_id] = bundle
            return bundle

    async def _build_plan(
        self,
        *,
        question: str,
        document,
        route_decision: ChatRouteDecision,
        telemetry: ChatTelemetryRecorder,
    ) -> ChatPlan:
        route = route_decision.route
        if route == "document_only":
            return ChatPlan(
                analysis_mode="document_only",
                document_query=question,
                external_search_kind="none",
            )
        if route == "concept_only":
            return ChatPlan(
                analysis_mode="concept_only",
                external_query=question,
                external_search_kind="concept",
                subquestions=[question],
            )
        if route == "unsupported":
            return ChatPlan(analysis_mode="unsupported", external_search_kind="none")

        with telemetry.track("planner_ms"):
            planned = await self._planner.plan(
                question=question,
                document=document,
                route_decision=route_decision,
                callback_manager=telemetry.callback_manager,
            )
        return _normalize_plan(planned, question=question, route_decision=route_decision)

    async def _record_conversation_turn(
        self,
        *,
        session_state: ChatSessionState,
        document,
        user_question: str,
        assistant_answer: str,
        telemetry: ChatTelemetryRecorder,
    ) -> None:
        updated_session = await self._memory.append_turn(
            document_id=session_state.document_id,
            session_id=session_state.session_id,
            user_message=user_question,
            assistant_message=assistant_answer,
        )
        try:
            with telemetry.track("memory_summarizer_ms"):
                summary = await self._memory_summarizer.summarize(
                    document=document,
                    existing_summary=updated_session.conversation_summary,
                    recent_messages=updated_session.recent_messages,
                    user_question=user_question,
                    assistant_answer=assistant_answer,
                    callback_manager=telemetry.callback_manager,
                )
            await self._memory.replace_summary(
                document_id=session_state.document_id,
                session_id=session_state.session_id,
                summary=summary,
            )
        except Exception:
            return


_chat_qa_service: ChatQAService | None = None


def get_chat_qa_service() -> ChatQAService:
    global _chat_qa_service
    if _chat_qa_service is None:
        _chat_qa_service = ChatQAService()
    return _chat_qa_service


def _normalize_plan(
    plan: ChatPlan,
    *,
    question: str,
    route_decision: ChatRouteDecision,
) -> ChatPlan:
    if plan.analysis_mode != "mixed":
        plan.analysis_mode = "mixed"

    plan.document_query = (plan.document_query or question).strip()
    plan.external_query = (plan.external_query or question).strip()
    if not plan.subquestions:
        plan.subquestions = [question]

    if plan.external_search_kind == "none":
        plan.external_search_kind = (
            "concept_and_background"
            if route_decision.needs_external_background or route_decision.needs_risk_reasoning
            else "concept"
        )
    return plan


def _resolve_effective_question(
    *,
    original_question: str,
    contextualization: ChatContextualization,
) -> str:
    candidate = contextualization.standalone_question.strip()
    if not candidate:
        return original_question
    return candidate


@dataclass(frozen=True)
class _DocumentRetrievalStrategy:
    primary_chunk_kind: str
    fallback_chunk_kind: str | None = None
    retrieval_mode: str = "semantic_with_filters"


def _select_document_retrieval_strategy(question: str) -> _DocumentRetrievalStrategy:
    normalized_question = _normalize_for_match(question)
    if any(_normalize_for_match(term) in normalized_question for term in _PAGE_TEXT_PREFER_TERMS):
        return _DocumentRetrievalStrategy(
            primary_chunk_kind=EvidenceKind.PAGE_TEXT.value,
            retrieval_mode="semantic_with_filters",
        )
    if any(_normalize_for_match(term) in normalized_question for term in _SECTION_TEXT_PREFER_TERMS):
        return _DocumentRetrievalStrategy(
            primary_chunk_kind=EvidenceKind.SECTION_TEXT.value,
            fallback_chunk_kind=EvidenceKind.PAGE_TEXT.value,
            retrieval_mode="semantic_with_filters",
        )
    return _DocumentRetrievalStrategy(
        primary_chunk_kind=EvidenceKind.PAGE_TEXT.value,
        retrieval_mode="semantic_with_filters",
    )


def _retrieve_document_evidence(
    *,
    retriever: DocumentChunkRetriever,
    document_id: str,
    question: str,
    callback_manager: CallbackManager | None,
    strategy: _DocumentRetrievalStrategy,
) -> tuple[list[RetrievedChunk], str]:
    primary_chunks = retriever.retrieve(
        document_id=document_id,
        question=question,
        chunk_kind=strategy.primary_chunk_kind,
        callback_manager=callback_manager,
    )
    if primary_chunks or strategy.fallback_chunk_kind is None:
        return primary_chunks, strategy.retrieval_mode

    fallback_chunks = retriever.retrieve(
        document_id=document_id,
        question=question,
        chunk_kind=strategy.fallback_chunk_kind,
        callback_manager=callback_manager,
    )
    if fallback_chunks:
        return fallback_chunks, strategy.retrieval_mode
    return primary_chunks, strategy.retrieval_mode


def _merge_keyword_fallback(
    *,
    question: str,
    document_id: str,
    semantic_chunks: list[RetrievedChunk],
    all_chunks: list[FilingChunk],
    retrieval_mode: str,
) -> tuple[list[RetrievedChunk], str]:
    keyword_terms = _extract_keyword_terms(question)

    if not keyword_terms:
        return semantic_chunks, retrieval_mode
    if semantic_chunks:
        return semantic_chunks, retrieval_mode

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
        return semantic_chunks, retrieval_mode

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
    used_external_citation_ids: list[str],
    external_citations: list[ChatCitation],
) -> list[ChatCitation]:
    citations: list[ChatCitation] = []

    retrieved_lookup = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    for index, chunk_id in enumerate(used_chunk_ids):
        chunk = retrieved_lookup.get(chunk_id)
        if chunk is None:
            continue
        citations.append(_chunk_to_chat_citation(index=index, chunk=chunk))

    external_lookup = {citation.citation_id: citation for citation in external_citations}
    for citation_id in used_external_citation_ids:
        citation = external_lookup.get(citation_id)
        if citation is None:
            continue
        citations.append(citation)

    return citations


def _chunk_to_chat_citation(*, index: int, chunk: RetrievedChunk) -> ChatCitation:
    return ChatCitation(
        citation_id=f"chat-citation-{index + 1}",
        source_type="document",
        page_number=chunk.page_number,
        quote=_truncate_quote(chunk.text),
    )


def _build_answer_sections(
    *,
    route: str,
    answer_draft,
    external_error: str | None,
) -> list[ChatAnswerSection]:
    sections: list[ChatAnswerSection] = []
    external_title = "外部解释"
    document_title = "文档证据"
    analysis_title = "分析与边界"

    if route == "mixed":
        external_title = "概念与外部背景"
        document_title = "文档事实"
        analysis_title = "综合分析与边界"

    if route in {"concept_only", "mixed"} and answer_draft.external_evidence:
        sections.append(
            ChatAnswerSection(
                section_type="external_evidence",
                title=external_title,
                items=answer_draft.external_evidence,
            )
        )
    if answer_draft.document_evidence:
        sections.append(
            ChatAnswerSection(
                section_type="document_evidence",
                title=document_title,
                items=answer_draft.document_evidence,
            )
        )
    if route == "document_only" and answer_draft.external_evidence:
        sections.append(
            ChatAnswerSection(
                section_type="external_evidence",
                title=external_title,
                items=answer_draft.external_evidence,
            )
        )

    analysis_items = list(answer_draft.analysis_and_limits)
    if external_error:
        analysis_items.append(
            "外部检索未完全可用，因此这次回答的外部背景信息可能不完整。"
        )
    if analysis_items:
        sections.append(
            ChatAnswerSection(
                section_type="analysis_and_limits",
                title=analysis_title,
                items=analysis_items,
            )
        )
    return sections


def _resolve_retrieval_mode(
    *,
    route: str,
    document_retrieval_mode: str | None,
    has_external_result: bool,
) -> str:
    if route == "unsupported":
        return "unsupported"
    if has_external_result and route in {"concept_only", "mixed"}:
        if route == "mixed" and document_retrieval_mode:
            return "mixed_document_external"
        return "external_web_search"
    if document_retrieval_mode:
        return document_retrieval_mode
    return "semantic_with_filters"


def _build_unsupported_answer(*, document_id: str, session_id: str, question: str) -> ChatAnswer:
    return ChatAnswer(
        document_id=document_id,
        session_id=session_id,
        question=question,
        answer="这个问题超出了当前演示系统的文档问答与概念解释范围。",
        route="unsupported",
        sections=[
            ChatAnswerSection(
                section_type="analysis_and_limits",
                title="分析与边界",
                items=["当前只支持基于公开信披材料的问答，以及与这份文档相关的外部概念解释。"],
            )
        ],
        retrieval_mode="unsupported",
    )


def _build_external_failure_answer(
    *,
    document_id: str,
    session_id: str,
    question: str,
    error_message: str,
) -> ChatAnswer:
    return ChatAnswer(
        document_id=document_id,
        session_id=session_id,
        question=question,
        answer="这个问题需要外部概念解释，但当前外部检索暂时不可用。",
        route="concept_only",
        sections=[
            ChatAnswerSection(
                section_type="analysis_and_limits",
                title="分析与边界",
                items=[error_message],
            )
        ],
        retrieval_mode="external_search_unavailable",
    )


def _truncate_quote(text: str, limit: int = 260) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3].rstrip()}..."


def _sanitize_synthesis_draft(answer_draft):
    answer_draft.answer = _sanitize_user_facing_text(answer_draft.answer)
    answer_draft.document_evidence = [
        _sanitize_user_facing_text(item)
        for item in answer_draft.document_evidence
        if _sanitize_user_facing_text(item)
    ]
    answer_draft.external_evidence = [
        _sanitize_user_facing_text(item)
        for item in answer_draft.external_evidence
        if _sanitize_user_facing_text(item)
    ]
    answer_draft.analysis_and_limits = [
        _sanitize_user_facing_text(item)
        for item in answer_draft.analysis_and_limits
        if _sanitize_user_facing_text(item)
    ]
    return answer_draft


def _sanitize_user_facing_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\[Chunk [^\]]+\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(Chunk [^)]+\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:DOC|WEB)_\d+\b", "", cleaned)
    cleaned = re.sub(r"\bsource\s*=\s*[\w-]+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bscore\s*=\s*[-+]?\d*\.?\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" -|,;：:")
