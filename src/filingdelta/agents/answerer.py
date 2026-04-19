from __future__ import annotations

from llama_index.llms.openai import OpenAI

from filingdelta.core.config import Settings, get_settings
from filingdelta.prompts.chat_answer import CHAT_ANSWER_PROMPT
from filingdelta.schemas.chat import ChatCitation, ChatPlan, ChatRouteDecision, ChatSynthesisDraft, RetrievedChunk
from filingdelta.schemas.filing import FilingDocument


class AnswererAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = OpenAI(
            model=self._settings.filingdelta_llm_model,
            temperature=0,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
            strict=True,
        )

    async def answer(
        self,
        *,
        question: str,
        standalone_question: str,
        document: FilingDocument,
        route_decision: ChatRouteDecision,
        plan: ChatPlan,
        retrieved_chunks: list[RetrievedChunk],
        external_citations: list[ChatCitation],
        external_summary: str = "",
    ) -> ChatSynthesisDraft:
        retrieved_context, document_ref_map = _build_retrieved_context(retrieved_chunks)
        external_context, external_ref_map = _build_external_context(
            external_summary=external_summary,
            external_citations=external_citations,
        )
        result = await self._llm.astructured_predict(
            ChatSynthesisDraft,
            CHAT_ANSWER_PROMPT,
            company_name=document.company_name,
            ticker=document.ticker or "",
            market=document.market.value,
            doc_type=document.doc_type.value,
            fiscal_period=document.fiscal_period or "",
            route=route_decision.route,
            analysis_mode=plan.analysis_mode,
            question=question,
            standalone_question=standalone_question,
            retrieved_context=retrieved_context,
            external_context=external_context or "No external evidence was provided.",
        )
        result.used_document_refs = _filter_refs(result.used_document_refs, document_ref_map)
        result.used_external_refs = _filter_refs(result.used_external_refs, external_ref_map)
        result.used_chunk_ids = _resolve_document_refs(result.used_document_refs, document_ref_map)
        result.used_external_citation_ids = _resolve_external_refs(result.used_external_refs, external_ref_map)
        return result


def _build_retrieved_context(retrieved_chunks: list[RetrievedChunk]) -> tuple[str, dict[str, str]]:
    parts: list[str] = []
    ref_map: dict[str, str] = {}
    for index, chunk in enumerate(retrieved_chunks, start=1):
        ref = f"DOC_{index}"
        page_label = chunk.page_number if chunk.page_number is not None else "unknown"
        ref_map[ref] = chunk.chunk_id
        parts.append(
            f"[Document evidence {ref}]\n"
            f"Page: {page_label}\n"
            f"Excerpt: {_truncate_text(chunk.text)}"
        )
    return "\n\n".join(parts), ref_map


def _filter_refs(refs: list[str], ref_map: dict[str, str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref not in ref_map or ref in seen:
            continue
        deduped.append(ref)
        seen.add(ref)
    return deduped[:4]


def _resolve_document_refs(
    refs: list[str],
    ref_map: dict[str, str],
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        chunk_id = ref_map.get(ref)
        if not chunk_id or chunk_id in seen:
            continue
        resolved.append(chunk_id)
        seen.add(chunk_id)
    return resolved[:4]


def _resolve_external_refs(
    refs: list[str],
    ref_map: dict[str, str],
) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        citation_id = ref_map.get(ref)
        if not citation_id or citation_id in seen:
            continue
        resolved.append(citation_id)
        seen.add(citation_id)
    return resolved[:4]


def _build_external_context(
    *,
    external_summary: str,
    external_citations: list[ChatCitation],
) -> tuple[str, dict[str, str]]:
    parts: list[str] = []
    ref_map: dict[str, str] = {}
    if external_summary.strip():
        parts.append(f"[External summary]\n{external_summary.strip()}")

    for index, citation in enumerate(external_citations, start=1):
        if citation.source_type != "external":
            continue
        ref = f"WEB_{index}"
        title = citation.title or citation.url or citation.citation_id
        url = citation.url or ""
        snippet = citation.snippet or ""
        ref_map[ref] = citation.citation_id
        parts.append(
            f"[External evidence {ref}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {_truncate_text(snippet, limit=320)}"
        )

    return "\n\n".join(parts), ref_map


def _truncate_text(text: str, limit: int = 700) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3].rstrip()}..."
