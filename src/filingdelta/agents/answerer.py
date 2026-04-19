from __future__ import annotations

from llama_index.llms.openai import OpenAI

from filingdelta.core.config import Settings, get_settings
from filingdelta.prompts.chat_answer import CHAT_ANSWER_PROMPT
from filingdelta.schemas.chat import ChatAnswerDraft, RetrievedChunk
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
        document: FilingDocument,
        retrieved_chunks: list[RetrievedChunk],
    ) -> ChatAnswerDraft:
        retrieved_context = _build_retrieved_context(retrieved_chunks)
        result = await self._llm.astructured_predict(
            ChatAnswerDraft,
            CHAT_ANSWER_PROMPT,
            company_name=document.company_name,
            ticker=document.ticker or "",
            market=document.market.value,
            doc_type=document.doc_type.value,
            fiscal_period=document.fiscal_period or "",
            question=question,
            retrieved_context=retrieved_context,
        )
        result.used_chunk_ids = _filter_chunk_ids(result.used_chunk_ids, retrieved_chunks)
        return result


def _build_retrieved_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for chunk in retrieved_chunks:
        score = f"{chunk.score:.3f}" if chunk.score is not None else "n/a"
        page_label = chunk.page_number if chunk.page_number is not None else "unknown"
        parts.append(
            f"[Chunk {chunk.chunk_id} | page={page_label} | source={chunk.retrieval_source} | score={score}]\n"
            f"{_truncate_text(chunk.text)}"
        )
    return "\n\n".join(parts)


def _filter_chunk_ids(chunk_ids: list[str], retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    available = {chunk.chunk_id for chunk in retrieved_chunks}
    deduped: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        if chunk_id not in available or chunk_id in seen:
            continue
        deduped.append(chunk_id)
        seen.add(chunk_id)
    return deduped[:4]


def _truncate_text(text: str, limit: int = 700) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3].rstrip()}..."
