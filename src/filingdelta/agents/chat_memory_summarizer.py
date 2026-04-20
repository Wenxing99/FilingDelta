from __future__ import annotations

from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI

from filingdelta.core.config import Settings, get_settings
from filingdelta.prompts.chat_memory_summary import CHAT_MEMORY_SUMMARY_PROMPT
from filingdelta.schemas.chat import ChatConversationMessage, ConversationSummary
from filingdelta.schemas.filing import FilingDocument


class ChatMemorySummarizerAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def summarize(
        self,
        *,
        document: FilingDocument,
        existing_summary: ConversationSummary,
        recent_messages: list[ChatConversationMessage],
        user_question: str,
        assistant_answer: str,
        callback_manager: CallbackManager | None = None,
    ) -> ConversationSummary:
        return await self._build_llm(callback_manager=callback_manager).astructured_predict(
            ConversationSummary,
            CHAT_MEMORY_SUMMARY_PROMPT,
            company_name=document.company_name,
            ticker=document.ticker or "",
            market=document.market.value,
            doc_type=document.doc_type.value,
            fiscal_period=document.fiscal_period or "",
            existing_summary=_format_summary(existing_summary),
            recent_messages=_format_recent_messages(recent_messages),
            user_question=user_question,
            assistant_answer=assistant_answer,
        )

    def _build_llm(self, *, callback_manager: CallbackManager | None = None) -> OpenAI:
        return OpenAI(
            model=self._settings.filingdelta_llm_model,
            temperature=0,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
            strict=True,
            callback_manager=callback_manager,
        )


def _format_summary(summary: ConversationSummary) -> str:
    parts: list[str] = []
    if summary.summary_text.strip():
        parts.append(f"Summary: {summary.summary_text.strip()}")
    if summary.discussed_terms:
        parts.append("Discussed terms: " + "; ".join(summary.discussed_terms))
    if summary.confirmed_facts:
        parts.append("Confirmed facts: " + "; ".join(summary.confirmed_facts))
    if summary.open_questions:
        parts.append("Open questions: " + "; ".join(summary.open_questions))
    return "\n".join(parts) if parts else "No existing summary."


def _format_recent_messages(messages: list[ChatConversationMessage]) -> str:
    if not messages:
        return "No recent messages."
    return "\n".join(f"{message.role}: {message.content}" for message in messages)
