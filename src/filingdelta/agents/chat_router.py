from __future__ import annotations

from llama_index.llms.openai import OpenAI

from filingdelta.core.config import Settings, get_settings
from filingdelta.prompts.chat_router import CHAT_ROUTER_PROMPT
from filingdelta.schemas.chat import ChatRouteDecision
from filingdelta.schemas.filing import FilingDocument


class ChatRouterAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = OpenAI(
            model="gpt-5-nano",
            temperature=0,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
            strict=True,
        )

    async def route(self, *, question: str, document: FilingDocument) -> ChatRouteDecision:
        return await self._llm.astructured_predict(
            ChatRouteDecision,
            CHAT_ROUTER_PROMPT,
            company_name=document.company_name,
            ticker=document.ticker or "",
            market=document.market.value,
            doc_type=document.doc_type.value,
            fiscal_period=document.fiscal_period or "",
            question=question,
        )
