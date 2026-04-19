from __future__ import annotations

from llama_index.llms.openai import OpenAI

from filingdelta.core.config import Settings, get_settings
from filingdelta.prompts.chat_plan import CHAT_PLAN_PROMPT
from filingdelta.schemas.chat import ChatPlan, ChatRouteDecision
from filingdelta.schemas.filing import FilingDocument


class ChatPlannerAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = OpenAI(
            model=self._settings.filingdelta_llm_model,
            temperature=0,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
            strict=True,
        )

    async def plan(
        self,
        *,
        question: str,
        document: FilingDocument,
        route_decision: ChatRouteDecision,
    ) -> ChatPlan:
        return await self._llm.astructured_predict(
            ChatPlan,
            CHAT_PLAN_PROMPT,
            route=route_decision.route,
            needs_external_background=route_decision.needs_external_background,
            needs_risk_reasoning=route_decision.needs_risk_reasoning,
            rationale=route_decision.rationale,
            company_name=document.company_name,
            ticker=document.ticker or "",
            market=document.market.value,
            doc_type=document.doc_type.value,
            fiscal_period=document.fiscal_period or "",
            question=question,
        )
