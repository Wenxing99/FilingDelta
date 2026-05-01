from __future__ import annotations

import json
from typing import Any, Literal

import httpx

from filingdelta.core.config import Settings, get_settings
from filingdelta.schemas.chat import ChatRouteDecision
from filingdelta.schemas.filing import FilingDocument


DocumentEvidenceIntent = Literal[
    "metric_value",
    "metric_attribution",
    "business_narrative",
    "fallback",
]

_METRIC_ATTRIBUTION_TOPIC_TERMS = (
    "经营指标",
    "运营指标",
    "财务指标",
    "存货",
    "库存",
    "渠道库存",
)
_METRIC_ATTRIBUTION_EFFICIENCY_TERMS = (
    "渠道效率",
    "周转",
    "周转率",
    "存货周转",
)
_METRIC_ATTRIBUTION_ABNORMAL_TERMS = (
    "异常",
    "是否异常",
    "偏高",
    "偏低",
)
_METRIC_ATTRIBUTION_CAUSE_TERMS = (
    "原因",
    "驱动",
    "为什么",
    "为何",
)
_METRIC_ATTRIBUTION_CHANGE_TERMS = (
    "变化",
    "变动",
    "提升",
    "下降",
    "改善",
    "恶化",
)
_METRIC_ATTRIBUTION_DISCUSSION_TERMS = (
    "解释",
    "描述",
)


def infer_direct_router_document_evidence_intent(
    question: str,
) -> DocumentEvidenceIntent | None:
    """Return deterministic narrow-slice intent hints for direct-router parity."""

    normalized_question = "".join(question.lower().split())
    has_topic = any(
        term in normalized_question for term in _METRIC_ATTRIBUTION_TOPIC_TERMS
    )
    has_efficiency_or_turnover = any(
        term in normalized_question for term in _METRIC_ATTRIBUTION_EFFICIENCY_TERMS
    )
    has_abnormal_signal = any(
        term in normalized_question for term in _METRIC_ATTRIBUTION_ABNORMAL_TERMS
    )
    has_cause_signal = any(
        term in normalized_question for term in _METRIC_ATTRIBUTION_CAUSE_TERMS
    )
    has_change_signal = any(
        term in normalized_question for term in _METRIC_ATTRIBUTION_CHANGE_TERMS
    )
    has_discussion_signal = any(
        term in normalized_question for term in _METRIC_ATTRIBUTION_DISCUSSION_TERMS
    )
    has_change_discussion = has_change_signal and has_discussion_signal
    has_efficiency_discussion = has_efficiency_or_turnover and has_discussion_signal
    if (has_topic or has_efficiency_or_turnover) and (
        has_abnormal_signal
        or has_cause_signal
        or has_change_discussion
        or has_efficiency_discussion
    ):
        return "metric_attribution"
    return None


class DirectJsonChatRouterAgent:
    """Small direct-OpenAI router used for exploratory router bake-off experiments."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str = "gpt-5-nano",
        response_mode: Literal["json_schema", "json_object"] = "json_schema",
        timeout: float = 30.0,
    ) -> None:
        self._settings = settings or get_settings()
        self._model = model
        self._response_mode = response_mode
        self._timeout = timeout

    async def route(
        self,
        *,
        question: str,
        document: FilingDocument,
    ) -> ChatRouteDecision:
        # Do not add `temperature=0` here. The gpt-5-nano Chat Completions
        # endpoint rejects non-default temperature values, so this direct
        # wrapper is useful as an exploratory backend comparison, not as a
        # strict deterministic replacement test for the LlamaIndex router.
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": _build_direct_router_system_prompt(self._response_mode),
                },
                {
                    "role": "user",
                    "content": _build_direct_router_user_prompt(
                        question=question,
                        document=document,
                    ),
                },
            ],
            "response_format": _build_response_format(self._response_mode),
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                _chat_completions_url(self._settings.openai_base_url),
                headers={
                    "Authorization": f"Bearer {self._settings.require_openai_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Direct router request failed: {response.status_code} {response.text}"
            )
        return parse_direct_router_response(response.json())


def parse_direct_router_response(payload: dict[str, Any]) -> ChatRouteDecision:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Direct router response is missing message content.") from error

    try:
        raw_decision = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Direct router response content is not valid JSON.") from error

    return ChatRouteDecision.model_validate(raw_decision)


def _build_direct_router_user_prompt(*, question: str, document: FilingDocument) -> str:
    intent_hint = infer_direct_router_document_evidence_intent(question)
    hint_block = ""
    if intent_hint is not None:
        hint_block = (
            "Local deterministic intent hint:\n"
            f"- document_evidence_intent: {intent_hint}\n\n"
        )

    return (
        "Current document:\n"
        f"- company_name: {document.company_name}\n"
        f"- ticker: {document.ticker or ''}\n"
        f"- market: {document.market.value}\n"
        f"- doc_type: {document.doc_type.value}\n"
        f"- fiscal_period: {document.fiscal_period or ''}\n\n"
        f"{hint_block}"
        f"User question:\n{question}"
    )


def _chat_completions_url(base_url: str | None) -> str:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    return f"{root}/chat/completions"


_DIRECT_ROUTER_SYSTEM_PROMPT = """You are the FilingDelta chat router.

Classify the user's question for a financial filing assistant.

Routes:
- document_only: answerable from the current filing alone.
- concept_only: mainly asks for a concept definition or external background, without filing-specific facts.
- mixed: needs both current filing facts and external concept/background knowledge.
- unsupported: outside filing analysis or financial concept explanation.

Rules:
- Disclosure questions about what the filing says, discloses, describes, manages, controls, changes, plans, or responds to are document_only.
- Do not route to mixed just because the question contains risk, strategy, AI, capital, asset quality, or policy words.
- "What is X?" is concept_only unless the user explicitly asks to combine it with current filing facts.
- For concept_only definitions, set needs_external_background=true and needs_risk_reasoning=false.
- Route to mixed only when the user asks for filing facts plus general concepts, usual implications, industry/regulatory background, or meaning beyond the filing.
- Requests for buy/sell/hold recommendations, target prices, portfolio advice, or investment advice are unsupported.
- Set needs_external_background only when external context is needed beyond the filing.
- Set needs_risk_reasoning only when the user asks about implications, usual effects, or what something means beyond the filing.
- Also classify document_evidence_intent when document evidence is needed:
  - metric_value: values, amounts, ratios, balances, percentages, or direct changes.
  - metric_attribution: causes, drivers, reasons, contributions, management explanation of a metric change, or whether an operating/financial metric looks abnormal.
  - business_narrative: business, risk, strategy, product, policy, or management-action narrative.
  - fallback: no document evidence or unclear document intent.
- Metric words do not automatically mean metric_value. Cause/driver/reason questions are metric_attribution.
- Inventory, channel inventory, channel efficiency, and turnover abnormality/explanation questions are metric_attribution, even when phrased as "how does the company describe".
- Inventory management strategy or supply-chain inventory management measure questions remain business_narrative unless they also ask abnormality, efficiency/turnover, causes, or change explanation.
- Plain narrative/product capability disclosure questions remain business_narrative.
Examples:
- 家电企业存货或渠道库存是否异常？公司如何描述渠道效率？ -> route=document_only, document_evidence_intent=metric_attribution.
- 腾讯如何描述 AI 广告能力？ -> route=document_only, document_evidence_intent=business_narrative.
"""


def _build_direct_router_system_prompt(
    response_mode: Literal["json_schema", "json_object"],
) -> str:
    if response_mode == "json_object":
        return (
            _DIRECT_ROUTER_SYSTEM_PROMPT
            + "\nReturn only a valid JSON object with exactly these keys: "
            "route, needs_external_background, needs_risk_reasoning, "
            "document_evidence_intent, rationale."
        )
    return _DIRECT_ROUTER_SYSTEM_PROMPT


def _build_response_format(
    response_mode: Literal["json_schema", "json_object"],
) -> dict[str, Any]:
    if response_mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "filingdelta_chat_route_decision",
            "strict": True,
            "schema": _CHAT_ROUTE_DECISION_JSON_SCHEMA,
        },
    }


_CHAT_ROUTE_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "route": {
            "type": "string",
            "enum": ["document_only", "concept_only", "mixed", "unsupported"],
        },
        "needs_external_background": {
            "type": "boolean",
        },
        "needs_risk_reasoning": {
            "type": "boolean",
        },
        "document_evidence_intent": {
            "type": "string",
            "enum": ["metric_value", "metric_attribution", "business_narrative", "fallback"],
        },
        "rationale": {
            "type": "string",
        },
    },
    "required": [
        "route",
        "needs_external_background",
        "needs_risk_reasoning",
        "document_evidence_intent",
        "rationale",
    ],
}
