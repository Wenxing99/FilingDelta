from __future__ import annotations

import json
from typing import Any, Literal

import httpx

from filingdelta.core.config import Settings, get_settings
from filingdelta.schemas.chat import ChatRouteDecision
from filingdelta.schemas.filing import FilingDocument


class DirectJsonChatRouterAgent:
    """Small direct-OpenAI router used for router bake-off experiments."""

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
    return (
        "Current document:\n"
        f"- company_name: {document.company_name}\n"
        f"- ticker: {document.ticker or ''}\n"
        f"- market: {document.market.value}\n"
        f"- doc_type: {document.doc_type.value}\n"
        f"- fiscal_period: {document.fiscal_period or ''}\n\n"
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
"""


def _build_direct_router_system_prompt(
    response_mode: Literal["json_schema", "json_object"],
) -> str:
    if response_mode == "json_object":
        return (
            _DIRECT_ROUTER_SYSTEM_PROMPT
            + "\nReturn only a valid JSON object with exactly these keys: "
            "route, needs_external_background, needs_risk_reasoning, rationale."
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
        "rationale": {
            "type": "string",
        },
    },
    "required": [
        "route",
        "needs_external_background",
        "needs_risk_reasoning",
        "rationale",
    ],
}
