from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal

import httpx

from filingdelta.core.config import Settings, get_settings
from filingdelta.schemas.chat import ChatCitation, ExternalEvidenceResult


_ALLOWED_DOMAINS = [
    "wikipedia.org",
    "investopedia.com",
    "sec.gov",
    "investor.gov",
    "sse.com.cn",
    "szse.cn",
    "hkex.com.hk",
    "cninfo.com.cn",
]


class ExternalSearchError(RuntimeError):
    """Raised when external web search fails or returns unusable data."""


class ExternalSearchService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def search(
        self,
        *,
        question: str,
        search_kind: Literal["concept", "background", "concept_and_background"],
    ) -> ExternalEvidenceResult:
        payload = {
            "model": self._settings.filingdelta_llm_model,
            "tools": [
                {
                    "type": "web_search",
                    "filters": {
                        "allowed_domains": _ALLOWED_DOMAINS,
                    },
                }
            ],
            "tool_choice": "auto",
            "include": ["web_search_call.action.sources"],
            "input": _build_search_prompt(question=question, search_kind=search_kind),
        }

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                _responses_url(self._settings.openai_base_url),
                headers={
                    "Authorization": f"Bearer {self._settings.require_openai_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if response.status_code >= 400:
            raise ExternalSearchError(
                f"External web search failed: {response.status_code} {response.text}"
            )

        payload = response.json()
        answer_text, annotations, source_items = _extract_web_search_outputs(payload)
        citations = _build_external_citations(
            answer_text=answer_text,
            annotations=annotations,
            source_items=source_items,
        )
        if not citations:
            raise ExternalSearchError(
                "External web search returned no cited sources."
            )
        if not answer_text:
            raise ExternalSearchError(
                "External web search returned citations but no usable answer text."
            )

        return ExternalEvidenceResult(
            search_query=question,
            search_kind=search_kind,
            answer_text=answer_text,
            citations=citations,
            usage=payload.get("usage"),
        )


def _build_search_prompt(
    *,
    question: str,
    search_kind: Literal["concept", "background", "concept_and_background"],
) -> str:
    if search_kind == "concept":
        objective = "Explain the concept or term clearly and concisely for a filing-analysis user."
    elif search_kind == "background":
        objective = "Summarize the relevant external background or risk framework concisely."
    else:
        objective = (
            "Explain the concept and provide the most relevant external background needed to "
            "understand the filing question."
        )

    return (
        "You are gathering external evidence for a financial disclosure analysis assistant.\n"
        f"Objective: {objective}\n"
        "Rules:\n"
        "- Prefer authoritative or reference-style sources.\n"
        "- Be concise and factual.\n"
        "- Do not give investment advice.\n"
        "- Return a compact answer grounded in web sources.\n\n"
        f"Question: {question}"
    )


def _responses_url(base_url: str | None) -> str:
    normalized = (base_url or "https://api.openai.com/v1").rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/responses"
    return f"{normalized}/v1/responses"


def _extract_web_search_outputs(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    output_items = payload.get("output", [])
    message_text_parts: list[str] = []
    annotations: list[dict[str, Any]] = []
    source_items: list[dict[str, Any]] = []

    for item in output_items:
        item_type = item.get("type")
        if item_type == "message":
            for content in item.get("content", []):
                if content.get("type") not in {"output_text", "text"}:
                    continue
                text = str(content.get("text") or "").strip()
                if text:
                    message_text_parts.append(text)
                annotations.extend(content.get("annotations", []))
        elif item_type == "web_search_call":
            action = item.get("action") or {}
            action_sources = action.get("sources") or []
            for source in action_sources:
                if isinstance(source, dict):
                    source_items.append(source)

    return "\n\n".join(message_text_parts).strip(), annotations, source_items


def _build_external_citations(
    *,
    answer_text: str,
    annotations: list[dict[str, Any]],
    source_items: list[dict[str, Any]],
) -> list[ChatCitation]:
    deduped: "OrderedDict[str, ChatCitation]" = OrderedDict()

    for index, annotation in enumerate(annotations):
        if annotation.get("type") != "url_citation":
            continue
        url = str(annotation.get("url") or "").strip()
        if not url:
            continue
        title = str(annotation.get("title") or url).strip()
        snippet = _annotation_snippet(answer_text, annotation)
        citation = ChatCitation(
            citation_id=f"external-citation-{index + 1}",
            source_type="external",
            url=url,
            title=title,
            snippet=snippet,
        )
        deduped.setdefault(url, citation)

    for source in source_items:
        url = str(source.get("url") or "").strip()
        if not url or url in deduped:
            continue
        title = str(source.get("title") or url).strip()
        deduped[url] = ChatCitation(
            citation_id=f"external-citation-{len(deduped) + 1}",
            source_type="external",
            url=url,
            title=title,
            snippet=_truncate_text(answer_text),
        )

    return list(deduped.values())


def _annotation_snippet(answer_text: str, annotation: dict[str, Any]) -> str:
    start_index = annotation.get("start_index")
    end_index = annotation.get("end_index")
    if isinstance(start_index, int) and isinstance(end_index, int):
        snippet = answer_text[start_index:end_index].strip()
        if snippet:
            return snippet
    return _truncate_text(answer_text)


def _truncate_text(text: str, limit: int = 220) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3].rstrip()}..."
