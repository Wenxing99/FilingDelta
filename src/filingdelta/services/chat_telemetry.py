from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator, Literal

from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from filingdelta.schemas.chat import ChatRetrievalTelemetry, ChatStepTelemetry, ChatTelemetry, ChatUsageTelemetry


ChatRouteType = Literal["document_only", "concept_only", "mixed", "unsupported"]
StepName = Literal[
    "index_build_ms",
    "contextualizer_ms",
    "router_ms",
    "planner_ms",
    "document_retrieval_ms",
    "external_search_ms",
    "answerer_ms",
    "memory_summarizer_ms",
]


class ChatTelemetryRecorder:
    """Lightweight, fail-open telemetry recorder for chat requests."""

    def __init__(self) -> None:
        self._started_at = perf_counter()
        self._steps = ChatStepTelemetry()
        self._usage = ChatUsageTelemetry()
        self._retrieval = ChatRetrievalTelemetry()
        self._token_handler = TokenCountingHandler()
        self._callback_manager = CallbackManager([self._token_handler])

    @property
    def callback_manager(self) -> CallbackManager:
        return self._callback_manager

    @property
    def token_handler(self) -> TokenCountingHandler:
        return self._token_handler

    @contextmanager
    def track(self, step_name: StepName) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self._set_step_duration(step_name, (perf_counter() - started) * 1000)

    def set_retrieval(
        self,
        *,
        document_top_k: int | None = None,
        document_retrieved_chunks: int | None = None,
        external_sources_count: int | None = None,
        used_document_citations_count: int | None = None,
        used_external_citations_count: int | None = None,
    ) -> None:
        if document_top_k is not None:
            self._retrieval.document_top_k = document_top_k
        if document_retrieved_chunks is not None:
            self._retrieval.document_retrieved_chunks = document_retrieved_chunks
        if external_sources_count is not None:
            self._retrieval.external_sources_count = external_sources_count
        if used_document_citations_count is not None:
            self._retrieval.used_document_citations_count = used_document_citations_count
        if used_external_citations_count is not None:
            self._retrieval.used_external_citations_count = used_external_citations_count

    def add_web_search_usage(self, usage: dict[str, object] | None) -> None:
        if not isinstance(usage, dict):
            return

        input_tokens = _as_int(usage.get("input_tokens"))
        output_tokens = _as_int(usage.get("output_tokens"))
        total_tokens = _as_int(usage.get("total_tokens"))
        output_details = usage.get("output_tokens_details")
        reasoning_tokens = 0
        if isinstance(output_details, dict):
            reasoning_tokens = _as_int(output_details.get("reasoning_tokens"))

        self._usage.web_search_input_tokens += input_tokens
        self._usage.web_search_output_tokens += output_tokens
        self._usage.web_search_total_tokens += total_tokens
        self._usage.reasoning_tokens += reasoning_tokens

    def build(self, *, route_type: ChatRouteType, succeeded: bool) -> ChatTelemetry:
        prompt_tokens = _as_int(self._token_handler.prompt_llm_token_count)
        completion_tokens = _as_int(self._token_handler.completion_llm_token_count)
        llm_total_tokens = _as_int(self._token_handler.total_llm_token_count)
        embedding_tokens = _as_int(self._token_handler.total_embedding_token_count)

        self._usage.llm_prompt_tokens = prompt_tokens
        self._usage.llm_completion_tokens = completion_tokens
        self._usage.llm_total_tokens = llm_total_tokens
        self._usage.embedding_tokens = embedding_tokens
        self._usage.total_tokens = (
            llm_total_tokens + embedding_tokens + self._usage.web_search_total_tokens
        )

        return ChatTelemetry(
            route_type=route_type,
            total_latency_ms=(perf_counter() - self._started_at) * 1000,
            succeeded=succeeded,
            steps=self._steps,
            usage=self._usage,
            retrieval=self._retrieval,
        )

    def _set_step_duration(self, step_name: StepName, duration_ms: float) -> None:
        setattr(self._steps, step_name, round(duration_ms, 2))


def _as_int(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
