from __future__ import annotations

from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters, VectorStoreQuery
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from filingdelta.core.config import Settings, get_settings
from filingdelta.retrieval.indexer import CHUNK_KIND_FILTER_KEY, COLLECTION_NAME, DOCUMENT_FILTER_KEY
from filingdelta.schemas.chat import RetrievedChunk


class DocumentChunkRetriever:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or QdrantClient(path=str(self._settings.qdrant_path))
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=COLLECTION_NAME,
        )
    def retrieve(
        self,
        *,
        document_id: str,
        question: str,
        top_k: int = 6,
        chunk_kind: str | None = None,
        callback_manager: CallbackManager | None = None,
    ) -> list[RetrievedChunk]:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._build_embed_model(callback_manager=callback_manager),
        )
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=_build_metadata_filters(document_id=document_id, chunk_kind=chunk_kind),
        )

        try:
            return [
                _node_to_retrieved_chunk(item.node, document_id=document_id, score=item.score)
                for item in retriever.retrieve(question)
            ]
        except ValueError as error:
            if "Dense vector text-dense is not found in the collection" not in str(error):
                raise
            return self._retrieve_with_vector_store_query(
                document_id=document_id,
                    question=question,
                    top_k=top_k,
                    chunk_kind=chunk_kind,
                    callback_manager=callback_manager,
            )

    def _retrieve_with_vector_store_query(
        self,
        *,
        document_id: str,
        question: str,
        top_k: int,
        chunk_kind: str | None = None,
        callback_manager: CallbackManager | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self._build_embed_model(
            callback_manager=callback_manager
        ).get_query_embedding(question)
        vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=COLLECTION_NAME,
            dense_vector_name="",
        )
        query_result = vector_store.query(
            VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=top_k,
                filters=_build_metadata_filters(document_id=document_id, chunk_kind=chunk_kind),
            )
        )

        retrieved = []
        for index, node in enumerate(query_result.nodes or []):
            score = None
            if query_result.similarities and index < len(query_result.similarities):
                score = query_result.similarities[index]
            retrieved.append(_node_to_retrieved_chunk(node, document_id=document_id, score=score))
        return retrieved

    def _build_embed_model(
        self, *, callback_manager: CallbackManager | None = None
    ) -> OpenAIEmbedding:
        return OpenAIEmbedding(
            model=self._settings.filingdelta_embed_model,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
            callback_manager=callback_manager,
        )


def _node_to_retrieved_chunk(
    node: object,
    *,
    document_id: str,
    score: float | None,
) -> RetrievedChunk:
    metadata = node.metadata
    return RetrievedChunk(
        chunk_id=node.node_id,
        document_id=str(metadata.get(DOCUMENT_FILTER_KEY) or document_id),
        page_number=_coerce_int(metadata.get("page_number")),
        source_path=Path(str(metadata.get("source_path") or "")),
        text=node.get_content(metadata_mode=MetadataMode.NONE),
        score=float(score) if score is not None else None,
        chunk_kind=_coerce_str(metadata.get(CHUNK_KIND_FILTER_KEY)),
        section_title=_coerce_str(metadata.get("section_title")),
        section_type=_coerce_str(metadata.get("section_type")),
        table_id=_coerce_str(metadata.get("table_id")),
        row_label=_coerce_str(metadata.get("row_label")),
        metric_tags=_coerce_str_list(metadata.get("metric_tags")),
        period_hint=_coerce_str(metadata.get("period_hint")),
        retrieval_source="semantic",
    )


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [text for item in value if (text := _coerce_str(item))]
    text = _coerce_str(value)
    return [text] if text else []


def _build_metadata_filters(*, document_id: str, chunk_kind: str | None) -> MetadataFilters:
    filters = [ExactMatchFilter(key=DOCUMENT_FILTER_KEY, value=document_id)]
    if chunk_kind:
        filters.append(ExactMatchFilter(key=CHUNK_KIND_FILTER_KEY, value=chunk_kind))
    return MetadataFilters(filters=filters)
