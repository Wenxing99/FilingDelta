from __future__ import annotations

from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from filingdelta.core.config import Settings, get_settings
from filingdelta.retrieval.indexer import COLLECTION_NAME, DOCUMENT_FILTER_KEY
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
        self._embed_model = OpenAIEmbedding(
            model=self._settings.filingdelta_embed_model,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
        )

    def retrieve(
        self,
        *,
        document_id: str,
        question: str,
        top_k: int = 6,
    ) -> list[RetrievedChunk]:
        index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key=DOCUMENT_FILTER_KEY, value=document_id)]
            ),
        )

        retrieved = []
        for item in retriever.retrieve(question):
            node = item.node
            metadata = node.metadata
            retrieved.append(
                RetrievedChunk(
                    chunk_id=node.node_id,
                    document_id=str(metadata.get(DOCUMENT_FILTER_KEY) or document_id),
                    page_number=_coerce_int(metadata.get("page_number")),
                    source_path=Path(str(metadata.get("source_path") or "")),
                    text=node.get_content(metadata_mode=MetadataMode.NONE),
                    score=float(item.score) if item.score is not None else None,
                    retrieval_source="semantic",
                )
            )
        return retrieved


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
