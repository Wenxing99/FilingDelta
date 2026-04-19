from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from filingdelta.core.config import Settings, get_settings
from filingdelta.schemas.filing import FilingChunk
from filingdelta.storage.paths import ensure_data_dirs


COLLECTION_NAME = "filingdelta_demo_chunks"
DOCUMENT_FILTER_KEY = "filing_document_id"


class DocumentChunkIndexer:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        ensure_data_dirs()
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

    def index_document(self, *, document_id: str, chunks: list[FilingChunk]) -> None:
        if not chunks:
            return

        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self._embed_model,
            show_progress=False,
        )
        index.insert_nodes([chunk_to_node(chunk, document_id=document_id) for chunk in chunks])


def chunk_node_id(chunk: FilingChunk, *, document_id: str | None = None) -> str:
    metadata = chunk.metadata
    effective_document_id = document_id or metadata.document_id
    stable_key = f"{effective_document_id}:{metadata.page_number}:{metadata.chunk_index}"
    return str(uuid5(NAMESPACE_URL, stable_key))


def chunk_to_node(chunk: FilingChunk, *, document_id: str) -> TextNode:
    metadata = chunk.metadata
    node_metadata = {
        DOCUMENT_FILTER_KEY: document_id,
        "company_name": metadata.company_name,
        "ticker": metadata.ticker or "",
        "market": metadata.market.value,
        "doc_type": metadata.doc_type.value,
        "fiscal_period": metadata.fiscal_period or "",
        "source_path": str(Path(metadata.source_path)),
        "page_number": metadata.page_number,
        "chunk_index": metadata.chunk_index,
        "parser_kind": metadata.parser_kind.value,
    }
    return TextNode(
        id_=chunk_node_id(chunk, document_id=document_id),
        text=chunk.text,
        metadata=node_metadata,
    )
