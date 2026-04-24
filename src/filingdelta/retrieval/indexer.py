from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from filingdelta.core.config import Settings, get_settings
from filingdelta.schemas.filing import EvidenceKind, EvidenceUnit, FilingChunk
from filingdelta.storage.paths import ensure_data_dirs


COLLECTION_NAME = "filingdelta_demo_chunks"
DOCUMENT_FILTER_KEY = "filing_document_id"
CHUNK_KIND_FILTER_KEY = "chunk_kind"


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
    def index_document(
        self,
        *,
        document_id: str,
        chunks: list[FilingChunk],
        evidence_units: list[EvidenceUnit] | None = None,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        nodes = _build_nodes(
            document_id=document_id,
            chunks=chunks,
            evidence_units=evidence_units,
        )
        if not nodes:
            return

        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self._build_embed_model(callback_manager=callback_manager),
            show_progress=False,
        )
        index.insert_nodes(nodes)

    def _build_embed_model(self, *, callback_manager: CallbackManager | None = None) -> OpenAIEmbedding:
        return OpenAIEmbedding(
            model=self._settings.filingdelta_embed_model,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
            callback_manager=callback_manager,
        )


def chunk_node_id(chunk: FilingChunk, *, document_id: str | None = None) -> str:
    metadata = chunk.metadata
    effective_document_id = document_id or metadata.document_id
    stable_key = f"{effective_document_id}:{metadata.page_number}:{metadata.chunk_index}"
    return str(uuid5(NAMESPACE_URL, stable_key))


def chunk_to_node(chunk: FilingChunk, *, document_id: str) -> TextNode:
    metadata = chunk.metadata
    node_metadata = {
        DOCUMENT_FILTER_KEY: document_id,
        CHUNK_KIND_FILTER_KEY: EvidenceKind.PAGE_TEXT.value,
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


def evidence_to_node(evidence: EvidenceUnit, *, document_id: str) -> TextNode:
    metadata = evidence.metadata
    node_metadata = {
        DOCUMENT_FILTER_KEY: document_id,
        CHUNK_KIND_FILTER_KEY: metadata.chunk_kind.value,
        "source_path": str(Path(metadata.source_path)),
        "page_number": metadata.page_number,
        "page_end": metadata.page_end or "",
        "parser_kind": metadata.parser_kind.value,
        "section_title": metadata.section_title or "",
        "section_type": metadata.section_type or "",
        "table_id": metadata.table_id or "",
        "row_label": metadata.row_label or "",
        "metric_tags": metadata.metric_tags,
        "period_hint": metadata.period_hint or "",
    }
    return TextNode(
        id_=evidence.evidence_id,
        text=evidence.text,
        metadata=node_metadata,
    )


def _build_nodes(
    *,
    document_id: str,
    chunks: list[FilingChunk],
    evidence_units: list[EvidenceUnit] | None,
) -> list[TextNode]:
    if evidence_units:
        return [evidence_to_node(evidence, document_id=document_id) for evidence in evidence_units]
    return [chunk_to_node(chunk, document_id=document_id) for chunk in chunks]
