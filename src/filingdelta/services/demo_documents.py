from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from filingdelta.core.config import REPO_ROOT
from filingdelta.schemas.demo import DemoDocument
from filingdelta.schemas.filing import FilingSource


MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "small_doc_benchmark.json"


def _build_document_id(index: int, source: FilingSource) -> str:
    digest = hashlib.md5(str(source.source_path).encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"doc-{index + 1}-{digest[:8]}"


def _source_kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".htm", ".html"}:
        return "html"
    return "other"


def _load_manifest_sources() -> list[FilingSource]:
    if not MANIFEST_PATH.exists():
        return []

    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = payload.get("entries") or []
    sources: list[FilingSource] = []
    for entry in entries:
        source = FilingSource.model_validate(entry)
        if not source.source_path.is_absolute():
            source.source_path = (REPO_ROOT / source.source_path).resolve()
        sources.append(source)
    return sources


@lru_cache(maxsize=1)
def get_demo_document_sources() -> dict[str, FilingSource]:
    sources = _load_manifest_sources()
    return {
        _build_document_id(index, source): source
        for index, source in enumerate(sources)
        if source.source_path.exists()
    }


def list_demo_documents() -> list[DemoDocument]:
    documents: list[DemoDocument] = []
    for document_id, source in get_demo_document_sources().items():
        documents.append(
            DemoDocument(
                document_id=document_id,
                label=source.source_path.name,
                company_name=source.company_name,
                ticker=source.ticker,
                market=source.market.value,
                doc_type=source.doc_type.value,
                fiscal_period=source.fiscal_period,
                language=source.language,
                source_kind=_source_kind_for_path(source.source_path),
                source_url=f"/api/demo/documents/{document_id}/source",
            )
        )
    return documents


def get_demo_document_source(document_id: str) -> FilingSource:
    source = get_demo_document_sources().get(document_id)
    if source is None:
        raise KeyError(f"Unknown demo document: {document_id}")
    return source
