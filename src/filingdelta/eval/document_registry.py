from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filingdelta.core.config import REPO_ROOT
from filingdelta.schemas.filing import FilingDocType, FilingSource, Market


class DocumentRegistryError(ValueError):
    """Raised when an eval document registry is malformed."""


@dataclass(frozen=True)
class RawEvalDocument:
    document_key: str
    source: FilingSource
    industry: str | None = None

    @property
    def source_path(self) -> Path:
        return self.source.source_path

    @property
    def exists(self) -> bool:
        return self.source_path.exists()

    def to_json(self) -> dict[str, object]:
        return {
            "document_key": self.document_key,
            "source_path": str(self.source_path),
            "company_name": self.source.company_name,
            "ticker": self.source.ticker,
            "market": self.source.market.value,
            "doc_type": self.source.doc_type.value,
            "fiscal_period": self.source.fiscal_period,
            "language": self.source.language,
            "industry": self.industry,
            "exists": self.exists,
        }


@dataclass(frozen=True)
class RawDocumentRegistry:
    documents: dict[str, RawEvalDocument]

    def require(self, document_key: str) -> RawEvalDocument:
        try:
            return self.documents[document_key]
        except KeyError as exc:
            raise DocumentRegistryError(f"Unknown document_key: {document_key}") from exc

    def missing_documents(self, document_keys: set[str] | None = None) -> list[RawEvalDocument]:
        selected_keys = document_keys or set(self.documents)
        return [
            document
            for key, document in self.documents.items()
            if key in selected_keys and not document.exists
        ]

    def to_json(self) -> dict[str, object]:
        return {
            key: document.to_json()
            for key, document in sorted(self.documents.items())
        }


def load_raw_document_registry(
    documents_payload: list[dict[str, Any]],
    *,
    base_dir: Path = REPO_ROOT,
) -> RawDocumentRegistry:
    if not isinstance(documents_payload, list):
        raise DocumentRegistryError("Manifest field 'documents' must be a list.")

    documents: dict[str, RawEvalDocument] = {}
    for index, payload in enumerate(documents_payload):
        if not isinstance(payload, dict):
            raise DocumentRegistryError(f"Document entry #{index + 1} must be an object.")

        document_key = str(_required(payload, "document_key", context=f"document #{index + 1}"))
        if document_key in documents:
            raise DocumentRegistryError(f"Duplicate document_key: {document_key}")

        source_path = _resolve_source_path(
            Path(str(_required(payload, "source_path", context=document_key))),
            base_dir=base_dir,
        )
        source = FilingSource(
            source_path=source_path,
            company_name=str(_required(payload, "company_name", context=document_key)),
            ticker=_optional_str(payload.get("ticker")),
            market=Market(str(payload.get("market") or Market.OTHER.value)),
            doc_type=FilingDocType(str(payload.get("doc_type") or FilingDocType.OTHER.value)),
            fiscal_period=_optional_str(payload.get("fiscal_period")),
            language=str(payload.get("language") or "zh"),
        )
        documents[document_key] = RawEvalDocument(
            document_key=document_key,
            source=source,
            industry=_optional_str(payload.get("industry")),
        )

    return RawDocumentRegistry(documents=documents)


def _resolve_source_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def _required(payload: dict[str, Any], field_name: str, *, context: str) -> object:
    value = payload.get(field_name)
    if value in (None, ""):
        raise DocumentRegistryError(f"Missing required field '{field_name}' in {context}.")
    return value


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
