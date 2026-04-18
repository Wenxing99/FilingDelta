from __future__ import annotations

from pathlib import Path
from typing import Protocol

import fitz
from llama_cloud import LlamaCloud
from llama_index.readers.file import HTMLTagReader, PyMuPDFReader, UnstructuredReader
from pypdf import PdfReader

from filingdelta.core.config import Settings
from filingdelta.schemas.filing import (
    FilingDocument,
    FilingSource,
    ParsedFiling,
    ParsedPage,
    ParserKind,
)


class FilingParser(Protocol):
    def parse(self, source: FilingSource) -> ParsedFiling: ...


class LocalFilingParser:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pdf_parser = PyMuPDFFilingParser()
        self._html_parser = HTMLTagFilingParser()
        self._unstructured_parser = UnstructuredFilingParser()
        self._fallback_parser = BasicFallbackFilingParser()

    def parse(self, source: FilingSource) -> ParsedFiling:
        suffix = source.source_path.suffix.lower()

        if suffix == ".pdf":
            try:
                return self._pdf_parser.parse(source)
            except (fitz.FileDataError, OSError, RuntimeError, ValueError):
                try:
                    return self._unstructured_parser.parse(source)
                except (ImportError, OSError, RuntimeError, ValueError):
                    return self._fallback_parser.parse(source)

        if suffix in {".htm", ".html"}:
            try:
                return self._html_parser.parse(source)
            except (OSError, UnicodeError, ValueError):
                try:
                    return self._unstructured_parser.parse(source)
                except (ImportError, OSError, RuntimeError, ValueError):
                    return self._fallback_parser.parse(source)

        try:
            return self._unstructured_parser.parse(source)
        except (ImportError, OSError, RuntimeError, ValueError):
            return self._fallback_parser.parse(source)


class LlamaParseFilingParser:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = LlamaCloud(**settings.llama_cloud_client_kwargs())

    def parse(self, source: FilingSource) -> ParsedFiling:
        with source.source_path.open("rb") as file_handle:
            uploaded_file = self._client.files.create(file=file_handle, purpose="parse")

        result = self._client.parsing.parse(
            file_id=uploaded_file.id,
            tier=self._settings.filingdelta_llama_parse_tier,
            version=self._settings.filingdelta_llama_parse_version,
            expand=["markdown", "metadata"],
            verbose=True,
        )

        pages = _build_pages_from_llama_cloud_response(result)
        return ParsedFiling(
            document=_build_document(source, ParserKind.LLAMA_PARSE, total_pages=len(pages)),
            pages=pages,
        )


class PyMuPDFFilingParser:
    def __init__(self) -> None:
        self._reader = PyMuPDFReader()

    def parse(self, source: FilingSource) -> ParsedFiling:
        documents = self._reader.load_data(source.source_path)
        pages = _build_pages_from_reader_documents(documents)
        if not pages:
            raise ValueError("PyMuPDFReader produced no readable pages.")

        return ParsedFiling(
            document=_build_document(source, ParserKind.PYMUPDF, total_pages=len(pages)),
            pages=pages,
        )


class HTMLTagFilingParser:
    _TAG_CANDIDATES = ("section", "article", "main", "body")

    def __init__(self) -> None:
        self._readers = {
            tag: HTMLTagReader(tag=tag, ignore_no_id=False) for tag in self._TAG_CANDIDATES
        }

    def parse(self, source: FilingSource) -> ParsedFiling:
        for tag in self._TAG_CANDIDATES:
            documents = self._readers[tag].load_data(source.source_path)
            pages = _build_pages_from_reader_documents(documents)
            if pages:
                return ParsedFiling(
                    document=_build_document(
                        source,
                        ParserKind.HTML_TAG,
                        total_pages=len(pages),
                    ),
                    pages=pages,
                )

        raise ValueError("HTMLTagReader produced no readable content for any supported tag.")


class UnstructuredFilingParser:
    def __init__(self) -> None:
        self._reader: UnstructuredReader | None = None

    def parse(self, source: FilingSource) -> ParsedFiling:
        if self._reader is None:
            self._reader = UnstructuredReader()
        documents = self._reader.load_data(file=source.source_path, split_documents=False)
        pages = _build_pages_from_reader_documents(documents)
        if not pages:
            raise ValueError("UnstructuredReader produced no readable content.")

        return ParsedFiling(
            document=_build_document(source, ParserKind.UNSTRUCTURED, total_pages=len(pages)),
            pages=pages,
        )


class BasicFallbackFilingParser:
    def parse(self, source: FilingSource) -> ParsedFiling:
        suffix = source.source_path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(source.source_path))
            pages = [
                ParsedPage(
                    page_number=index,
                    text=(page.extract_text() or "").strip(),
                    markdown=(page.extract_text() or "").strip(),
                )
                for index, page in enumerate(reader.pages, start=1)
            ]
        else:
            text = source.source_path.read_text(encoding="utf-8", errors="ignore").strip()
            pages = [ParsedPage(page_number=1, text=text, markdown=text)]

        pages = [page for page in pages if page.text]
        if not pages:
            raise ValueError("Fallback parser produced no readable text.")

        return ParsedFiling(
            document=_build_document(source, ParserKind.FALLBACK, total_pages=len(pages)),
            pages=pages,
        )


def get_filing_parser(settings: Settings) -> FilingParser:
    if settings.filingdelta_parse_provider == "llama_cloud":
        return LlamaParseFilingParser(settings)
    return LocalFilingParser(settings)


def _build_document(source: FilingSource, parser_kind: ParserKind, *, total_pages: int) -> FilingDocument:
    return FilingDocument(
        document_id=_make_document_id(source.source_path),
        company_name=source.company_name,
        ticker=source.ticker,
        market=source.market,
        doc_type=source.doc_type,
        fiscal_period=source.fiscal_period,
        language=source.language,
        source_path=source.source_path.resolve(),
        parser_kind=parser_kind,
        total_pages=total_pages,
    )


def _make_document_id(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def _build_pages_from_reader_documents(documents: list[object]) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    for index, document in enumerate(documents, start=1):
        metadata = _normalize_reader_metadata(getattr(document, "metadata", {}) or {})
        text = _coerce_reader_text(getattr(document, "text", "")).strip()
        if not text:
            continue

        page_number = _coerce_page_number(metadata.get("source"), default=index)
        pages.append(
            ParsedPage(
                page_number=page_number,
                text=text,
                markdown=text,
                metadata=metadata,
            )
        )
    return pages


def _coerce_reader_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _coerce_page_number(value: str | int | float | bool | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _normalize_reader_metadata(metadata: dict[str, object]) -> dict[str, str | int | float | bool | None]:
    normalized: dict[str, str | int | float | bool | None] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = value
        else:
            normalized[str(key)] = str(value)
    return normalized


def _build_pages_from_llama_cloud_response(result: object) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    markdown_result = getattr(result, "markdown", None)
    metadata_result = getattr(result, "metadata", None)

    metadata_by_page: dict[int, dict[str, str | int | float | bool | None]] = {}
    if metadata_result and getattr(metadata_result, "pages", None):
        for page_meta in metadata_result.pages:
            page_number = int(getattr(page_meta, "page_number"))
            metadata_by_page[page_number] = page_meta.model_dump(mode="python")

    if markdown_result and getattr(markdown_result, "pages", None):
        for page in markdown_result.pages:
            if not getattr(page, "success", False):
                continue

            page_number = int(getattr(page, "page_number"))
            page_markdown = str(getattr(page, "markdown", "")).strip()
            if not page_markdown:
                continue

            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=page_markdown,
                    markdown=page_markdown,
                    metadata=metadata_by_page.get(page_number, {}),
                )
            )

    if pages:
        return pages

    full_markdown = str(getattr(result, "markdown_full", "") or getattr(result, "text_full", "")).strip()
    if not full_markdown:
        raise ValueError("No page-level or full-document content returned by LlamaCloud parsing.")

    raw_pages = [chunk.strip() for chunk in full_markdown.split("\n---\n") if chunk.strip()]
    if not raw_pages:
        raw_pages = [full_markdown]

    for index, page_markdown in enumerate(raw_pages, start=1):
        pages.append(
            ParsedPage(
                page_number=index,
                text=page_markdown,
                markdown=page_markdown,
                metadata=metadata_by_page.get(index, {}),
            )
        )
    return pages
