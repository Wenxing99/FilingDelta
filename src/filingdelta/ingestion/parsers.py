from __future__ import annotations

from pathlib import Path
from typing import Protocol

import fitz
from llama_cloud import LlamaCloud

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
    def parse(self, source: FilingSource) -> ParsedFiling:
        doc = fitz.open(source.source_path)
        try:
            pages = [
                ParsedPage(page_number=page.number + 1, text=page.get_text("text").strip())
                for page in doc
            ]
        finally:
            doc.close()

        return ParsedFiling(
            document=_build_document(source, ParserKind.PYMUPDF, total_pages=len(pages)),
            pages=pages,
        )


def get_filing_parser(settings: Settings) -> FilingParser:
    if settings.filingdelta_use_llama_parse:
        return LlamaParseFilingParser(settings)
    return PyMuPDFFilingParser()


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
