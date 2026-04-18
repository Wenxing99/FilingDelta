from __future__ import annotations

from typing import Iterable
from uuid import uuid4

from llama_index.core.node_parser import SentenceSplitter

from filingdelta.schemas.filing import ChunkMetadata, FilingChunk, ParsedFiling


def build_chunks(
    parsed_filing: ParsedFiling,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[FilingChunk]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[FilingChunk] = []

    for page in parsed_filing.pages:
        for chunk_index, text in enumerate(_split_page_text(splitter, page.text)):
            chunks.append(
                FilingChunk(
                    chunk_id=f"{parsed_filing.document.document_id}-{page.page_number}-{chunk_index}-{uuid4().hex[:8]}",
                    text=text,
                    metadata=ChunkMetadata(
                        document_id=parsed_filing.document.document_id,
                        company_name=parsed_filing.document.company_name,
                        ticker=parsed_filing.document.ticker,
                        market=parsed_filing.document.market,
                        doc_type=parsed_filing.document.doc_type,
                        fiscal_period=parsed_filing.document.fiscal_period,
                        source_path=parsed_filing.document.source_path,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        parser_kind=parsed_filing.document.parser_kind,
                    ),
                )
            )

    return chunks


def _split_page_text(splitter: SentenceSplitter, text: str) -> Iterable[str]:
    clean_text = text.strip()
    if not clean_text:
        return []
    return splitter.split_text(clean_text)
