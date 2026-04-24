from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from filingdelta.ingestion.section_evidence import build_section_evidence
from filingdelta.ingestion.table_row_evidence import build_table_row_evidence
from filingdelta.schemas.filing import EvidenceKind, EvidenceMetadata, EvidenceUnit, FilingChunk, ParsedFiling


def build_evidence_units(
    *,
    parsed_filing: ParsedFiling,
    chunks: list[FilingChunk],
) -> list[EvidenceUnit]:
    page_text_units = [_chunk_to_page_text_evidence(chunk) for chunk in chunks]
    section_units = build_section_evidence(parsed_filing)
    table_row_units = build_table_row_evidence(parsed_filing)
    return page_text_units + section_units + table_row_units


def _chunk_to_page_text_evidence(chunk: FilingChunk) -> EvidenceUnit:
    metadata = chunk.metadata
    return EvidenceUnit(
        evidence_id=_page_text_evidence_id(chunk),
        text=chunk.text,
        metadata=EvidenceMetadata(
            document_id=metadata.document_id,
            source_path=Path(metadata.source_path),
            page_number=metadata.page_number,
            page_end=metadata.page_number,
            parser_kind=metadata.parser_kind,
            chunk_kind=EvidenceKind.PAGE_TEXT,
        ),
    )


def _page_text_evidence_id(chunk: FilingChunk) -> str:
    metadata = chunk.metadata
    stable_key = f"{metadata.document_id}:{metadata.page_number}:{metadata.chunk_index}"
    return str(uuid5(NAMESPACE_URL, stable_key))
