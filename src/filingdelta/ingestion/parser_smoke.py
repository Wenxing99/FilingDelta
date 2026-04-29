from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Iterator, Literal

import fitz
from pydantic import BaseModel, Field

from filingdelta.core.config import REPO_ROOT
from filingdelta.ingestion.raw_registry import (
    RawDocumentRegistry,
    RawRegistryEntry,
    SUPPORTED_RAW_SUFFIXES,
)
from filingdelta.schemas.filing import ParserKind


ParserSmokeStatus = Literal["passed", "failed", "skipped"]

WARNING_EMPTY_TEXT = "empty_extracted_text"


class ParserSmokeDocumentResult(BaseModel):
    document_key: str
    local_path: str
    suffix: str
    status: ParserSmokeStatus
    parser_kind_candidate: str
    page_count_estimate: int | None = None
    sample_text_chars: int = 0
    sample_pages_checked: list[int] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ParserSmokeSummary(BaseModel):
    total_documents: int = 0
    supported_documents: int = 0
    unsupported_documents: int = 0
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    parser_kind_candidate_counts: dict[str, int] = Field(default_factory=dict)
    error_counts: dict[str, int] = Field(default_factory=dict)
    warning_count: int = 0
    warnings_by_type: dict[str, int] = Field(default_factory=dict)


class ParserSmokeReport(BaseModel):
    schema_version: Literal["parser_smoke_report.v1"] = "parser_smoke_report.v1"
    raw_dir: str
    sample_pages_per_pdf: int
    min_text_chars: int
    documents: list[ParserSmokeDocumentResult] = Field(default_factory=list)
    summary: ParserSmokeSummary


@dataclass(frozen=True)
class ParserSmokeSample:
    page_count_estimate: int | None
    sample_text: str
    sample_pages_checked: list[int]


def load_raw_document_registry_json(
    path: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
) -> RawDocumentRegistry:
    registry_path = _resolve_path(Path(path), repo_root=repo_root)
    return RawDocumentRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))


def run_parser_smoke_check(
    registry: RawDocumentRegistry,
    *,
    repo_root: Path = REPO_ROOT,
    sample_pages_per_pdf: int = 3,
    min_text_chars: int = 1,
) -> ParserSmokeReport:
    if sample_pages_per_pdf < 1:
        raise ValueError("sample_pages_per_pdf must be at least 1.")
    if min_text_chars < 1:
        raise ValueError("min_text_chars must be at least 1.")

    repo_root = repo_root.resolve()
    results = [
        _smoke_check_entry(
            entry,
            repo_root=repo_root,
            sample_pages_per_pdf=sample_pages_per_pdf,
            min_text_chars=min_text_chars,
        )
        for entry in registry.documents
    ]
    return ParserSmokeReport(
        raw_dir=registry.raw_dir,
        sample_pages_per_pdf=sample_pages_per_pdf,
        min_text_chars=min_text_chars,
        documents=results,
        summary=summarize_parser_smoke_results(results),
    )


def summarize_parser_smoke_results(
    results: list[ParserSmokeDocumentResult],
) -> ParserSmokeSummary:
    status_counts = Counter(result.status for result in results)
    parser_counts = Counter(result.parser_kind_candidate for result in results)
    error_counts = Counter(result.error_type for result in results if result.error_type)
    warning_counts = Counter(warning for result in results for warning in result.warnings)

    return ParserSmokeSummary(
        total_documents=len(results),
        supported_documents=sum(1 for result in results if result.suffix in SUPPORTED_RAW_SUFFIXES),
        unsupported_documents=sum(1 for result in results if result.suffix not in SUPPORTED_RAW_SUFFIXES),
        passed_count=status_counts.get("passed", 0),
        failed_count=status_counts.get("failed", 0),
        skipped_count=status_counts.get("skipped", 0),
        status_counts=dict(sorted(status_counts.items())),
        parser_kind_candidate_counts=dict(sorted(parser_counts.items())),
        error_counts=dict(sorted(error_counts.items())),
        warning_count=sum(warning_counts.values()),
        warnings_by_type=dict(sorted(warning_counts.items())),
    )


def _smoke_check_entry(
    entry: RawRegistryEntry,
    *,
    repo_root: Path,
    sample_pages_per_pdf: int,
    min_text_chars: int,
) -> ParserSmokeDocumentResult:
    parser_kind_candidate = _candidate_parser_kind(entry.suffix)
    warnings = list(entry.warnings)

    if entry.suffix not in SUPPORTED_RAW_SUFFIXES:
        return ParserSmokeDocumentResult(
            document_key=entry.document_key,
            local_path=entry.local_path,
            suffix=entry.suffix,
            status="skipped",
            parser_kind_candidate=parser_kind_candidate,
            warnings=warnings,
        )

    source_path = _resolve_path(Path(entry.local_path), repo_root=repo_root)
    if not source_path.is_file():
        return ParserSmokeDocumentResult(
            document_key=entry.document_key,
            local_path=entry.local_path,
            suffix=entry.suffix,
            status="failed",
            parser_kind_candidate=parser_kind_candidate,
            error_type="FileNotFoundError",
            error_message=f"File not found: {entry.local_path}",
            warnings=warnings,
        )

    try:
        sample = _extract_sample_for_suffix(
            source_path,
            suffix=entry.suffix,
            sample_pages_per_pdf=sample_pages_per_pdf,
        )
    except Exception as error:  # noqa: BLE001 - report per-document parser failures.
        return ParserSmokeDocumentResult(
            document_key=entry.document_key,
            local_path=entry.local_path,
            suffix=entry.suffix,
            status="failed",
            parser_kind_candidate=parser_kind_candidate,
            error_type=type(error).__name__,
            error_message=str(error) or "Parser smoke check failed.",
            warnings=warnings,
        )

    sample_text_chars = len(_normalize_sample_text(sample.sample_text))
    if sample_text_chars < min_text_chars:
        return ParserSmokeDocumentResult(
            document_key=entry.document_key,
            local_path=entry.local_path,
            suffix=entry.suffix,
            status="failed",
            parser_kind_candidate=parser_kind_candidate,
            page_count_estimate=sample.page_count_estimate,
            sample_text_chars=sample_text_chars,
            sample_pages_checked=sample.sample_pages_checked,
            error_type="EmptyTextError",
            error_message=(
                f"Extracted {sample_text_chars} text chars; "
                f"required at least {min_text_chars}."
            ),
            warnings=[*warnings, WARNING_EMPTY_TEXT],
        )

    return ParserSmokeDocumentResult(
        document_key=entry.document_key,
        local_path=entry.local_path,
        suffix=entry.suffix,
        status="passed",
        parser_kind_candidate=parser_kind_candidate,
        page_count_estimate=sample.page_count_estimate,
        sample_text_chars=sample_text_chars,
        sample_pages_checked=sample.sample_pages_checked,
        warnings=warnings,
    )


def _extract_sample_for_suffix(
    source_path: Path,
    *,
    suffix: str,
    sample_pages_per_pdf: int,
) -> ParserSmokeSample:
    if suffix == ".pdf":
        return _extract_pdf_sample(source_path, sample_pages_per_pdf=sample_pages_per_pdf)
    if suffix in {".htm", ".html"}:
        return _extract_html_sample(source_path)
    raise ValueError(f"Unsupported suffix for parser smoke check: {suffix}")


def _extract_pdf_sample(
    source_path: Path,
    *,
    sample_pages_per_pdf: int,
) -> ParserSmokeSample:
    with _quiet_mupdf_messages():
        document = fitz.open(source_path)
        with document:
            page_count = len(document)
            pages_to_check = list(range(1, min(sample_pages_per_pdf, page_count) + 1))
            text_parts = []
            for page_number in pages_to_check:
                page = document.load_page(page_number - 1)
                text_parts.append(page.get_text("text") or "")

    return ParserSmokeSample(
        page_count_estimate=page_count,
        sample_text="\n".join(text_parts),
        sample_pages_checked=pages_to_check,
    )


def _extract_html_sample(source_path: Path) -> ParserSmokeSample:
    html_text = _decode_bytes(source_path.read_bytes())
    return ParserSmokeSample(
        page_count_estimate=1,
        sample_text=_extract_visible_html_text(html_text),
        sample_pages_checked=[1],
    )


def _candidate_parser_kind(suffix: str) -> str:
    if suffix == ".pdf":
        return ParserKind.PYMUPDF.value
    if suffix in {".htm", ".html"}:
        return ParserKind.HTML_TAG.value
    return "unsupported"


def _decode_bytes(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="ignore")


def _extract_visible_html_text(html_text: str) -> str:
    parser = _VisibleHTMLTextExtractor()
    parser.feed(html_text)
    parser.close()
    return _normalize_sample_text(parser.text)


def _normalize_sample_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


@contextmanager
def _quiet_mupdf_messages() -> Iterator[None]:
    display_errors = bool(fitz.TOOLS.mupdf_display_errors())
    display_warnings = bool(fitz.TOOLS.mupdf_display_warnings())
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    try:
        yield
    finally:
        fitz.TOOLS.mupdf_display_errors(display_errors)
        fitz.TOOLS.mupdf_display_warnings(display_warnings)


class _VisibleHTMLTextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript"}
    _BLOCK_TAGS = {
        "article",
        "aside",
        "body",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    @property
    def text(self) -> str:
        return " ".join(part for part in self._parts if part.strip())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized_tag = tag.casefold()
        if normalized_tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and normalized_tag in self._BLOCK_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and normalized_tag in self._BLOCK_TAGS:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)
