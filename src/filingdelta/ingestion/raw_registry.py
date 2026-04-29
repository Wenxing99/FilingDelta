from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

from filingdelta.core.config import REPO_ROOT
from filingdelta.schemas.filing import FilingDocType, Market


SUPPORTED_RAW_SUFFIXES = frozenset({".pdf", ".htm", ".html"})
DEFAULT_RAW_DIR = Path("data/raw")
SMALL_FILE_THRESHOLD_BYTES = 1024

WARNING_UNSUPPORTED_SUFFIX = "unsupported_suffix"
WARNING_MISSING_COMPANY = "missing_company"
WARNING_MISSING_FISCAL_YEAR = "missing_fiscal_year"
WARNING_MISSING_DOC_TYPE = "missing_doc_type"
WARNING_SUSPICIOUSLY_SMALL_FILE = "suspiciously_small_file"
WARNING_DUPLICATE_CHECKSUM = "duplicate_checksum"

WARNING_ORDER = (
    WARNING_UNSUPPORTED_SUFFIX,
    WARNING_MISSING_COMPANY,
    WARNING_MISSING_FISCAL_YEAR,
    WARNING_MISSING_DOC_TYPE,
    WARNING_SUSPICIOUSLY_SMALL_FILE,
    WARNING_DUPLICATE_CHECKSUM,
)

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_NUMERIC_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_CHINESE_YEAR_RE = re.compile(r"([二〇零一二三四五六七八九]{4})年?")
_METADATA_SEPARATOR_RE = re.compile(r"[\s_\-–—]+")
_TOKEN_SPLIT_RE = re.compile(r"[\s_\-–—()\[\]{}（）【】,，.]+")

_METADATA_PATTERNS = (
    r"\bfy\s*20\d{2}\b",
    r"20\d{2}(?:年(?!度|报|報))?",
    r"[二〇零一二三四五六七八九]{4}(?:年(?!度|报|報))?",
    r"form\s*20\s*f",
    r"\b20\s*f\b",
    r"form\s*6\s*k",
    r"\b6\s*k\b",
    r"annual\s+report",
    r"interim\s+report",
    r"half\s+year\s+report",
    r"quarterly\s+report",
    r"q[1-4]\s+report",
    r"unaudited\s+financial\s+results",
    r"earnings\s+release",
    r"results\s+announcement",
    r"profit\s+warning",
    r"response\s+letter",
    r"a\s+share",
    r"h\s+share",
    r"\badr\b",
    r"\bsec\b",
    r"\bnyse\b",
    r"\bnasdaq\b",
    r"\bhkex\b",
    r"\bsse\b",
    r"\bszse\b",
    r"年度报告",
    r"年度報告",
    r"年报",
    r"年報",
    r"中期报告",
    r"中期報告",
    r"半年报",
    r"半年報",
    r"半年度报告",
    r"半年度報告",
    r"季度报告",
    r"季度報告",
    r"一季报",
    r"一季報",
    r"三季报",
    r"三季報",
    r"业绩公告",
    r"業績公告",
    r"业绩报告",
    r"業績報告",
    r"业绩预告",
    r"業績預告",
    r"问询函回复",
    r"問詢函回覆",
    r"a股",
    r"h股",
    r"港股",
    r"上交所",
    r"深交所",
    r"联交所",
    r"聯交所",
)

_GENERIC_COMPANY_TOKENS = {
    "document",
    "doc",
    "filing",
    "raw",
    "report",
    "unknown",
    "untitled",
    "sample",
    "test",
    "misc",
    "zh",
    "cn",
    "en",
    "hk",
    "us",
    "sec",
    "未命名",
    "未知",
    "样本",
    "樣本",
}


class RawRegistryEntry(BaseModel):
    document_key: str
    local_path: str
    filename: str
    suffix: str
    file_size: int
    checksum_sha256: str
    company_id: str | None = None
    ticker: str | None = None
    industry: str | None = None
    inferred_company_name: str | None = None
    inferred_fiscal_year: int | None = None
    fiscal_period: str | None = None
    inferred_doc_type: str | None = None
    inferred_market: str | None = None
    language: Literal["zh", "en"]
    source_url: str | None = None
    notes: str | None = None
    status: Literal[
        "registered",
        "validated",
        "indexed",
        "failed_validation",
        "failed_ingestion",
    ] = "registered"
    warnings: list[str] = Field(default_factory=list)


class RawRegistrySummary(BaseModel):
    total_files: int = 0
    supported_files: int = 0
    unsupported_files: int = 0
    warning_count: int = 0
    warnings_by_type: dict[str, int] = Field(default_factory=dict)
    duplicate_checksum_groups: int = 0
    suffix_counts: dict[str, int] = Field(default_factory=dict)


class RawDocumentRegistry(BaseModel):
    schema_version: Literal["raw_document_registry.v1"] = "raw_document_registry.v1"
    raw_dir: str
    documents: list[RawRegistryEntry] = Field(default_factory=list)
    summary: RawRegistrySummary


@dataclass(frozen=True)
class _FilenameMetadata:
    company_name: str | None
    fiscal_year: int | None
    doc_type: str | None
    market: str | None
    language: Literal["zh", "en"]


def scan_raw_document_registry(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    small_file_threshold_bytes: int = SMALL_FILE_THRESHOLD_BYTES,
) -> RawDocumentRegistry:
    repo_root = repo_root.resolve()
    raw_dir_path = _resolve_path(Path(raw_dir), repo_root=repo_root)
    entries = [
        _build_entry(
            path=path,
            repo_root=repo_root,
            small_file_threshold_bytes=small_file_threshold_bytes,
        )
        for path in _iter_files(raw_dir_path, repo_root=repo_root)
    ]
    entries = _with_duplicate_checksum_warnings(entries)
    return RawDocumentRegistry(
        raw_dir=_stable_path(raw_dir_path, repo_root=repo_root),
        documents=entries,
        summary=_summarize(entries),
    )


def _iter_files(raw_dir: Path, *, repo_root: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        (path for path in raw_dir.rglob("*") if path.is_file()),
        key=lambda path: _stable_path(path, repo_root=repo_root).casefold(),
    )


def _build_entry(
    *,
    path: Path,
    repo_root: Path,
    small_file_threshold_bytes: int,
) -> RawRegistryEntry:
    local_path = _stable_path(path, repo_root=repo_root)
    suffix = path.suffix.lower()
    file_size = path.stat().st_size
    checksum = _sha256_file(path)
    metadata = _infer_filename_metadata(path)

    warnings: list[str] = []
    if suffix not in SUPPORTED_RAW_SUFFIXES:
        warnings.append(WARNING_UNSUPPORTED_SUFFIX)
    if metadata.company_name is None:
        warnings.append(WARNING_MISSING_COMPANY)
    if metadata.fiscal_year is None:
        warnings.append(WARNING_MISSING_FISCAL_YEAR)
    if metadata.doc_type is None:
        warnings.append(WARNING_MISSING_DOC_TYPE)
    if file_size < small_file_threshold_bytes:
        warnings.append(WARNING_SUSPICIOUSLY_SMALL_FILE)

    return RawRegistryEntry(
        document_key=_document_key(metadata=metadata, checksum_sha256=checksum),
        local_path=local_path,
        filename=path.name,
        suffix=suffix,
        file_size=file_size,
        checksum_sha256=checksum,
        inferred_company_name=metadata.company_name,
        inferred_fiscal_year=metadata.fiscal_year,
        fiscal_period=str(metadata.fiscal_year) if metadata.fiscal_year is not None else None,
        inferred_doc_type=metadata.doc_type,
        inferred_market=metadata.market,
        language=metadata.language,
        warnings=_sort_warnings(warnings),
    )


def _infer_filename_metadata(path: Path) -> _FilenameMetadata:
    stem = _normalize_text(path.stem)
    return _FilenameMetadata(
        company_name=_infer_company_name(stem),
        fiscal_year=_infer_fiscal_year(stem),
        doc_type=_infer_doc_type(stem),
        market=_infer_market(stem),
        language="zh" if _CJK_RE.search(stem) else "en",
    )


def _infer_company_name(stem: str) -> str | None:
    cleaned = _normalize_metadata_separators(stem)
    for pattern in _METADATA_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    tokens = [
        token.strip()
        for token in _TOKEN_SPLIT_RE.split(cleaned)
        if token.strip()
    ]
    tokens = [token for token in tokens if not _is_generic_company_token(token)]
    if not tokens:
        return None
    if all(_CJK_RE.search(token) and not re.search(r"[A-Za-z]", token) for token in tokens):
        company = "".join(tokens)
    else:
        company = " ".join(tokens)
    company = re.sub(r"\s+", " ", company).strip(" _-—–,，.()（）[]【】")
    return company or None


def _is_generic_company_token(token: str) -> bool:
    normalized = token.casefold()
    if normalized in _GENERIC_COMPANY_TOKENS:
        return True
    if re.fullmatch(r"\d+", token):
        return True
    return False


def _infer_fiscal_year(stem: str) -> int | None:
    numeric_match = _NUMERIC_YEAR_RE.search(stem)
    if numeric_match is not None:
        return int(numeric_match.group(1))
    chinese_match = _CHINESE_YEAR_RE.search(stem)
    if chinese_match is None:
        return None
    digits = "".join(str(_chinese_digit_to_int(char)) for char in chinese_match.group(1))
    return int(digits) if digits.startswith("20") else None


def _chinese_digit_to_int(char: str) -> int:
    return {
        "〇": 0,
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }[char]


def _infer_doc_type(stem: str) -> str | None:
    normalized = _normalize_metadata_separators(stem).casefold()
    if _contains_any(normalized, ("20 f", "20f", "form 20 f", "form 20f")):
        return FilingDocType.FORM_20F.value
    if _contains_any(normalized, ("6 k", "6k", "form 6 k", "form 6k")):
        return FilingDocType.FORM_6K.value
    if _contains_any(normalized, ("业绩预告", "業績預告", "profit warning")):
        return FilingDocType.EARNINGS_PREVIEW.value
    if _contains_any(normalized, ("问询函回复", "問詢函回覆", "response letter")):
        return FilingDocType.RESPONSE_LETTER.value
    if _contains_any(
        normalized,
        (
            "earnings release",
            "unaudited financial results",
            "results announcement",
            "业绩公告",
            "業績公告",
            "业绩报告",
            "業績報告",
        ),
    ):
        return FilingDocType.EARNINGS_RELEASE.value
    if _contains_any(
        normalized,
        (
            "interim report",
            "half year report",
            "quarterly report",
            "q1 report",
            "q2 report",
            "q3 report",
            "q4 report",
            "中期报告",
            "中期報告",
            "半年报",
            "半年報",
            "半年度报告",
            "半年度報告",
            "季度报告",
            "季度報告",
            "一季报",
            "一季報",
            "三季报",
            "三季報",
        ),
    ):
        return FilingDocType.INTERIM_REPORT.value
    if _contains_any(
        normalized,
        ("annual report", "年度报告", "年度報告", "年报", "年報"),
    ):
        return FilingDocType.ANNUAL_REPORT.value
    return None


def _infer_market(stem: str) -> str | None:
    normalized = _normalize_metadata_separators(stem).casefold()
    if _contains_any(normalized, ("adr", "nyse", "nasdaq", "sec", "20 f", "20f", "6 k", "6k")):
        return Market.ADR.value
    if _contains_any(normalized, ("h share", "h股", "hkex", "港股", "联交所", "聯交所")):
        return Market.H_SHARE.value
    if _contains_any(normalized, ("a share", "a股", "sse", "szse", "上交所", "深交所")):
        return Market.A_SHARE.value
    return None


def _contains_any(text: str, candidates: tuple[str, ...]) -> bool:
    return any(candidate in text for candidate in candidates)


def _with_duplicate_checksum_warnings(entries: list[RawRegistryEntry]) -> list[RawRegistryEntry]:
    by_checksum: dict[str, list[RawRegistryEntry]] = defaultdict(list)
    for entry in entries:
        by_checksum[entry.checksum_sha256].append(entry)

    updated: list[RawRegistryEntry] = []
    for entry in entries:
        warnings = list(entry.warnings)
        if len(by_checksum[entry.checksum_sha256]) > 1:
            warnings.append(WARNING_DUPLICATE_CHECKSUM)
        updated.append(entry.model_copy(update={"warnings": _sort_warnings(warnings)}))
    return updated


def _summarize(entries: list[RawRegistryEntry]) -> RawRegistrySummary:
    warning_counts = Counter(warning for entry in entries for warning in entry.warnings)
    checksum_counts = Counter(entry.checksum_sha256 for entry in entries)
    suffix_counts = Counter(entry.suffix or "<none>" for entry in entries)

    return RawRegistrySummary(
        total_files=len(entries),
        supported_files=sum(1 for entry in entries if entry.suffix in SUPPORTED_RAW_SUFFIXES),
        unsupported_files=sum(1 for entry in entries if entry.suffix not in SUPPORTED_RAW_SUFFIXES),
        warning_count=sum(warning_counts.values()),
        warnings_by_type=_ordered_counts(warning_counts),
        duplicate_checksum_groups=sum(1 for count in checksum_counts.values() if count > 1),
        suffix_counts=dict(sorted(suffix_counts.items())),
    )


def _ordered_counts(counts: Counter[str]) -> dict[str, int]:
    ordered = {warning: counts[warning] for warning in WARNING_ORDER if counts[warning]}
    for warning in sorted(set(counts) - set(ordered)):
        ordered[warning] = counts[warning]
    return ordered


def _sort_warnings(warnings: list[str]) -> list[str]:
    seen = set()
    unique = [warning for warning in warnings if not (warning in seen or seen.add(warning))]
    order = {warning: index for index, warning in enumerate(WARNING_ORDER)}
    return sorted(unique, key=lambda warning: (order.get(warning, len(order)), warning))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_key(*, metadata: _FilenameMetadata, checksum_sha256: str) -> str:
    key_parts = [
        metadata.company_name,
        str(metadata.fiscal_year) if metadata.fiscal_year is not None else None,
        metadata.doc_type,
        metadata.market,
    ]
    slug = _slugify("_".join(part for part in key_parts if part))
    short_hash = checksum_sha256[:8]
    if not slug:
        return f"document-{short_hash}"
    if len(slug) > 80:
        slug = slug[:80].rstrip("-_")
    return f"{slug}-{short_hash}"


def _slugify(value: str) -> str:
    normalized = _normalize_text(value).casefold().replace("\\", "/")
    slug = re.sub(r"[^\w]+", "-", normalized, flags=re.UNICODE)
    return slug.strip("-_")


def _stable_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _normalize_metadata_separators(text: str) -> str:
    return _METADATA_SEPARATOR_RE.sub(" ", _normalize_text(text))
