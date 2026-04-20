from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from datetime import datetime

from filingdelta.core.config import REPO_ROOT
from filingdelta.schemas.demo import DemoDocument
from filingdelta.schemas.filing import FilingDocType, FilingSource, Market


RAW_DOCS_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "small_doc_benchmark.json"
SUPPORTED_SUFFIXES = {".pdf", ".htm", ".html"}


def _build_document_id(source: FilingSource) -> str:
    digest = hashlib.md5(str(source.source_path).encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"doc-{digest[:10]}"


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


def get_demo_document_sources() -> dict[str, FilingSource]:
    ordered_sources: list[FilingSource] = []
    known_paths: set[Path] = set()

    for source in _load_manifest_sources():
        resolved_path = source.source_path.resolve()
        if not resolved_path.exists():
            continue
        source.source_path = resolved_path
        ordered_sources.append(source)
        known_paths.add(resolved_path)

    for loose_path in _iter_loose_document_paths():
        if loose_path in known_paths:
            continue
        ordered_sources.append(_infer_source_from_path(loose_path))
        known_paths.add(loose_path)

    return {_build_document_id(source): source for source in ordered_sources}


def list_demo_documents() -> list[DemoDocument]:
    documents: list[DemoDocument] = []
    for document_id, source in get_demo_document_sources().items():
        documents.append(_source_to_demo_document(document_id, source))
    return documents


def get_demo_document_source(document_id: str) -> FilingSource:
    source = get_demo_document_sources().get(document_id)
    if source is None:
        raise KeyError(f"Unknown demo document: {document_id}")
    return source


def import_demo_document(filename: str, content: bytes) -> DemoDocument:
    if not filename.strip():
        raise ValueError("导入文件失败：文件名不能为空。")
    if not content:
        raise ValueError("导入文件失败：文件内容为空。")

    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("导入文件失败：当前仅支持 PDF 和单文件 HTML（.htm/.html）。")

    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = _allocate_import_path(RAW_DOCS_DIR / safe_name)
    target_path.write_bytes(content)

    source = _infer_source_from_path(target_path.resolve())
    return _source_to_demo_document(_build_document_id(source), source)


def _source_to_demo_document(document_id: str, source: FilingSource) -> DemoDocument:
    return DemoDocument(
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


def _iter_loose_document_paths() -> list[Path]:
    if not RAW_DOCS_DIR.exists():
        return []

    loose_paths: list[Path] = []
    for path in RAW_DOCS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        loose_paths.append(path.resolve())

    loose_paths.sort(key=lambda item: item.name.lower())
    return loose_paths


def _infer_source_from_path(path: Path) -> FilingSource:
    stem = path.stem
    return FilingSource(
        source_path=path,
        company_name=_infer_company_name(stem),
        market=_infer_market(stem),
        doc_type=_infer_doc_type(stem),
        fiscal_period=_infer_fiscal_period(stem),
        language=_infer_language(stem),
    )


def _infer_company_name(stem: str) -> str:
    company_name = _normalized_stem(stem)
    cleanup_patterns = [
        r"20\d{2}\s*年(?:第[一二三四1-4]季度|半年度|年度)?报告(?:摘要)?",
        r"20\d{2}\s*年度报告摘要",
        r"20\d{2}\s*年度报告",
        r"20\d{2}\s*年第三季度报告",
        r"20\d{2}\s*年中期报告",
        r"20\d{2}\s*annual report",
        r"20\d{2}\s*interim report",
        r"20\d{2}\s*quarterly report",
        r"20\d{2}\s*unaudited financial results",
        r"annual report",
        r"interim report",
        r"quarterly report",
        r"unaudited financial results",
        r"financial results",
        r"earnings release",
        r"results announcement",
        r"报告摘要",
        r"年度报告",
        r"中期报告",
        r"季度报告",
        r"摘要",
        r"业绩公告",
        r"业绩快报",
    ]
    for pattern in cleanup_patterns:
        company_name = re.sub(pattern, "", company_name, flags=re.IGNORECASE)
    company_name = re.sub(r"\s+", " ", company_name).strip(" _-.")
    return company_name or stem


def _infer_market(stem: str) -> Market:
    normalized = _normalized_stem(stem)
    lowered = normalized.lower()
    if "a股" in stem or "a-share" in lowered or "ashare" in lowered:
        return Market.A_SHARE
    if "h股" in stem or "港股" in stem or "h-share" in lowered or ".hk" in lowered:
        return Market.H_SHARE
    if "adr" in lowered or "ads" in lowered:
        return Market.ADR
    return Market.OTHER


def _infer_doc_type(stem: str) -> FilingDocType:
    normalized = _normalized_stem(stem)
    lowered = normalized.lower()
    if "20-f" in lowered or "20f" in lowered:
        return FilingDocType.FORM_20F
    if "6-k" in lowered or "6k" in lowered:
        return FilingDocType.FORM_6K
    if "preview" in lowered or "预告" in stem:
        return FilingDocType.EARNINGS_PREVIEW
    if any(token in lowered for token in {"unaudited financial results", "earnings release", "results announcement"}):
        return FilingDocType.EARNINGS_RELEASE
    if any(token in stem for token in {"业绩公告", "业绩快报"}):
        return FilingDocType.EARNINGS_RELEASE
    if any(token in lowered for token in {"interim report", "quarterly report", "q1", "q2", "q3", "q4"}):
        return FilingDocType.INTERIM_REPORT
    if any(token in stem for token in {"中期报告", "季度报告", "第一季度", "第二季度", "第三季度", "第四季度", "半年度"}):
        return FilingDocType.INTERIM_REPORT
    if "annual report" in lowered or "年度报告" in stem:
        return FilingDocType.ANNUAL_REPORT
    return FilingDocType.OTHER


def _infer_fiscal_period(stem: str) -> str | None:
    normalized = _normalized_stem(stem)
    chinese_period = re.search(
        r"(20\d{2}\s*年(?:第[一二三四1-4]季度|半年度|年度)报告(?:摘要)?)",
        normalized,
    )
    if chinese_period:
        return chinese_period.group(1).replace(" ", "")

    english_period = re.search(
        r"(20\d{2}\s*(?:annual report|interim report|quarterly report|unaudited financial results))",
        normalized,
        flags=re.IGNORECASE,
    )
    if english_period:
        return english_period.group(1)

    quarter_match = re.search(r"(20\d{2}\s*Q[1-4])", normalized, flags=re.IGNORECASE)
    if quarter_match:
        return quarter_match.group(1).upper()

    year_match = re.search(r"(20\d{2})", normalized)
    if year_match:
        return year_match.group(1)

    return None


def _infer_language(stem: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", stem) else "en"


def _normalized_stem(stem: str) -> str:
    normalized = re.sub(r"[_\-]+", " ", stem)
    return re.sub(r"\s+", " ", normalized).strip()


def _allocate_import_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    candidate = target_path.with_name(f"{target_path.stem}_{timestamp}{target_path.suffix}")
    if not candidate.exists():
        return candidate

    counter = 2
    while True:
        candidate = target_path.with_name(f"{target_path.stem}_{timestamp}_{counter}{target_path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
