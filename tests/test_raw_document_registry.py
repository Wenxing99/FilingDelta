from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from filingdelta.ingestion.raw_registry import (
    WARNING_DUPLICATE_CHECKSUM,
    WARNING_MISSING_COMPANY,
    WARNING_MISSING_DOC_TYPE,
    WARNING_MISSING_FISCAL_YEAR,
    WARNING_SUSPICIOUSLY_SMALL_FILE,
    WARNING_UNSUPPORTED_SUFFIX,
    scan_raw_document_registry,
)


def _load_build_registry_main():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_raw_document_registry.py"
    spec = importlib.util.spec_from_file_location("build_raw_document_registry", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_scan_raw_document_registry_handles_empty_directory(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    assert registry.documents == []
    assert registry.summary.total_files == 0
    assert registry.summary.warning_count == 0
    assert registry.summary.warnings_by_type == {}


def test_scan_raw_document_registry_registers_normal_pdf(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    content = b"%PDF-1.4\n" + b"x" * 2048
    source_path = raw_dir / "招商银行_2025_年度报告_A股.pdf"
    source_path.write_bytes(content)

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    assert registry.summary.total_files == 1
    entry = registry.documents[0]
    assert entry.local_path == "data/raw/招商银行_2025_年度报告_A股.pdf"
    assert entry.filename == "招商银行_2025_年度报告_A股.pdf"
    assert entry.suffix == ".pdf"
    assert entry.file_size == len(content)
    assert entry.checksum_sha256 == hashlib.sha256(content).hexdigest()
    assert entry.document_key.startswith("招商银行_2025_annual_report_a_share-")
    assert entry.company_id is None
    assert entry.ticker is None
    assert entry.industry is None
    assert entry.inferred_company_name == "招商银行"
    assert entry.inferred_fiscal_year == 2025
    assert entry.fiscal_period == "2025"
    assert entry.inferred_doc_type == "annual_report"
    assert entry.inferred_market == "a_share"
    assert entry.language == "zh"
    assert entry.source_url is None
    assert entry.notes is None
    assert entry.status == "registered"
    assert entry.warnings == []


def test_scan_raw_document_registry_recognizes_new_industry_pdf_names(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    content = b"%PDF-1.4\n" + b"x" * 2048
    for filename in (
        "中国石油2025年年度报告.pdf",
        "分众传媒2025年年度报告.PDF",
        "长江电力2025年年度报告.pdf",
    ):
        (raw_dir / filename).write_bytes(content + filename.encode("utf-8"))

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    by_company = {entry.inferred_company_name: entry for entry in registry.documents}
    assert set(by_company) == {"中国石油", "分众传媒", "长江电力"}
    assert {entry.inferred_fiscal_year for entry in registry.documents} == {2025}
    assert {entry.inferred_doc_type for entry in registry.documents} == {"annual_report"}
    assert {entry.suffix for entry in registry.documents} == {".pdf"}
    assert all(entry.warnings == [] for entry in registry.documents)


def test_scan_raw_document_registry_marks_duplicate_checksums(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    content = b"%PDF-1.4\n" + b"same" * 512
    (raw_dir / "Tencent_2025_Annual_Report_H_Share.pdf").write_bytes(content)
    (raw_dir / "Tencent_copy_2025_Annual_Report_H_Share.pdf").write_bytes(content)

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    assert registry.summary.duplicate_checksum_groups == 1
    assert registry.summary.warnings_by_type[WARNING_DUPLICATE_CHECKSUM] == 2
    assert all(WARNING_DUPLICATE_CHECKSUM in entry.warnings for entry in registry.documents)
    assert {entry.inferred_doc_type for entry in registry.documents} == {"annual_report"}
    assert {entry.inferred_market for entry in registry.documents} == {"h_share"}


def test_document_key_is_not_based_on_local_path_when_metadata_and_checksum_match(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    first_dir = raw_dir / "first"
    second_dir = raw_dir / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    content = b"%PDF-1.4\n" + b"same filing" * 256
    filename = "Tencent_2025_Annual_Report_H_Share.pdf"
    (first_dir / filename).write_bytes(content)
    (second_dir / filename).write_bytes(content)

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    assert len({entry.document_key for entry in registry.documents}) == 1
    assert len({entry.local_path for entry in registry.documents}) == 2
    assert all(WARNING_DUPLICATE_CHECKSUM in entry.warnings for entry in registry.documents)


def test_scan_raw_document_registry_supports_chinese_html_filename(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    html = "<html>" + ("腾讯控股业绩" * 200) + "</html>"
    source_path = raw_dir / "腾讯控股_二零二五年_年报_H股.html"
    source_path.write_text(html, encoding="utf-8")

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    entry = registry.documents[0]
    assert entry.local_path == "data/raw/腾讯控股_二零二五年_年报_H股.html"
    assert entry.suffix == ".html"
    assert entry.inferred_company_name == "腾讯控股"
    assert entry.inferred_fiscal_year == 2025
    assert entry.inferred_doc_type == "annual_report"
    assert entry.inferred_market == "h_share"
    assert entry.language == "zh"
    assert WARNING_UNSUPPORTED_SUFFIX not in entry.warnings


def test_scan_raw_document_registry_infers_english_quarterly_report(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "cmb_2024_q1_report.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 2048)

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    entry = registry.documents[0]
    assert entry.inferred_company_name == "cmb"
    assert entry.inferred_fiscal_year == 2024
    assert entry.fiscal_period == "2024"
    assert entry.inferred_doc_type == "interim_report"
    assert WARNING_MISSING_DOC_TYPE not in entry.warnings


def test_scan_raw_document_registry_infers_unaudited_financial_results(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "PDD_2025_Unaudited_Financial_Results.htm").write_bytes(
        b"<html>" + b"x" * 2048 + b"</html>"
    )

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    entry = registry.documents[0]
    assert entry.inferred_company_name == "PDD"
    assert entry.inferred_fiscal_year == 2025
    assert entry.inferred_doc_type == "earnings_release"
    assert WARNING_MISSING_DOC_TYPE not in entry.warnings


def test_scan_raw_document_registry_keeps_unsupported_suffix(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    content = b"notes" * 512
    (raw_dir / "招商银行_2025_年度报告_A股.txt").write_bytes(content)

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    entry = registry.documents[0]
    assert entry.suffix == ".txt"
    assert entry.status == "registered"
    assert WARNING_UNSUPPORTED_SUFFIX in entry.warnings
    assert registry.summary.unsupported_files == 1
    assert registry.summary.warnings_by_type[WARNING_UNSUPPORTED_SUFFIX] == 1


def test_scan_raw_document_registry_warns_on_small_file_and_missing_metadata(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "unknown.pdf").write_bytes(b"x")

    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    entry = registry.documents[0]
    assert entry.inferred_company_name is None
    assert entry.inferred_fiscal_year is None
    assert entry.inferred_doc_type is None
    assert {
        WARNING_MISSING_COMPANY,
        WARNING_MISSING_FISCAL_YEAR,
        WARNING_MISSING_DOC_TYPE,
        WARNING_SUSPICIOUSLY_SMALL_FILE,
    }.issubset(set(entry.warnings))


def test_build_raw_document_registry_script_writes_json_and_prints_summary(
    tmp_path,
    capsys,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_path = tmp_path / "registry.json"
    (raw_dir / "Tencent_2025_Annual_Report_ADR.pdf").write_bytes(
        b"%PDF-1.4\n" + b"x" * 2048
    )
    build_registry_main = _load_build_registry_main()

    registry = build_registry_main(
        ["--raw-dir", str(raw_dir), "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert registry.summary.total_files == 1
    assert payload["summary"]["total_files"] == 1
    assert payload["documents"][0]["status"] == "registered"
    assert payload["documents"][0]["inferred_doc_type"] == "annual_report"
    assert payload["documents"][0]["inferred_market"] == "adr"
    assert "summary: total_files=1" in captured.out
