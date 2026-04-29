from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from filingdelta.ingestion import parser_smoke
from filingdelta.ingestion.parser_smoke import (
    ParserSmokeSample,
    load_raw_document_registry_json,
    run_parser_smoke_check,
)
from filingdelta.ingestion.raw_registry import scan_raw_document_registry


def _load_run_parser_smoke_main():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_parser_smoke_check.py"
    spec = importlib.util.spec_from_file_location("run_parser_smoke_check", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_pdf_entry_passes_with_stubbed_pymupdf_sample(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pdf_path = raw_dir / "Tencent_2025_Annual_Report_H_Share.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nplaceholder")
    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    def fake_extract_pdf_sample(source_path: Path, *, sample_pages_per_pdf: int):
        assert source_path == pdf_path
        assert sample_pages_per_pdf == 2
        return ParserSmokeSample(
            page_count_estimate=12,
            sample_text="Tencent annual report readable text",
            sample_pages_checked=[1, 2],
        )

    monkeypatch.setattr(parser_smoke, "_extract_pdf_sample", fake_extract_pdf_sample)

    report = run_parser_smoke_check(
        registry,
        repo_root=tmp_path,
        sample_pages_per_pdf=2,
    )

    result = report.documents[0]
    assert result.status == "passed"
    assert result.parser_kind_candidate == "pymupdf"
    assert result.page_count_estimate == 12
    assert result.sample_text_chars == len("Tencent annual report readable text")
    assert result.sample_pages_checked == [1, 2]
    assert report.summary.passed_count == 1


def test_pdf_extractor_error_fails_with_structured_error(tmp_path, monkeypatch) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pdf_path = raw_dir / "Tencent_2025_Annual_Report_H_Share.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nplaceholder")
    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    def fake_extract_pdf_sample(source_path: Path, *, sample_pages_per_pdf: int):
        assert source_path == pdf_path
        assert sample_pages_per_pdf == 3
        raise RuntimeError("cannot open malformed pdf")

    monkeypatch.setattr(parser_smoke, "_extract_pdf_sample", fake_extract_pdf_sample)

    report = run_parser_smoke_check(registry, repo_root=tmp_path)

    result = report.documents[0]
    assert result.status == "failed"
    assert result.parser_kind_candidate == "pymupdf"
    assert result.error_type == "RuntimeError"
    assert result.error_message == "cannot open malformed pdf"
    assert result.sample_text_chars == 0
    assert result.sample_pages_checked == []
    assert report.summary.failed_count == 1
    assert report.summary.error_counts == {"RuntimeError": 1}


def test_html_entry_passes_with_visible_text(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "Tencent_2025_Annual_Report_H_Share.html").write_text(
        "<html><head><script>hidden()</script></head>"
        "<body><main><p>Visible filing text</p></main></body></html>",
        encoding="utf-8",
    )
    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    report = run_parser_smoke_check(registry, repo_root=tmp_path)

    result = report.documents[0]
    assert result.status == "passed"
    assert result.parser_kind_candidate == "html_tag"
    assert result.page_count_estimate == 1
    assert result.sample_pages_checked == [1]
    assert result.sample_text_chars == len("Visible filing text")


def test_unsupported_entry_is_skipped(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "Tencent_2025_Annual_Report_H_Share.txt").write_text("notes", encoding="utf-8")
    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    report = run_parser_smoke_check(registry, repo_root=tmp_path)

    result = report.documents[0]
    assert result.status == "skipped"
    assert result.parser_kind_candidate == "unsupported"
    assert result.error_type is None
    assert report.summary.skipped_count == 1
    assert report.summary.unsupported_documents == 1


def test_missing_supported_file_fails_with_structured_error(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pdf_path = raw_dir / "Tencent_2025_Annual_Report_H_Share.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nplaceholder")
    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)
    pdf_path.unlink()

    report = run_parser_smoke_check(registry, repo_root=tmp_path)

    result = report.documents[0]
    assert result.status == "failed"
    assert result.error_type == "FileNotFoundError"
    assert "File not found" in result.error_message
    assert report.summary.failed_count == 1


def test_empty_html_text_fails_with_empty_text_error(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "Tencent_2025_Annual_Report_H_Share.html").write_text(
        "<html><body><script>hidden()</script>   </body></html>",
        encoding="utf-8",
    )
    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)

    report = run_parser_smoke_check(registry, repo_root=tmp_path)

    result = report.documents[0]
    assert result.status == "failed"
    assert result.error_type == "EmptyTextError"
    assert result.sample_text_chars == 0
    assert result.sample_pages_checked == [1]
    assert "empty_extracted_text" in result.warnings
    assert report.summary.error_counts == {"EmptyTextError": 1}


def test_registry_json_loader_round_trips_raw_registry(tmp_path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "Tencent_2025_Annual_Report_H_Share.html").write_text(
        "<html><body>Visible filing text</body></html>",
        encoding="utf-8",
    )
    registry = scan_raw_document_registry(raw_dir=raw_dir, repo_root=tmp_path)
    registry_path = tmp_path / "raw_document_registry.json"
    registry_path.write_text(
        json.dumps(registry.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )

    loaded = load_raw_document_registry_json(registry_path, repo_root=tmp_path)

    assert loaded.documents[0].document_key == registry.documents[0].document_key
    assert loaded.raw_dir == "data/raw"


def test_run_parser_smoke_check_script_writes_json_and_prints_summary(
    tmp_path,
    capsys,
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "Tencent_2025_Annual_Report_H_Share.html").write_text(
        "<html><body>Visible filing text</body></html>",
        encoding="utf-8",
    )
    (raw_dir / "notes.txt").write_text("unsupported", encoding="utf-8")
    output_path = tmp_path / "parser_smoke_report.json"
    run_parser_smoke_main = _load_run_parser_smoke_main()

    report = run_parser_smoke_main(
        [
            "--raw-dir",
            str(raw_dir),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert report.summary.total_documents == 2
    assert payload["schema_version"] == "parser_smoke_report.v1"
    assert payload["summary"]["passed_count"] == 1
    assert payload["summary"]["skipped_count"] == 1
    assert "summary: total_documents=2" in captured.out


def test_run_parser_smoke_check_script_exits_on_failure_when_requested(
    tmp_path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pdf_path = raw_dir / "Tencent_2025_Annual_Report_H_Share.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nplaceholder")
    output_path = tmp_path / "parser_smoke_report.json"

    def fake_extract_pdf_sample(source_path: Path, *, sample_pages_per_pdf: int):
        assert source_path == pdf_path
        raise RuntimeError("cannot open malformed pdf")

    monkeypatch.setattr(parser_smoke, "_extract_pdf_sample", fake_extract_pdf_sample)
    run_parser_smoke_main = _load_run_parser_smoke_main()

    with pytest.raises(SystemExit) as caught:
        run_parser_smoke_main(
            [
                "--raw-dir",
                str(raw_dir),
                "--output",
                str(output_path),
                "--fail-on-failure",
            ]
        )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert caught.value.code == 1
    assert payload["summary"]["failed_count"] == 1
    assert payload["summary"]["error_counts"] == {"RuntimeError": 1}
