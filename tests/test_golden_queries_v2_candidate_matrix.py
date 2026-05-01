from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def test_builder_promotes_new_raw_files_out_of_blocked_queries(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_candidate_matrix")
    raw_registry_path = tmp_path / "raw_document_registry.json"
    parser_smoke_path = tmp_path / "parser_smoke_report.json"

    documents = [
        _registry_doc(
            document_key="中国石油_2025_annual_report-ptr",
            filename="中国石油2025年年度报告.pdf",
            local_path="data/raw/中国石油2025年年度报告.pdf",
            company_name="中国石油",
        ),
        _registry_doc(
            document_key="长江电力_2025_annual_report-hydro",
            filename="长江电力2025年年度报告.pdf",
            local_path="data/raw/长江电力2025年年度报告.pdf",
            company_name="长江电力",
        ),
        _registry_doc(
            document_key="分众传媒_2025_annual_report-media",
            filename="分众传媒2025年年度报告.PDF",
            local_path="data/raw/分众传媒2025年年度报告.PDF",
            company_name="分众传媒",
        ),
    ]
    raw_registry_path.write_text(
        json.dumps({"documents": documents}, ensure_ascii=False),
        encoding="utf-8",
    )
    parser_smoke_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"document_key": document["document_key"], "status": "passed"}
                    for document in documents
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_candidate_matrix_report(
        raw_registry_path=raw_registry_path,
        parser_smoke_report_path=parser_smoke_path,
    )

    query_ids = {
        query_id
        for row in report["primary_current_year_candidates"]
        for query_id in row["industry_candidates"]
    }
    assert query_ids == {
        "PTR-01",
        "HYDRO-01",
        "HYDRO-02",
        "HYDRO-03",
        "MEDIA-01",
        "MEDIA-02",
    }
    assert report["blocked_design_queries"] == []
    assert report["counts"]["primary_industry_candidate_instances"] == 6
    assert report["counts"]["blocked_design_queries"] == 0


def test_builder_keeps_raw_without_parser_smoke_pass_blocked(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_candidate_matrix")
    raw_registry_path = tmp_path / "raw_document_registry.json"
    parser_smoke_path = tmp_path / "parser_smoke_report.json"
    document = _registry_doc(
        document_key="长江电力_2025_annual_report-hydro",
        filename="长江电力2025年年度报告.pdf",
        local_path="data/raw/长江电力2025年年度报告.pdf",
        company_name="长江电力",
    )
    raw_registry_path.write_text(
        json.dumps({"documents": [document]}, ensure_ascii=False),
        encoding="utf-8",
    )
    parser_smoke_path.write_text(
        json.dumps(
            {"documents": [{"document_key": document["document_key"], "status": "failed"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_candidate_matrix_report(
        raw_registry_path=raw_registry_path,
        parser_smoke_report_path=parser_smoke_path,
    )

    blocked = {row["query_id"] for row in report["blocked_design_queries"]}
    assert {"HYDRO-01", "HYDRO-02", "HYDRO-03"}.issubset(blocked)
    assert report["primary_current_year_candidates"] == []


def test_builder_accepts_registry_20f_doc_type_value(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_candidate_matrix")
    raw_registry_path = tmp_path / "raw_document_registry.json"
    parser_smoke_path = tmp_path / "parser_smoke_report.json"
    document = _registry_doc(
        document_key="trip-com_2025_20f_adr-test",
        filename="Trip.com_2025_20F.pdf",
        local_path="data/raw/Trip.com_2025_20F.pdf",
        company_name="Trip com",
        doc_type="20f",
    )
    raw_registry_path.write_text(
        json.dumps({"documents": [document]}, ensure_ascii=False),
        encoding="utf-8",
    )
    parser_smoke_path.write_text(
        json.dumps(
            {"documents": [{"document_key": document["document_key"], "status": "passed"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_candidate_matrix_report(
        raw_registry_path=raw_registry_path,
        parser_smoke_report_path=parser_smoke_path,
    )

    assert report["primary_current_year_candidates"][0]["filing_class"] == "20f"
    assert report["primary_current_year_candidates"][0]["industry_candidates"] == [
        "OTA-01",
        "OTA-02",
    ]


def _registry_doc(
    *,
    document_key: str,
    filename: str,
    local_path: str,
    company_name: str,
    doc_type: str = "annual_report",
) -> dict:
    return {
        "document_key": document_key,
        "local_path": local_path,
        "filename": filename,
        "suffix": Path(filename).suffix.lower(),
        "file_size": 2048,
        "checksum_sha256": "x" * 64,
        "inferred_company_name": company_name,
        "inferred_fiscal_year": 2025,
        "fiscal_period": "2025",
        "inferred_doc_type": doc_type,
        "inferred_market": None,
        "language": "zh",
        "warnings": [],
    }


def _load_script_module(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
