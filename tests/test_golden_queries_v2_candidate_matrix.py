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


def test_universal6_selection_is_fixed_and_unique() -> None:
    module = _load_script_module("build_golden_queries_v2_universal6_anchor")

    assert module.DEFAULT_REVIEW_NOTES == Path(
        "docs/eval_inputs/golden_queries_v2_universal6_review_notes.json"
    )
    selected = [
        (definition.query_id, definition.company, definition.query)
        for definition in module.UNIVERSAL6_DEFINITIONS
    ]

    assert selected == [
        ("U-01", "招商银行", "招商银行本报告期营业收入、归母净利润和 ROE/ROAE 分别是多少？"),
        ("U-02", "腾讯控股", "腾讯控股收入按业务分部如何构成？哪个分部最大？"),
        ("U-03", "贵州茅台", "贵州茅台本期经营活动现金流净额是多少？与净利润相比如何？"),
        ("U-04", "阿里巴巴", "阿里巴巴收入或利润变化的主要原因是什么？"),
        ("U-06", "比亚迪", "比亚迪本期研发投入、资本开支或长期投资有什么披露？"),
        ("U-08", "中国平安", "中国平安披露了哪些主要风险以及应对措施？"),
    ]
    assert len({query_id for query_id, _company, _query in selected}) == 6
    assert len({company for _query_id, company, _query in selected}) == 6
    assert len({query for _query_id, _company, query in selected}) == 6


def test_universal6_rows_are_smoke_manifest_builder_compatible(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_universal6_anchor")
    manifest_module = _load_script_module("build_golden_queries_v2_smoke_manifest")
    candidate_matrix_path, raw_registry_path = _write_universal6_inputs(tmp_path, module)

    report = module.build_universal6_anchor_report(
        candidate_matrix_path=candidate_matrix_path,
        raw_registry_path=raw_registry_path,
        top_pages=1,
    )

    assert report["schema_version"] == "golden_queries_v2_universal6_anchor_matrix.v1"
    assert report["selected_query_ids"] == ["U-01", "U-02", "U-03", "U-04", "U-06", "U-08"]
    assert report["summary"]["total_rows"] == 6
    assert report["summary"]["ready_for_manifest"] == 0
    assert all(row["human_confirmed_pages"] == [] for row in report["rows"])
    assert all(row["human_corrected_pages"] == [] for row in report["rows"])
    assert all(row["expected_pages"] == [] for row in report["rows"])

    manifest_matrix_path = tmp_path / "universal6_matrix.json"
    manifest_row = dict(report["rows"][0])
    manifest_row["human_confirmed_pages"] = [3]
    manifest_row["human_corrected_pages"] = [4, 3]
    manifest_row["expected_pages"] = module._expected_pages_from_human(manifest_row)
    manifest_matrix_path.write_text(
        json.dumps({"rows": [manifest_row]}, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest_report = manifest_module.build_manifest_report(matrix_path=manifest_matrix_path)

    assert manifest_report["summary"]["included_cases"] == 1
    assert manifest_report["included_cases"][0]["expected_pages"] == [3, 4]
    assert manifest_report["manifest"]["queries"][0]["expected_pages"] == [3, 4]


def test_universal6_expected_pages_only_come_from_human_fields() -> None:
    module = _load_script_module("build_golden_queries_v2_universal6_anchor")
    row = {
        "candidate_pages": [7, 8],
        "human_confirmed_pages": [8],
        "human_corrected_pages": [9, 8],
    }

    assert module._expected_pages_from_human(row) == [8, 9]

    row["human_confirmed_pages"] = []
    row["human_corrected_pages"] = []
    assert module._expected_pages_from_human(row) == []


def test_universal6_review_notes_keep_codex_suggestions_out_of_expected_pages(
    tmp_path: Path,
) -> None:
    module = _load_script_module("build_golden_queries_v2_universal6_anchor")
    candidate_matrix_path, raw_registry_path = _write_universal6_inputs(tmp_path, module)
    review_notes_path = tmp_path / "review_notes.json"
    review_notes_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "company": "招商银行",
                        "query_id": "U-01",
                        "status": "human_confirmed_candidate_page",
                        "human_confirmed_pages": [8],
                    },
                    {
                        "company": "腾讯控股",
                        "query_id": "U-02",
                        "status": "human_confirmed_candidate_hit_pending_gold_page",
                        "codex_suggested_gold_pages": [9],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_universal6_anchor_report(
        candidate_matrix_path=candidate_matrix_path,
        raw_registry_path=raw_registry_path,
        review_notes_path=review_notes_path,
        top_pages=1,
    )

    rows = {row["query_id"]: row for row in report["rows"]}
    assert rows["U-01"]["expected_pages"] == [8]
    assert rows["U-01"]["manifest_readiness"] == "ready_for_manifest"
    assert rows["U-02"]["codex_suggested_gold_pages"] == [9]
    assert rows["U-02"]["expected_pages"] == []
    assert rows["U-02"]["manifest_readiness"] == "needs_anchor_confirmation"


def test_universal6_review_packet_contains_full_query_text(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_universal6_anchor")
    candidate_matrix_path, raw_registry_path = _write_universal6_inputs(tmp_path, module)

    report = module.build_universal6_anchor_report(
        candidate_matrix_path=candidate_matrix_path,
        raw_registry_path=raw_registry_path,
        top_pages=1,
    )
    rendered = module.render_review_packet(report)

    assert "完整 query" in rendered
    assert "human 页" in rendered
    assert "Codex 建议页" in rendered
    for definition in module.UNIVERSAL6_DEFINITIONS:
        assert definition.company in rendered
        assert definition.query_id in rendered
        assert definition.query in rendered


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


def _write_universal6_inputs(tmp_path: Path, module: ModuleType) -> tuple[Path, Path]:
    candidate_rows = []
    raw_documents = []
    for definition in module.UNIVERSAL6_DEFINITIONS:
        raw_path = tmp_path / f"{definition.query_id}.html"
        raw_path.write_text(
            "<html><body><p>"
            + " ".join(definition.search_terms[:5])
            + " 2025 100 90 人民币 百万元"
            + "</p></body></html>",
            encoding="utf-8",
        )
        document_key = f"{definition.company}_{definition.query_id}_doc"
        candidate_rows.append(
            {
                "company": definition.company,
                "document_key": document_key,
                "local_path": str(raw_path),
                "filing_class": "annual_report",
                "universal_candidates": [definition.query_id],
                "industry_candidates": [],
                "status": "candidate_anchor_pending",
                "notes": [],
            }
        )
        raw_documents.append(
            _registry_doc(
                document_key=document_key,
                filename=raw_path.name,
                local_path=str(raw_path),
                company_name=definition.company,
            )
        )

    candidate_matrix_path = tmp_path / "candidate_matrix.json"
    raw_registry_path = tmp_path / "raw_registry.json"
    candidate_matrix_path.write_text(
        json.dumps({"primary_current_year_candidates": candidate_rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    raw_registry_path.write_text(
        json.dumps({"documents": raw_documents}, ensure_ascii=False),
        encoding="utf-8",
    )
    return candidate_matrix_path, raw_registry_path


def _load_script_module(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
