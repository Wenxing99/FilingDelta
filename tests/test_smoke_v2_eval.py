from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from filingdelta.eval.smoke_v2 import (
    SmokeV2ManifestError,
    SmokeV2Observation,
    build_smoke_v2_case_result,
    build_builtin_placeholder_manifest_payload,
    build_smoke_v2_report,
    evaluate_answer_hygiene,
    load_smoke_v2_manifest_from_payload,
    select_smoke_v2_cases,
)


def test_load_manifest_with_document_registry_and_query_schema() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())

    assert manifest.version == "test_smoke_v2"
    assert manifest.documents.require("cmb_2025_annual").exists is True
    assert manifest.queries[0].case_id == "CMB-DEP-01"
    assert manifest.queries[0].expected_evidence_kinds == ("table_row", "page_text")


def test_manifest_rejects_unknown_document_key() -> None:
    payload = _manifest_payload()
    payload["queries"][0]["document_key"] = "missing_doc"

    with pytest.raises(SmokeV2ManifestError, match="Unknown document_key"):
        load_smoke_v2_manifest_from_payload(payload, base_dir=_base_dir())


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("expected_route", "web_search", "invalid expected_route"),
        (
            "expected_document_evidence_intent",
            "deep_research",
            "invalid expected_document_evidence_intent",
        ),
        ("primary_evidence_kind", "spreadsheet_cell", "invalid primary_evidence_kind"),
    ],
)
def test_manifest_rejects_invalid_route_intent_and_primary_evidence_kind(
    field_name,
    bad_value,
    message,
) -> None:
    payload = _manifest_payload()
    payload["queries"][0][field_name] = bad_value

    with pytest.raises(SmokeV2ManifestError, match=message):
        load_smoke_v2_manifest_from_payload(payload, base_dir=_base_dir())


def test_manifest_rejects_invalid_secondary_evidence_kind() -> None:
    payload = _manifest_payload()
    payload["queries"][0]["secondary_evidence_kinds"] = ["page_text", "spreadsheet_cell"]

    with pytest.raises(SmokeV2ManifestError, match="invalid secondary_evidence_kinds"):
        load_smoke_v2_manifest_from_payload(payload, base_dir=_base_dir())


def test_manifest_rejects_duplicate_query_id() -> None:
    payload = _manifest_payload()
    payload["queries"].append(dict(payload["queries"][0]))

    with pytest.raises(SmokeV2ManifestError, match="Duplicate query id"):
        load_smoke_v2_manifest_from_payload(payload, base_dir=_base_dir())


@pytest.mark.parametrize("default_top_k", [0, -1])
def test_manifest_rejects_non_positive_default_top_k(default_top_k) -> None:
    payload = _manifest_payload()
    payload["default_top_k"] = default_top_k

    with pytest.raises(SmokeV2ManifestError, match="default_top_k must be an integer >= 1"):
        load_smoke_v2_manifest_from_payload(payload, base_dir=_base_dir())


def test_manifest_rejects_unknown_hygiene_check_id() -> None:
    payload = _manifest_payload()
    payload["queries"][0]["answer_hygiene_checks"] = ["no_raw_metadata", "unknown_check"]

    with pytest.raises(SmokeV2ManifestError, match="invalid answer_hygiene_checks"):
        load_smoke_v2_manifest_from_payload(payload, base_dir=_base_dir())


def test_select_cases_filters_by_case_and_intent() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())

    selected = select_smoke_v2_cases(
        manifest.queries,
        case_ids={"CMB-DEP-01"},
        intents={"metric_value"},
    )

    assert [case.case_id for case in selected] == ["CMB-DEP-01"]


def test_build_case_result_scores_observed_route_intent_and_evidence() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]

    result = build_smoke_v2_case_result(
        case=case,
        observation=SmokeV2Observation(
            route="document_only",
            document_evidence_intent="metric_value",
            retrieved_evidence_kinds=("table_row", "page_text"),
            citation_pages=(30, 47),
            retrieved_row_labels=("customer deposits",),
            retrieved_metric_tags=("customer_deposits",),
            answer_field_ids=("deposit_balance", "deposit_change"),
            answer_text="In the 2025 annual report, customer deposits were RMB 98.36 billion.",
            latency_ms=12,
        ),
        top_k=6,
    )

    assert result["scores"]["route_hit"] is True
    assert result["scores"]["intent_hit"] is True
    assert result["scores"]["evidence_kind_hit@6"] is True
    assert result["scores"]["page_hit@6"] is True
    assert result["scores"]["table_row_label_hit@6"] is True
    assert result["scores"]["metric_tag_hit@6"] is True
    assert result["scores"]["citation_anchor_valid"] is True
    assert result["scores"]["required_fields_present"] is True
    assert result["scores"]["output_hygiene_passed"] is True


def test_evaluated_empty_observation_scores_false_for_required_live_fields() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]

    result = build_smoke_v2_case_result(
        case=case,
        observation=SmokeV2Observation(),
        top_k=6,
        status="evaluated",
    )

    assert result["scores"]["route_hit"] is False
    assert result["scores"]["intent_hit"] is False
    assert result["scores"]["evidence_kind_hit@6"] is False
    assert result["scores"]["page_hit@6"] is False
    assert result["scores"]["table_row_label_hit@6"] is False
    assert result["scores"]["metric_tag_hit@6"] is False
    assert result["scores"]["required_fields_present"] is False
    assert result["scores"]["citation_anchor_valid"] is False
    assert result["scores"]["forbidden_failure_absent"] is False
    assert result["scores"]["output_hygiene_passed"] is False


def test_hygiene_checks_catch_raw_metadata_and_empty_parentheses() -> None:
    checks = evaluate_answer_hygiene(
        answer_text="Revenue was RMB 10 billion in 2025 (DOC_1). () score=0.9",
        check_ids=("no_raw_metadata", "no_empty_parentheses", "unit_period_present"),
    )

    assert checks["no_raw_metadata"]["passed"] is False
    assert checks["no_empty_parentheses"]["passed"] is False
    assert checks["unit_period_present"]["passed"] is True


def test_direct_hygiene_evaluator_reports_unknown_check_defensively() -> None:
    checks = evaluate_answer_hygiene(
        answer_text="Revenue was RMB 10 billion in 2025.",
        check_ids=("unknown_check",),
    )

    assert checks["unknown_check"]["passed"] is None
    assert checks["unknown_check"]["message"] == "Unknown hygiene check: unknown_check"


def test_validate_only_and_dry_run_statuses_are_distinct() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    cases = list(manifest.queries)

    validate_report = build_smoke_v2_report(
        manifest=manifest,
        cases=cases,
        mode="validate_only",
        top_k=6,
    )
    dry_run_report = build_smoke_v2_report(
        manifest=manifest,
        cases=cases,
        mode="dry_run",
        top_k=6,
    )

    assert validate_report["queries"][0]["status"] == "validated"
    assert validate_report["queries"][0]["skip_reason"] is None
    assert dry_run_report["queries"][0]["status"] == "dry_run_skipped"
    assert dry_run_report["queries"][0]["skip_reason"] == (
        "dry run requested; live router/retriever execution is not wired in this skeleton"
    )
    assert dry_run_report["queries"][0]["scores"]["evidence_kind_hit@6"] is None
    assert dry_run_report["queries"][0]["scores"]["required_fields_present"] is None


@pytest.mark.parametrize("top_k", [0, -1])
def test_report_rejects_non_positive_top_k(top_k) -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())

    with pytest.raises(SmokeV2ManifestError, match="top_k must be an integer >= 1"):
        build_smoke_v2_report(
            manifest=manifest,
            cases=list(manifest.queries),
            mode="validate_only",
            top_k=top_k,
        )


def test_dry_run_report_is_structured_when_source_document_is_missing() -> None:
    payload = _manifest_payload()
    payload["documents"][0]["source_path"] = "does-not-exist.pdf"
    manifest = load_smoke_v2_manifest_from_payload(payload, base_dir=_base_dir())

    report = build_smoke_v2_report(
        manifest=manifest,
        cases=list(manifest.queries),
        mode="dry_run",
        top_k=6,
    )

    assert report["mode"] == "dry_run"
    assert report["summary"]["total_queries"] == 1
    assert report["queries"][0]["status"] == "dry_run_skipped"
    assert report["queries"][0]["observed"]["answer_hygiene"]["no_raw_metadata"]["passed"] is None


def test_builtin_placeholder_manifest_loads_cmb_and_tencent_cases() -> None:
    manifest = load_smoke_v2_manifest_from_payload(
        build_builtin_placeholder_manifest_payload(),
        base_dir=_base_dir(),
    )

    assert manifest.version == "smoke_v2_placeholder_v0"
    assert [case.case_id for case in manifest.queries] == ["CMB-DEP-01", "TCEHY-CAPEX-01"]
    assert {case.primary_evidence_kind for case in manifest.queries} == {"table_row"}


def test_runner_list_cases_prints_selected_cases(capsys) -> None:
    run_smoke_v2_main = _load_run_smoke_v2_main()

    result = run_smoke_v2_main(["--use-built-in-placeholders", "--list-cases"])

    captured = capsys.readouterr()
    assert result is None
    assert "CMB-DEP-01" in captured.out
    assert "TCEHY-CAPEX-01" in captured.out


def test_runner_exits_when_filters_select_no_cases(tmp_path) -> None:
    run_smoke_v2_main = _load_run_smoke_v2_main()

    with pytest.raises(SystemExit) as caught:
        run_smoke_v2_main(
            [
                "--use-built-in-placeholders",
                "--company",
                "NoSuchCompany",
                "--output",
                str(tmp_path / "smoke_v2_report.json"),
            ]
        )

    assert caught.value.code == "No smoke_v2 cases selected."


def test_runner_writes_json_output_for_validate_only(tmp_path) -> None:
    output_path = tmp_path / "smoke_v2_report.json"
    run_smoke_v2_main = _load_run_smoke_v2_main()

    report = run_smoke_v2_main(
        [
            "--use-built-in-placeholders",
            "--validate-only",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert report is not None
    assert payload["version"] == "smoke_v2_eval_v0"
    assert payload["mode"] == "validate_only"
    assert payload["summary"]["total_queries"] == 2
    assert payload["summary"]["status_counts"] == {"validated": 2}


@pytest.mark.parametrize("top_k", [0, -1])
def test_runner_rejects_non_positive_top_k(tmp_path, top_k) -> None:
    run_smoke_v2_main = _load_run_smoke_v2_main()

    with pytest.raises(SmokeV2ManifestError, match="top_k must be an integer >= 1"):
        run_smoke_v2_main(
            [
                "--use-built-in-placeholders",
                "--validate-only",
                "--top-k",
                str(top_k),
                "--output",
                str(tmp_path / "smoke_v2_report.json"),
            ]
        )


def _load_run_smoke_v2_main():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_smoke_v2_eval.py"
    spec = importlib.util.spec_from_file_location("run_smoke_v2_eval", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _base_dir():
    return Path(__file__).resolve().parent


def _manifest_payload() -> dict:
    return {
        "version": "test_smoke_v2",
        "suite": "golden_queries_v2",
        "default_top_k": 6,
        "documents": [
            {
                "document_key": "cmb_2025_annual",
                "source_path": "test_smoke_v2_eval.py",
                "company_name": "CMB",
                "ticker": "600036",
                "market": "a_share",
                "doc_type": "annual_report",
                "fiscal_period": "2025 annual report",
                "language": "zh",
                "industry": "banking",
            }
        ],
        "queries": [
            {
                "id": "CMB-DEP-01",
                "tier": "smoke_v2",
                "company": "CMB",
                "industry": "banking",
                "document_key": "cmb_2025_annual",
                "query": "How did customer deposits change?",
                "query_aliases": [],
                "expected_route": "document_only",
                "expected_document_evidence_intent": "metric_value",
                "primary_evidence_kind": "table_row",
                "secondary_evidence_kinds": ["page_text"],
                "expected_pages": [30, 47],
                "expected_row_labels": ["customer deposits"],
                "expected_metric_tags": ["customer_deposits"],
                "expected_section_types": [],
                "expected_document_area_ids": ["deposit_table"],
                "expected_answer_field_ids": ["deposit_balance", "deposit_change"],
                "forbidden_failure_modes": ["company-only deposits"],
                "answer_hygiene_checks": [
                    "no_raw_metadata",
                    "no_empty_parentheses",
                    "unit_period_present",
                ],
                "mvp_status": "immediate",
            }
        ],
    }
