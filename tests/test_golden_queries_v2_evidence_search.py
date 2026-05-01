from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def test_forbidden_failure_terms_are_diagnostic_not_positive_scoring() -> None:
    module = _load_script_module("search_golden_queries_v2_evidence")
    row = _search_row(local_path="unused.html")

    forbidden_only_hits = module._search_pages(
        row=row,
        pages=[(1, "future target used as historical actual 2025 100 90")],
        top_pages=5,
    )

    assert forbidden_only_hits == []

    hits = module._search_pages(
        row=row,
        pages=[
            (
                1,
                "\n".join(
                    [
                        "核心本地商业收入 100 90",
                        "新业务收入 20 10",
                        "future target used as historical actual",
                    ]
                ),
            )
        ],
        top_pages=5,
    )

    assert hits
    payload = hits[0].to_json()
    assert "future target used as historical actual" not in payload["matched_terms"]
    assert "future target used as historical actual" in payload["forbidden_matched_terms"]


def test_evidence_search_report_schema_and_review_state(tmp_path: Path) -> None:
    module = _load_script_module("search_golden_queries_v2_evidence")
    raw_path = tmp_path / "filing.html"
    raw_path.write_text(
        "\n".join(
            [
                "<html><body>",
                "<p>核心本地商业收入 100 90</p>",
                "<p>新业务收入 20 10</p>",
                "<p>future target used as historical actual</p>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "rows": [
                    _search_row(local_path=str(raw_path)),
                    {
                        "case_id": "blocked::HYDRO-01",
                        "query_id": "HYDRO-01",
                        "company": "长江电力 / Yangtze Power",
                        "industry": "hydropower",
                        "manifest_readiness": "blocked_missing_raw",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_evidence_search_report(matrix_path=matrix_path, top_pages=1)
    searched = report["cases"][0]
    blocked = report["cases"][1]

    assert report["schema_version"] == "golden_queries_v2_evidence_search.v1"
    assert report["summary"]["cases_total"] == 2
    assert report["summary"]["blocked_missing_raw"] == 1
    assert searched["candidate_pages"] == [1]
    assert searched["page_number_policy"] == module.PAGE_NUMBER_POLICY
    assert searched["review_priority"] == "later_anchor_confirmation"
    assert searched["codex_probe_status"] == "not_requested"
    assert searched["forbidden_matched_terms"] == ["future target used as historical actual"]
    assert "future target used as historical actual" not in searched["matched_terms"]
    assert blocked["status"] == "blocked_missing_raw"
    assert blocked["review_priority"] == "must_review"
    assert blocked["page_number_policy"] == module.PAGE_NUMBER_POLICY


def test_no_hit_rows_are_codex_probe_not_user_review(tmp_path: Path) -> None:
    module = _load_script_module("search_golden_queries_v2_evidence")
    raw_path = tmp_path / "filing.html"
    raw_path.write_text("<html><body><p>irrelevant text only</p></body></html>", encoding="utf-8")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps({"rows": [_search_row(local_path=str(raw_path))]}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = module.build_evidence_search_report(matrix_path=matrix_path, top_pages=1)
    row = report["cases"][0]

    assert row["status"] == "needs_manual_probe"
    assert row["review_priority"] == "codex_probe"
    assert row["codex_probe_status"] == "not_requested"
    assert row["candidate_pages"] == []


def test_codex_deep_probe_records_no_hit_after_targeted_search(tmp_path: Path) -> None:
    module = _load_script_module("search_golden_queries_v2_evidence")
    raw_path = tmp_path / "haier.html"
    raw_path.write_text("<html><body><p>债务融资工具存续情况</p></body></html>", encoding="utf-8")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps({"rows": [_haier_search_row(local_path=str(raw_path))]}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = module.build_evidence_search_report(matrix_path=matrix_path, top_pages=1)
    row = report["cases"][0]

    assert row["status"] == "needs_manual_probe"
    assert row["review_priority"] == "codex_probe"
    assert row["codex_probe_status"] == "codex_probe_no_hit_after_deep_search"
    assert row["codex_probe_pages"] == []
    assert "存货" in row["codex_probe_terms"]


def test_codex_deep_probe_can_supply_candidate_pages(tmp_path: Path) -> None:
    module = _load_script_module("search_golden_queries_v2_evidence")
    raw_path = tmp_path / "haier.html"
    raw_path.write_text(
        "<html><body><p>渠道库存周转效率提升，库存周转率改善 10%。</p></body></html>",
        encoding="utf-8",
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps({"rows": [_haier_search_row(local_path=str(raw_path))]}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = module.build_evidence_search_report(matrix_path=matrix_path, top_pages=1)
    row = report["cases"][0]

    assert row["candidate_pages"] == [1]
    assert row["codex_probe_status"] == "codex_probe_candidate_found"
    assert row["codex_probe_pages"] == [1]
    assert row["codex_probe_score"] > 0


def test_industry_matrix_keeps_candidates_out_of_manifest_gold(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_industry_matrix")
    candidate_matrix_path = tmp_path / "candidate_matrix.json"
    anchor_probe_path = tmp_path / "anchor_probe.json"
    evidence_search_path = tmp_path / "evidence_search.json"

    candidate_matrix_path.write_text(
        json.dumps(
            {
                "primary_current_year_candidates": [
                    {
                        "company": "美团",
                        "document_key": "meituan_doc",
                        "local_path": "data/raw/meituan.html",
                        "industry_candidates": ["LOCAL-01"],
                        "status": "candidate_anchor_pending",
                    }
                ],
                "blocked_design_queries": [
                    {
                        "query_id": "HYDRO-01",
                        "required_company": "长江电力 / Yangtze Power",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    anchor_probe_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "meituan_doc::LOCAL-01",
                        "status": "anchor_probe_hit",
                        "matched_pages": [{"page": 4}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    evidence_search_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "meituan_doc::LOCAL-01",
                        "status": "auto_anchor_low_confidence",
                        "review_priority": "later_anchor_confirmation",
                        "candidate_pages": [4],
                        "candidate_snippets": ["核心本地商业收入 100 90"],
                        "matched_terms": ["核心本地商业", "收入"],
                        "forbidden_matched_terms": [],
                        "evidence_search_score": 18,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_matrix_report(
        candidate_matrix_path=candidate_matrix_path,
        anchor_probe_path=anchor_probe_path,
        evidence_search_path=evidence_search_path,
    )
    rows = report["rows"]
    searched = next(row for row in rows if row["query_id"] == "LOCAL-01")

    assert report["summary"]["total_rows"] == 2
    assert report["summary"]["review_rows"] == 1
    assert report["summary"]["ready_for_manifest"] == 0
    assert searched["manifest_readiness"] == "needs_anchor_confirmation"
    assert searched["review_priority"] == "later_anchor_confirmation"
    assert searched["candidate_pages"] == [4]
    assert searched["expected_pages"] == []
    assert "expected_evidence_kinds" not in searched
    assert searched["primary_evidence_kind"] == "table_row"
    assert searched["secondary_evidence_kinds"] == []


def test_industry_matrix_routes_no_hit_rows_to_codex_probe(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_industry_matrix")
    candidate_matrix_path = tmp_path / "candidate_matrix.json"
    anchor_probe_path = tmp_path / "anchor_probe.json"
    evidence_search_path = tmp_path / "evidence_search.json"

    candidate_matrix_path.write_text(
        json.dumps(
            {
                "primary_current_year_candidates": [
                    {
                        "company": "海尔智家",
                        "document_key": "haier_doc",
                        "local_path": "data/raw/haier.pdf",
                        "industry_candidates": ["HA-03"],
                        "status": "candidate_anchor_pending",
                    }
                ],
                "blocked_design_queries": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    anchor_probe_path.write_text(
        json.dumps({"cases": []}),
        encoding="utf-8",
    )
    evidence_search_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "haier_doc::HA-03",
                        "status": "needs_manual_probe",
                        "review_priority": "codex_probe",
                        "candidate_pages": [],
                        "candidate_snippets": [],
                        "matched_terms": [],
                        "forbidden_matched_terms": [],
                        "evidence_search_score": 0,
                        "codex_probe_status": "codex_probe_no_hit_after_deep_search",
                        "codex_probe_pages": [],
                        "codex_probe_snippets": [],
                        "codex_probe_matched_terms": [],
                        "codex_probe_score": 0,
                        "codex_probe_notes": "Codex deep probe 未找到可用候选页。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_matrix_report(
        candidate_matrix_path=candidate_matrix_path,
        anchor_probe_path=anchor_probe_path,
        evidence_search_path=evidence_search_path,
    )
    row = report["rows"][0]

    assert report["summary"]["review_rows"] == 0
    assert report["summary"]["codex_probe_rows"] == 1
    assert report["summary"]["codex_probe_no_hit_rows"] == 1
    assert row["review_priority"] == "codex_probe"
    assert row["candidate_pages"] == []
    assert row["codex_probe_status"] == "codex_probe_no_hit_after_deep_search"


def test_industry_matrix_merges_human_review_notes_without_expected_pages(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_industry_matrix")
    candidate_matrix_path = tmp_path / "candidate_matrix.json"
    anchor_probe_path = tmp_path / "anchor_probe.json"
    evidence_search_path = tmp_path / "evidence_search.json"
    review_notes_path = tmp_path / "review_notes.json"

    candidate_matrix_path.write_text(
        json.dumps(
            {
                "primary_current_year_candidates": [
                    {
                        "company": "美的",
                        "document_key": "midea_doc",
                        "local_path": "data/raw/midea.pdf",
                        "industry_candidates": ["HA-01"],
                        "status": "candidate_anchor_pending",
                    }
                ],
                "blocked_design_queries": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    anchor_probe_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    evidence_search_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "midea_doc::HA-01",
                        "status": "auto_anchor_high_confidence",
                        "review_priority": "sample_review",
                        "candidate_pages": [42],
                        "candidate_snippets": ["智能家居业务"],
                        "matched_terms": ["智能家居"],
                        "forbidden_matched_terms": [],
                        "evidence_search_score": 48,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    review_notes_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "company": "美的",
                        "query_id": "HA-01",
                        "status": "human_corrected_page",
                        "human_corrected_pages": [49],
                        "human_review_notes": "候选页未命中，人工核对后第 49 页正确。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_matrix_report(
        candidate_matrix_path=candidate_matrix_path,
        anchor_probe_path=anchor_probe_path,
        evidence_search_path=evidence_search_path,
        anchor_review_notes_path=review_notes_path,
    )
    row = report["rows"][0]

    assert report["summary"]["human_reviewed_rows"] == 1
    assert report["summary"]["human_corrected_rows"] == 1
    assert row["candidate_pages"] == [42]
    assert row["human_corrected_pages"] == [49]
    assert row["expected_pages"] == []


def test_anchor_confirmation_draft_keeps_manifest_gold_empty(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_anchor_confirmation")
    raw_path = tmp_path / "haier.html"
    raw_path.write_text(
        "\n".join(
            [
                "<html><body>",
                "<p>中国市场增长来自渠道效率和产品结构升级。</p>",
                "<p>海外市场增长由品牌和高端产品驱动。</p>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "rows": [
                    _anchor_matrix_row(
                        company="海尔智家",
                        query_id="HA-02",
                        local_path=str(raw_path),
                        candidate_pages=[1],
                        expected_answer_field_ids=[
                            "china_drivers",
                            "overseas_drivers",
                            "product_or_channel_examples",
                        ],
                    ),
                    _anchor_matrix_row(
                        company="美团",
                        query_id="LOCAL-02",
                        local_path=str(raw_path),
                        candidate_pages=[],
                        expected_answer_field_ids=[
                            "operating_loss",
                            "management_stated_changes",
                        ],
                    ),
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_confirmation_report(matrix_path=matrix_path)
    haier = next(case for case in report["cases"] if case["query_id"] == "HA-02")
    meituan = next(case for case in report["cases"] if case["query_id"] == "LOCAL-02")

    assert report["schema_version"] == "golden_queries_v2_anchor_confirmation_draft.v1"
    assert report["summary"]["total_rows"] == 2
    assert report["summary"]["candidate_rows"] == 1
    assert report["summary"]["ready_for_manifest"] == 0
    assert haier["codex_anchor_status"] == "codex_anchor_confirmed_candidate"
    assert haier["codex_anchor_pages"] == [1]
    assert haier["codex_anchor_missing_fields"] == []
    assert "expected_pages" not in haier
    assert meituan["codex_anchor_status"] == "codex_anchor_deferred_no_hit"
    assert meituan["codex_anchor_pages"] == []
    assert "美团 PDF 文本抽取" in meituan["codex_anchor_notes"]
    assert all("human_confirmed_pages" not in case for case in report["cases"])


def test_anchor_confirmation_review_packet_is_small_user_subset(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_anchor_confirmation")
    raw_path = tmp_path / "filing.html"
    raw_path.write_text(
        "<html><body><p>集装箱航运业务货运量 TEU 收入 同比 人民币 元</p></body></html>",
        encoding="utf-8",
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "rows": [
                    _anchor_matrix_row(
                        company="中远海控",
                        query_id="SHIP-01",
                        local_path=str(raw_path),
                        candidate_pages=[1],
                        expected_answer_field_ids=["teu_volume", "revenue", "yoy", "unit"],
                    ),
                    _anchor_matrix_row(
                        company="比亚迪",
                        query_id="NEV-01",
                        local_path=str(raw_path),
                        candidate_pages=[1],
                        expected_answer_field_ids=["auto_revenue"],
                    ),
                    _anchor_matrix_row(
                        company="美团",
                        query_id="LOCAL-02",
                        local_path=str(raw_path),
                        candidate_pages=[],
                        expected_answer_field_ids=["operating_loss"],
                    ),
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_confirmation_report(matrix_path=matrix_path)
    packet_keys = {
        (case["company"], case["query_id"]) for case in report["review_packet"]["cases"]
    }

    assert packet_keys == {("中远海控", "SHIP-01"), ("美团", "LOCAL-02")}
    assert report["summary"]["review_packet_rows"] == 2
    assert report["summary"]["candidate_rows"] == 2
    rendered = module.render_review_packet(report)
    assert "问题" in rendered
    assert "中远海控" in rendered
    assert "测试问题" in rendered
    assert "比亚迪" not in rendered


def test_anchor_confirmation_review_packet_separates_user_feedback(tmp_path: Path) -> None:
    module = _load_script_module("build_golden_queries_v2_anchor_confirmation")
    raw_path = tmp_path / "haier.html"
    raw_path.write_text(
        "<html><body><p>库存周转效率 渠道 数字库存 第 31 页 第 32 页</p></body></html>",
        encoding="utf-8",
    )
    row = _anchor_matrix_row(
        company="海尔智家",
        query_id="HA-03",
        local_path=str(raw_path),
        candidate_pages=[1],
        expected_answer_field_ids=["inventory_value_or_turnover", "channel_explanation"],
    )
    row.update(
        {
            "anchor_review_status": "human_corrected_page",
            "human_confirmed_pages": [31],
            "human_corrected_pages": [32],
            "human_review_notes": "总体没问题，比较重要的信息位于第 31-32 页；候选页缺第 32 页。",
        }
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps({"rows": [row]}, ensure_ascii=False), encoding="utf-8")

    report = module.build_confirmation_report(matrix_path=matrix_path)
    rendered = module.render_review_packet(report)

    assert report["summary"]["review_packet_reviewed_rows"] == 1
    assert report["summary"]["review_packet_pending_rows"] == 0
    assert "## 已收到用户反馈" in rendered
    assert "## 需要用户抽查" not in rendered
    assert "31" in rendered
    assert "32" in rendered
    assert "候选页缺第 32 页" in rendered


def _search_row(*, local_path: str) -> dict:
    return {
        "case_id": "meituan_doc::LOCAL-01",
        "query_id": "LOCAL-01",
        "company": "美团",
        "industry": "local_services",
        "local_path": local_path,
        "manifest_readiness": "needs_anchor_confirmation",
        "query": "美团核心本地商业和新业务收入分别是多少？",
        "area": "segment revenue",
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence_kind": "table_row",
        "secondary_evidence_kinds": [],
        "expected_answer_field_ids": [
            "core_local_commerce_revenue",
            "new_initiatives_revenue",
        ],
        "forbidden_failure_modes": ["future target used as historical actual"],
    }


def _anchor_matrix_row(
    *,
    company: str,
    query_id: str,
    local_path: str,
    candidate_pages: list[int],
    expected_answer_field_ids: list[str],
) -> dict:
    return {
        "case_id": f"{company}_doc::{query_id}",
        "query_id": query_id,
        "company": company,
        "industry": "test_industry",
        "document_key": f"{company}_doc",
        "local_path": local_path,
        "query": "测试问题",
        "primary_evidence_kind": "table_row",
        "secondary_evidence_kinds": ["section_text"],
        "candidate_pages": candidate_pages,
        "candidate_snippets": [],
        "matched_terms": ["收入", "同比", "渠道", "海外市场", "中国市场"],
        "codex_probe_matched_terms": [],
        "evidence_search_score": 42 if candidate_pages else 0,
        "auto_anchor_status": "auto_anchor_high_confidence"
        if candidate_pages
        else "needs_manual_probe",
        "review_priority": "later_anchor_confirmation",
        "codex_probe_status": "not_requested",
        "codex_probe_pages": [],
        "expected_answer_field_ids": expected_answer_field_ids,
        "forbidden_failure_modes": [],
        "manifest_readiness": "needs_anchor_confirmation"
        if candidate_pages
        else "needs_manual_probe",
        "anchor_review_status": "not_reviewed",
        "human_confirmed_pages": [],
        "human_corrected_pages": [],
        "human_rejected_candidate_pages": [],
        "human_missing_fields": [],
        "human_review_notes": "",
    }


def _haier_search_row(*, local_path: str) -> dict:
    return {
        "case_id": "haier_doc::HA-03",
        "query_id": "HA-03",
        "company": "海尔智家",
        "industry": "home_appliances",
        "local_path": local_path,
        "manifest_readiness": "needs_manual_probe",
        "query": "家电企业存货或渠道库存是否异常？公司如何描述渠道效率？",
        "area": "inventory, channel review",
        "expected_document_evidence_intent": "metric_attribution",
        "primary_evidence_kind": "table_row",
        "secondary_evidence_kinds": ["section_text"],
        "expected_answer_field_ids": [
            "inventory_value_or_turnover",
            "channel_explanation",
        ],
        "forbidden_failure_modes": ["all inventory increases described as unsold goods"],
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
