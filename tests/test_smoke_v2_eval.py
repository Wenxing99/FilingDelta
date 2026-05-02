from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from filingdelta.eval.retrieval_diagnosis import (
    BM25Index,
    RetrievalCandidate,
    RankSource,
    build_live_pilot_context_from_query,
    build_mode_result,
    build_original_failed_rescue,
    diagnosis_strategy_for_case,
    enrich_case_result_with_diagnosis_context,
    rank_semantic_chunks,
    reciprocal_rank_fusion,
    render_diagnosis_markdown,
)
from filingdelta.eval.smoke_v2 import (
    SmokeV2ManifestError,
    SmokeV2Observation,
    build_smoke_v2_case_result,
    build_builtin_placeholder_manifest_payload,
    build_smoke_v2_report,
    evaluate_answer_hygiene,
    load_smoke_v2_manifest_from_payload,
    render_smoke_v2_markdown_summary,
    select_smoke_v2_cases,
)
from filingdelta.schemas.chat import RetrievedChunk


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


def test_live_retrieval_report_scores_observed_router_and_retrieval_fields() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]

    report = build_smoke_v2_report(
        manifest=manifest,
        cases=[case],
        mode="live_retrieval",
        top_k=6,
        observations={
            case.case_id: SmokeV2Observation(
                executed=True,
                route="document_only",
                document_evidence_intent="metric_value",
                retrieval_mode="semantic_with_filters",
                retrieved_evidence_kinds=("table_row", "page_text"),
                citation_pages=(47, 30),
                latency_ms=25,
            )
        },
    )

    result = report["queries"][0]
    assert result["status"] == "passed"
    assert result["failure_reasons"] == []
    assert result["scores"]["route_hit"] is True
    assert result["scores"]["intent_hit"] is True
    assert result["scores"]["evidence_kind_hit@6"] is True
    assert result["scores"]["page_hit@6"] is True
    assert result["scores"]["required_fields_present"] is None
    assert result["scores"]["output_hygiene_passed"] is None
    assert report["summary"]["status_counts"] == {"passed": 1}


def test_live_retrieval_report_marks_page_miss_without_changing_gold() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]

    report = build_smoke_v2_report(
        manifest=manifest,
        cases=[case],
        mode="live_retrieval",
        top_k=6,
        observations={
            case.case_id: SmokeV2Observation(
                executed=True,
                route="document_only",
                document_evidence_intent="metric_value",
                retrieved_evidence_kinds=("table_row",),
                citation_pages=(99,),
            )
        },
    )

    result = report["queries"][0]
    assert result["status"] == "failed"
    assert result["expected"]["pages"] == [30, 47]
    assert any("page miss" in reason for reason in result["failure_reasons"])


def test_live_retrieval_markdown_summary_lists_case_results() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]
    report = build_smoke_v2_report(
        manifest=manifest,
        cases=[case],
        mode="live_retrieval",
        observations={
            case.case_id: SmokeV2Observation(
                executed=True,
                route="document_only",
                document_evidence_intent="metric_value",
                retrieved_evidence_kinds=("page_text",),
                citation_pages=(99,),
            )
        },
    )

    rendered = render_smoke_v2_markdown_summary(report)

    assert "Golden Queries v2 Smoke Pilot Summary" in rendered
    assert "CMB-DEP-01" in rendered
    assert "page miss" in rendered
    assert "1 条 anchor-confirmed case" in rendered
    assert "14 条 anchor-confirmed case" not in rendered


def test_bm25_scorer_ranks_matching_chunk_first() -> None:
    chunks = [
        _retrieved_chunk(
            chunk_id="weak",
            text="客户服务和渠道建设。",
            page_number=1,
        ),
        _retrieved_chunk(
            chunk_id="strong",
            text="核心本地商业收入和新业务收入分别披露。",
            page_number=2,
        ),
    ]

    results = BM25Index(chunks).search("核心本地商业收入是多少", top_k=2)

    assert [candidate.chunk.chunk_id for candidate in results] == ["strong"]
    assert results[0].rank_sources[0].source == "bm25"
    assert results[0].rank_sources[0].rank == 1


def test_rrf_fusion_dedupes_and_keeps_rank_sources() -> None:
    chunk_a_semantic = _retrieved_chunk(
        chunk_id="a-semantic",
        text="target semantic copy",
        page_number=1,
        chunk_kind="table_row",
        row_label="营业收入",
    )
    chunk_a_bm25 = _retrieved_chunk(
        chunk_id="a-bm25",
        text="target bm25 copy",
        page_number=1,
        chunk_kind="table_row",
        row_label="营业收入",
    )
    chunk_b = _retrieved_chunk(chunk_id="b", text="semantic only", page_number=2)
    chunk_c = _retrieved_chunk(chunk_id="c", text="bm25 only", page_number=3)
    semantic = [
        RetrievalCandidate(
            chunk=chunk_b,
            score=0.9,
            rank_sources=(RankSource(source="semantic", rank=1, score=0.9),),
        ),
        RetrievalCandidate(
            chunk=chunk_a_semantic,
            score=0.8,
            rank_sources=(RankSource(source="semantic", rank=2, score=0.8),),
        ),
    ]
    bm25 = [
        RetrievalCandidate(
            chunk=chunk_a_bm25,
            score=3.0,
            rank_sources=(RankSource(source="bm25", rank=1, score=3.0),),
        ),
        RetrievalCandidate(
            chunk=chunk_c,
            score=2.0,
            rank_sources=(RankSource(source="bm25", rank=2, score=2.0),),
        ),
    ]

    fused = reciprocal_rank_fusion(semantic, bm25, rrf_k=60)

    assert [candidate.chunk.chunk_id for candidate in fused] == ["a-semantic", "b", "c"]
    assert [(source.source, source.rank) for source in fused[0].rank_sources] == [
        ("semantic", 2),
        ("bm25", 1),
    ]


def test_retrieval_diagnosis_report_does_not_modify_expected_pages() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]

    result = build_mode_result(
        case=case,
        mode="bm25_only",
        candidates=[
            RetrievalCandidate(
                chunk=_retrieved_chunk(chunk_id="miss", page_number=99),
                score=1.0,
                rank_sources=(RankSource(source="bm25", rank=1, score=1.0),),
            )
        ],
        final_top_k=6,
        retrieval_ms=3,
    )

    assert case.expected_pages == (30, 47)
    assert result["expected_pages"] == [30, 47]
    assert result["retrieved_pages"] == [99]
    assert result["hit"] is False


def test_retrieval_diagnosis_strategy_matches_live_metric_value_top_k() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]

    strategy = diagnosis_strategy_for_case(case)

    assert strategy.primary_chunk_kind == "table_row"
    assert strategy.fallback_chunk_kinds == ("page_text",)
    assert strategy.include_fallback_when_primary_found is True
    assert strategy.primary_top_k == 8
    assert strategy.fallback_top_k == 4


def test_retrieval_diagnosis_strategy_matches_live_attribution_and_narrative_top_k() -> None:
    attribution_payload = _manifest_payload()
    attribution_payload["queries"][0]["expected_document_evidence_intent"] = "metric_attribution"
    attribution_payload["queries"][0]["primary_evidence_kind"] = "section_text"
    attribution_payload["queries"][0]["secondary_evidence_kinds"] = ["table_row", "page_text"]
    attribution_manifest = load_smoke_v2_manifest_from_payload(
        attribution_payload,
        base_dir=_base_dir(),
    )

    attribution_strategy = diagnosis_strategy_for_case(attribution_manifest.queries[0])

    assert attribution_strategy.primary_chunk_kind == "section_text"
    assert attribution_strategy.fallback_chunk_kinds == ("table_row", "page_text")
    assert attribution_strategy.include_fallback_when_primary_found is True
    assert attribution_strategy.primary_top_k == 4
    assert attribution_strategy.fallback_top_k == 3

    narrative_payload = _manifest_payload()
    narrative_payload["queries"][0]["expected_document_evidence_intent"] = "business_narrative"
    narrative_payload["queries"][0]["primary_evidence_kind"] = "section_text"
    narrative_payload["queries"][0]["secondary_evidence_kinds"] = ["page_text"]
    narrative_manifest = load_smoke_v2_manifest_from_payload(
        narrative_payload,
        base_dir=_base_dir(),
    )

    narrative_strategy = diagnosis_strategy_for_case(narrative_manifest.queries[0])

    assert narrative_strategy.primary_chunk_kind == "section_text"
    assert narrative_strategy.fallback_chunk_kinds == ("page_text",)
    assert narrative_strategy.include_fallback_when_primary_found is False
    assert narrative_strategy.primary_top_k == 6
    assert narrative_strategy.fallback_top_k == 6


def test_diagnosis_rescue_marks_live_intent_mismatch_as_page_only() -> None:
    live_pilot = build_live_pilot_context_from_query(
        {
            "status": "failed",
            "expected": {"document_evidence_intent": "metric_attribution"},
            "observed": {"document_evidence_intent": "business_narrative"},
            "scores": {"intent_hit": False, "page_hit@6": False, "route_hit": True},
            "failure_reasons": ["intent mismatch"],
        }
    )
    case_result = enrich_case_result_with_diagnosis_context(
        {
            "id": "海尔智家_2025_annual_report-14186f9f::HA-03",
            "query_id": "HA-03",
            "company": "海尔智家",
            "expected_document_evidence_intent": "metric_attribution",
            "expected_pages": [31, 32],
            "modes": {
                "semantic_only": _diagnosis_mode(hit=False),
                "bm25_only": _diagnosis_mode(hit=True),
                "hybrid_rrf": _diagnosis_mode(hit=True),
            },
        },
        live_pilot=live_pilot,
    )

    rescue = next(
        row for row in build_original_failed_rescue([case_result]) if row["query_id"] == "HA-03"
    )

    assert rescue["rescue_status"] == "page_rescued_but_live_intent_mismatch"
    assert rescue["pilot_status"] == "failed"
    assert rescue["pilot_observed_intent"] == "business_narrative"
    assert rescue["pilot_intent_hit"] is False
    assert rescue["full_live_pilot_rescue_claimed"] is False
    assert "page_rescued_but_live_intent_mismatch" in case_result["diagnosis_notes"]


def test_diagnosis_markdown_shows_page_hit_only_caveat_and_live_pilot_context() -> None:
    case_result = enrich_case_result_with_diagnosis_context(
        {
            "id": "海尔智家_2025_annual_report-14186f9f::HA-03",
            "query_id": "HA-03",
            "company": "海尔智家",
            "expected_document_evidence_intent": "metric_attribution",
            "expected_pages": [31, 32],
            "modes": {
                "semantic_only": _diagnosis_mode(hit=False),
                "bm25_only": _diagnosis_mode(hit=True),
                "hybrid_rrf": _diagnosis_mode(hit=True),
            },
        },
        live_pilot={
            "pilot_status": "failed",
            "pilot_expected_intent": "metric_attribution",
            "pilot_observed_intent": "business_narrative",
            "pilot_intent_hit": False,
            "pilot_page_hit_at_6": False,
            "pilot_route_hit": True,
            "pilot_failure_reasons": ["intent mismatch"],
        },
    )
    rescue_rows = build_original_failed_rescue([case_result])
    report = {
        "manifest_path": "manifest.json",
        "live_pilot_report": {
            "path": "pilot.json",
            "status": "loaded",
            "loaded_cases": 1,
        },
        "summary": {
            "total_cases": 1,
            "mode_hits": {
                "semantic_only": "0/1",
                "bm25_only": "1/1",
                "hybrid_rrf": "1/1",
            },
        },
        "original_failed_case_rescue": rescue_rows,
        "cases": [case_result],
    }

    rendered = render_diagnosis_markdown(report)

    assert "expected_intent_diagnosis/page_hit_only" in rendered
    assert "不等于 full live pilot rescue" in rendered
    assert "business_narrative" in rendered
    assert "page_rescued_but_live_intent_mismatch" in rendered


def test_semantic_only_mode_result_exposes_live_observation_shape() -> None:
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]
    semantic_candidates = rank_semantic_chunks(
        [
            _retrieved_chunk(
                chunk_id="hit",
                page_number=30,
                chunk_kind="table_row",
                row_label="customer deposits",
                score=0.7,
            )
        ],
        top_k=20,
    )

    result = build_mode_result(
        case=case,
        mode="semantic_only",
        candidates=semantic_candidates,
        final_top_k=6,
        retrieval_ms=5,
    )

    observed = result["observed"]
    assert observed["route"] == "document_only"
    assert observed["document_evidence_intent"] == "metric_value"
    assert observed["retrieval_mode"] == "semantic_only"
    assert observed["retrieved_evidence_kinds"] == ("table_row",)
    assert observed["citation_pages"] == (30,)


def test_page_text_shadow_mode_result_preserves_expected_pages_and_rejects_other_kinds() -> None:
    module = _load_page_text_shadow_module()
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]
    page_text_candidate = RetrievalCandidate(
        chunk=_retrieved_chunk(
            chunk_id="page-hit",
            page_number=30,
            chunk_kind="page_text",
        ),
        score=1.0,
        rank_sources=(RankSource(source="bm25", rank=1, score=1.0),),
    )

    result = module.build_shadow_mode_result(
        case=case,
        mode="bm25_page_text",
        candidates=[page_text_candidate],
        final_top_k=6,
        retrieval_ms=2,
    )

    assert case.expected_pages == (30, 47)
    assert result["expected_pages"] == [30, 47]
    assert result["retrieved_pages"] == [30]
    assert result["page_match_status"] == "primary_hit"
    assert result["observed"]["retrieved_evidence_kinds"] == ("page_text",)
    with pytest.raises(ValueError, match="only accepts page_text"):
        module.build_shadow_mode_result(
            case=case,
            mode="semantic_page_text",
            candidates=[
                RetrievalCandidate(
                    chunk=_retrieved_chunk(
                        chunk_id="table-row",
                        page_number=30,
                        chunk_kind="table_row",
                    ),
                    score=1.0,
                    rank_sources=(RankSource(source="semantic", rank=1, score=1.0),),
                )
            ],
            final_top_k=6,
            retrieval_ms=2,
        )


def test_page_text_mode_result_distinguishes_supporting_page_hit() -> None:
    module = _load_page_text_shadow_module()
    payload = _manifest_payload()
    payload["queries"][0]["supporting_pages"] = [81]
    manifest = load_smoke_v2_manifest_from_payload(payload, base_dir=_base_dir())
    case = manifest.queries[0]

    result = module.build_shadow_mode_result(
        case=case,
        mode="bm25_page_text",
        candidates=[
            RetrievalCandidate(
                chunk=_retrieved_chunk(
                    chunk_id="support-only",
                    page_number=81,
                    chunk_kind="page_text",
                ),
                score=1.0,
                rank_sources=(RankSource(source="bm25", rank=1, score=1.0),),
            )
        ],
        final_top_k=6,
        retrieval_ms=2,
    )

    assert result["hit"] is False
    assert result["supporting_hit"] is True
    assert result["supporting_pages"] == [81]
    assert result["supporting_hit_pages"] == [81]
    assert result["page_match_status"] == "partial_support_only"
    page_text_candidates = [
        RetrievalCandidate(
            chunk=_retrieved_chunk(
                chunk_id=f"page-text-{index}",
                page_number=index,
                chunk_kind="page_text",
            ),
            score=1.0,
            rank_sources=(RankSource(source="bm25", rank=index, score=1.0),),
        )
        for index in range(1, 7)
    ]
    with pytest.raises(ValueError, match="only accepts page_text"):
        module.build_shadow_mode_result(
            case=case,
            mode="semantic_page_text",
            candidates=[
                *page_text_candidates,
                RetrievalCandidate(
                    chunk=_retrieved_chunk(
                        chunk_id="late-table-row",
                        page_number=30,
                        chunk_kind="table_row",
                    ),
                    score=0.1,
                    rank_sources=(RankSource(source="semantic", rank=7, score=0.1),),
                ),
            ],
            final_top_k=6,
            retrieval_ms=2,
        )


def test_page_text_shadow_rrf_rank_sources_are_traceable() -> None:
    module = _load_page_text_shadow_module()
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]
    semantic = [
        RetrievalCandidate(
            chunk=_retrieved_chunk(
                chunk_id="semantic-hit",
                text="customer deposits changed in 2025",
                page_number=30,
                chunk_kind="page_text",
            ),
            score=0.8,
            rank_sources=(RankSource(source="semantic", rank=1, score=0.8),),
        )
    ]
    bm25 = BM25Index(
        [
            _retrieved_chunk(
                chunk_id="bm25-hit",
                text="customer deposits changed in 2025",
                page_number=30,
                chunk_kind="page_text",
            )
        ]
    ).search("customer deposits", top_k=1)

    result = module.build_shadow_mode_result(
        case=case,
        mode="hybrid_rrf_page_text",
        candidates=reciprocal_rank_fusion(semantic, bm25),
        final_top_k=6,
        retrieval_ms=2,
    )

    rank_sources = result["top_results"][0]["rank_sources"]
    assert result["top_results"][0]["chunk_kind"] == "page_text"
    assert {source["source"] for source in rank_sources} == {"semantic", "bm25"}


def test_page_text_shadow_summary_tracks_regression_and_baseline_page_miss_rescue() -> None:
    module = _load_page_text_shadow_module()
    baseline_passed = _page_text_shadow_case_result(
        case_id="doc::PASS-01",
        query_id="PASS-01",
        baseline_status="passed",
        baseline_page_hit=True,
        mode_hits={
            "semantic_page_text": True,
            "bm25_page_text": False,
            "hybrid_rrf_page_text": True,
        },
    )
    baseline_page_miss = _page_text_shadow_case_result(
        case_id="doc::MISS-01",
        query_id="MISS-01",
        baseline_status="failed",
        baseline_page_hit=False,
        mode_hits={
            "semantic_page_text": False,
            "bm25_page_text": True,
            "hybrid_rrf_page_text": True,
        },
    )
    partial_support = _page_text_shadow_case_result(
        case_id="doc::U-04",
        query_id="U-04",
        baseline_status="failed",
        baseline_page_hit=False,
        mode_hits={
            "semantic_page_text": False,
            "bm25_page_text": False,
            "hybrid_rrf_page_text": False,
        },
    )
    partial_support["supporting_pages"] = [81]
    partial_support["modes"]["hybrid_rrf_page_text"].update(
        {
            "supporting_hit": True,
            "supporting_pages": [81],
            "supporting_hit_pages": [81],
            "retrieved_pages": [81],
            "page_match_status": "partial_support_only",
        }
    )

    summary = module.build_page_text_shadow_summary(
        case_results=[baseline_passed, baseline_page_miss, partial_support],
        final_top_k=6,
    )

    assert summary["baseline_passed_total"] == 1
    assert summary["baseline_page_miss_total"] == 2
    assert summary["previous_passed_regressions"]["bm25_page_text"][0]["query_id"] == "PASS-01"
    rescue = summary["baseline_page_miss_rescue"][0]
    assert rescue["query_id"] == "MISS-01"
    assert rescue["rescued_by_modes"] == ["bm25_page_text", "hybrid_rrf_page_text"]
    assert summary["mode_summary"]["hybrid_rrf_page_text"]["supporting_page_hit_count"] == 1
    assert summary["mode_summary"]["hybrid_rrf_page_text"]["partial_support_only_cases"] == [
        "doc::U-04"
    ]
    assert summary["by_intent"]["metric_value"]["bm25_page_text"]["hit_count"] == 1
    assert summary["by_primary_evidence_kind"]["table_row"]["bm25_page_text"]["hit_count"] == 1


def test_page_text_shadow_markdown_shows_summary_rescue_regression_and_snippets() -> None:
    module = _load_page_text_shadow_module()
    case_result = _page_text_shadow_case_result(
        case_id="doc::MISS-01",
        query_id="MISS-01",
        baseline_status="failed",
        baseline_page_hit=False,
        mode_hits={
            "semantic_page_text": False,
            "bm25_page_text": True,
            "hybrid_rrf_page_text": True,
        },
    )
    summary = module.build_page_text_shadow_summary(
        case_results=[case_result],
        final_top_k=6,
    )
    summary["total_cases"] = 20
    report = {
        "manifest_path": "manifest.json",
        "baseline_report": {"path": "baseline.json"},
        "final_top_k": 6,
        "summary": summary,
        "cases": [case_result],
    }

    rendered = module.render_page_text_shadow_markdown(report)

    assert "## 20-case 总结" in rendered
    assert "Case 总数：`20`" in rendered
    assert "Baseline Page-Miss Rescue" in rendered
    assert "Baseline Passed Regression" in rendered
    assert "MISS-01" in rendered
    assert "top snippet for MISS-01" in rendered


def test_page_text_hybrid_grid_builds_coarse_config_count() -> None:
    module = _load_page_text_grid_module()

    configs = module.build_grid_configs()

    assert len(configs) == 160
    assert configs[0].semantic_top_n == 5
    assert configs[0].bm25_top_n == 5
    assert configs[0].rrf_k == 20
    assert configs[0].alpha_semantic == 0.25


def test_weighted_rrf_alpha_extremes_follow_single_source_rank() -> None:
    module = _load_page_text_grid_module()
    semantic = [
        _grid_candidate("semantic-1", page_number=1, source="semantic", rank=1),
        _grid_candidate("semantic-2", page_number=2, source="semantic", rank=2),
    ]
    bm25 = [
        _grid_candidate("bm25-3", page_number=3, source="bm25", rank=1),
        _grid_candidate("bm25-4", page_number=4, source="bm25", rank=2),
    ]

    semantic_only = module.weighted_reciprocal_rank_fusion(
        semantic,
        bm25,
        alpha_semantic=1.0,
        rrf_k=20,
    )
    bm25_only = module.weighted_reciprocal_rank_fusion(
        semantic,
        bm25,
        alpha_semantic=0.0,
        rrf_k=20,
    )

    assert [candidate.chunk.page_number for candidate in semantic_only] == [1, 2]
    assert [candidate.chunk.page_number for candidate in bm25_only] == [3, 4]


def test_page_text_hybrid_grid_rejects_non_page_text_candidates() -> None:
    module = _load_page_text_grid_module()

    with pytest.raises(ValueError, match="only accepts page_text"):
        module.weighted_reciprocal_rank_fusion(
            [_grid_candidate("semantic-1", page_number=1, source="semantic", rank=1)],
            [
                RetrievalCandidate(
                    chunk=_retrieved_chunk(
                        chunk_id="table-row",
                        page_number=2,
                        chunk_kind="table_row",
                    ),
                    score=1.0,
                    rank_sources=(RankSource(source="bm25", rank=1, score=1.0),),
                )
            ],
            alpha_semantic=0.5,
            rrf_k=20,
        )


def test_page_text_hybrid_grid_slices_cached_candidates_and_preserves_gold() -> None:
    module = _load_page_text_grid_module()
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]
    original_expected_pages = case.expected_pages
    max_candidates = [
        {
            "case": case,
            "baseline": {"status": "failed", "page_hit_at_k": False},
            "semantic_candidates": [
                _grid_candidate("s1", page_number=99, source="semantic", rank=1),
                _grid_candidate("s2", page_number=98, source="semantic", rank=2),
                _grid_candidate("s3", page_number=30, source="semantic", rank=3),
            ],
            "bm25_candidates": [
                _grid_candidate("b1", page_number=97, source="bm25", rank=1),
            ],
        }
    ]

    miss = module.evaluate_grid_config(
        config=module.GridConfig(
            semantic_top_n=2,
            bm25_top_n=1,
            rrf_k=20,
            alpha_semantic=0.75,
        ),
        max_candidates=max_candidates,
        final_top_k=6,
    )
    hit = module.evaluate_grid_config(
        config=module.GridConfig(
            semantic_top_n=3,
            bm25_top_n=1,
            rrf_k=20,
            alpha_semantic=0.75,
        ),
        max_candidates=max_candidates,
        final_top_k=6,
    )

    assert case.expected_pages == original_expected_pages
    assert miss["page_hit_count"] == 0
    assert hit["page_hit_count"] == 1


def test_page_text_hybrid_grid_retrieves_max_candidates_once_and_reuses_in_memory() -> None:
    module = _load_page_text_grid_module()
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]
    retriever = _CountingPageTextRetriever()
    bm25 = _CountingBm25PageTextIndex()

    max_candidates = module._retrieve_max_candidates(
        cases=[case],
        document_contexts={
            case.document_key: {
                "document_id": "doc",
                "bm25_page_text": bm25,
            }
        },
        retriever=retriever,
        baseline_by_case_id={},
        semantic_top_n=15,
        bm25_top_n=15,
    )
    module.evaluate_grid_config(
        config=module.GridConfig(semantic_top_n=5, bm25_top_n=5, rrf_k=20, alpha_semantic=0.5),
        max_candidates=max_candidates,
        final_top_k=6,
    )
    module.evaluate_grid_config(
        config=module.GridConfig(semantic_top_n=15, bm25_top_n=15, rrf_k=60, alpha_semantic=0.25),
        max_candidates=max_candidates,
        final_top_k=6,
    )

    assert retriever.calls == 1
    assert retriever.call_args == [
        {
            "document_id": "doc",
            "question": case.query,
            "top_k": 15,
            "chunk_kind": "page_text",
        }
    ]
    assert bm25.calls == 1
    assert bm25.top_ks == [15]


def test_page_text_hybrid_grid_summary_tracks_best_regression_and_rescue() -> None:
    module = _load_page_text_grid_module()
    baseline_passed = _page_text_grid_case_result(
        case_id="doc::PASS-01",
        query_id="PASS-01",
        baseline_status="passed",
        baseline_page_hit=True,
        hit=False,
    )
    baseline_miss = _page_text_grid_case_result(
        case_id="doc::MISS-01",
        query_id="MISS-01",
        baseline_status="failed",
        baseline_page_hit=False,
        hit=True,
    )
    support_only = _page_text_grid_case_result(
        case_id="doc::U-04",
        query_id="U-04",
        baseline_status="failed",
        baseline_page_hit=False,
        hit=False,
        supporting_hit=True,
    )
    evaluation = {
        "config": module.GridConfig(5, 5, 20, 0.5).to_json(),
        "page_hit_count": 1,
        "primary_page_hit_count": 1,
        "supporting_page_hit_count": 1,
        "partial_support_only_count": 1,
        "partial_support_only_query_ids": ["U-04"],
        "total_cases": 3,
        "page_hit_rate": 1 / 3,
        "baseline_passed_regression_count": 1,
        "baseline_page_miss_rescue_count": 1,
        "regressed_query_ids": ["PASS-01"],
        "rescued_query_ids": ["MISS-01"],
        "watchlist": {"INS-01": {"hit": None}, "U-02": {"hit": None}},
        "cases": [baseline_passed, baseline_miss, support_only],
    }

    summary = module.build_grid_summary(evaluations=[evaluation], final_top_k=6)
    rendered = module.render_page_text_hybrid_grid_markdown(
        {
            "manifest_path": "manifest.json",
            "baseline_report": {"path": "baseline.json"},
            "summary": summary,
            "best_cases": [baseline_passed, baseline_miss, support_only],
        }
    )

    assert summary["best_overall"]["baseline_passed_regression_count"] == 1
    assert summary["best_overall"]["baseline_page_miss_rescue_count"] == 1
    assert summary["best_overall"]["supporting_page_hit_count"] == 1
    assert summary["best_overall"]["partial_support_only_query_ids"] == ["U-04"]
    assert "best_overall" in rendered
    assert "partial_support_only" in rendered
    assert "PASS-01" in rendered
    assert "MISS-01" in rendered


def test_page_text_hybrid_guard_preserves_expected_pages_and_rejects_other_kinds() -> None:
    module = _load_page_text_guard_module()
    manifest = load_smoke_v2_manifest_from_payload(_manifest_payload(), base_dir=_base_dir())
    case = manifest.queries[0]
    original_expected_pages = case.expected_pages

    with pytest.raises(ValueError, match="only accepts page_text"):
        module.build_guard_candidates(
            variant=module.GuardVariant(id="bad", family="bad", description="bad"),
            case=case,
            baseline={},
            semantic_candidates=[
                _grid_candidate("semantic-page", page_number=30, source="semantic", rank=1)
            ],
            bm25_candidates=[
                RetrievalCandidate(
                    chunk=_retrieved_chunk(
                        chunk_id="table-row",
                        page_number=30,
                        chunk_kind="table_row",
                    ),
                    score=1.0,
                    rank_sources=(RankSource(source="bm25", rank=1, score=1.0),),
                )
            ],
            final_top_k=6,
        )

    assert case.expected_pages == original_expected_pages


def test_page_text_hybrid_guard_semantic_floor_retains_semantic_page() -> None:
    module = _load_page_text_guard_module()
    semantic = [_grid_candidate("semantic-hit", page_number=30, source="semantic", rank=1)]
    ranked = [
        _grid_candidate(f"bm25-only-{index}", page_number=90 + index, source="bm25", rank=index)
        for index in range(1, 7)
    ]

    guarded = module.apply_semantic_floor(
        ranked_candidates=ranked,
        semantic_candidates=semantic,
        floor_count=1,
        final_top_k=6,
    )

    assert guarded[0].chunk.page_number == 30
    assert len(guarded) == 6


def test_page_text_hybrid_guard_bm25_only_cap_limits_non_semantic_pages() -> None:
    module = _load_page_text_guard_module()
    semantic = [_grid_candidate("semantic-hit", page_number=30, source="semantic", rank=1)]
    ranked = [
        _grid_candidate("bm25-only-1", page_number=91, source="bm25", rank=1),
        _grid_candidate("bm25-only-2", page_number=92, source="bm25", rank=2),
        _grid_candidate("semantic-copy", page_number=30, source="semantic", rank=1),
        _grid_candidate("bm25-only-3", page_number=93, source="bm25", rank=3),
    ]

    guarded = module.apply_bm25_only_cap(
        ranked_candidates=ranked,
        semantic_candidates=semantic,
        cap=1,
        final_top_k=3,
    )

    assert [candidate.chunk.page_number for candidate in guarded] == [91, 30]


def test_page_text_hybrid_guard_overlap_boost_promotes_overlap_page() -> None:
    module = _load_page_text_guard_module()
    overlap = _grid_candidate("overlap", page_number=30, source="semantic", rank=1, score=0.01)
    other = _grid_candidate("other", page_number=99, source="bm25", rank=1, score=0.011)

    boosted = module.apply_overlap_boost(
        candidates=[other, overlap],
        semantic_candidates=[overlap],
        bm25_candidates=[_grid_candidate("bm25-overlap", page_number=30, source="bm25", rank=1)],
        boost_weight=0.10,
        rrf_k=20,
    )

    assert boosted[0].chunk.page_number == 30
    assert boosted[0].rank_sources[-1].source == "overlap_boost"


def test_page_text_hybrid_guard_summary_tracks_regression_and_rescue() -> None:
    module = _load_page_text_guard_module()
    variant = module.GuardVariant(
        id="semantic_floor_top1",
        family="semantic_floor",
        description="test",
    )
    evaluation = module._build_variant_evaluation(
        variant=variant,
        cases=[
            _page_text_grid_case_result(
                case_id="doc::PASS-01",
                query_id="PASS-01",
                baseline_status="passed",
                baseline_page_hit=True,
                hit=False,
            ),
            _page_text_grid_case_result(
                case_id="doc::MISS-01",
                query_id="MISS-01",
                baseline_status="failed",
                baseline_page_hit=False,
                hit=True,
            ),
            _page_text_grid_case_result(
                case_id="doc::U-04",
                query_id="U-04",
                baseline_status="failed",
                baseline_page_hit=False,
                hit=False,
                supporting_hit=True,
            ),
        ],
    )

    summary = module.build_guard_summary(evaluations=[evaluation], final_top_k=6)

    assert summary["best_guard_variant"]["variant"]["id"] == "semantic_floor_top1"
    assert summary["best_guard_variant"]["baseline_passed_regression_count"] == 1
    assert summary["best_guard_variant"]["baseline_page_miss_rescue_count"] == 1
    assert summary["best_guard_variant"]["supporting_page_hit_count"] == 1
    assert summary["best_guard_variant"]["partial_support_only_query_ids"] == ["U-04"]


def test_page_text_hybrid_guard_markdown_includes_u02_gold_refresh_check() -> None:
    module = _load_page_text_guard_module()
    variant = module.GuardVariant(
        id="semantic_floor_top1",
        family="semantic_floor",
        description="test",
    )
    u02_case = _page_text_grid_case_result(
        case_id="doc::U-02",
        query_id="U-02",
        baseline_status="passed",
        baseline_page_hit=True,
        hit=True,
    )
    evaluation = module._build_variant_evaluation(variant=variant, cases=[u02_case])
    summary = module.build_guard_summary(evaluations=[evaluation], final_top_k=6)
    report = {
        "manifest_path": "manifest.json",
        "baseline_report": {"path": "baseline.json"},
        "summary": summary,
        "best_cases": [u02_case],
        "u02_gold_refresh_check": {
            "case_id": "doc::U-02",
            "query": "腾讯控股收入按业务分部如何构成？哪个分部最大？",
            "expected_pages": [9, 196, 194],
            "baseline": {"status": "passed", "page_hit_at_k": True},
            "semantic_top_pages": [9, 10],
            "bm25_top_pages": [196, 193],
            "variant_pages": {
                "weighted_rrf_best": [196, 193, 194, 193, 223, 167],
                "semantic_floor_top1": [9, 196, 193, 194, 223, 167],
            },
            "best_variant_id": "semantic_floor_top1",
            "expected_page_in_weighted_rrf_top6": True,
            "expected_page_in_best_guard_top6": True,
            "check_note": (
                "After the gold refresh, U-02 is hit by weighted_rrf_best; "
                "this is no longer a regression and should not be attributed to guard logic."
            ),
        },
    }

    rendered = module.render_page_text_hybrid_guard_markdown(report)

    assert "U-02 Gold Refresh Check" in rendered
    assert "U-02 Regression Diagnosis" not in rendered
    assert "no longer a regression" in rendered
    assert "guard avoids the weighted RRF regression" not in rendered
    assert "weighted_rrf_best" in rendered


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


def test_runner_default_mode_is_live_retrieval_not_dry_run(tmp_path) -> None:
    module = _load_run_smoke_v2_module()
    args = module.build_parser().parse_args(
        [
            "--use-built-in-placeholders",
            "--output",
            str(tmp_path / "smoke_v2_report.json"),
        ]
    )

    assert module._resolve_mode(args) == "live_retrieval"


def test_anchor_confirmed_manifest_builder_includes_only_human_confirmed_cases(tmp_path) -> None:
    builder = _load_manifest_builder_module()
    source_path = tmp_path / "filing.pdf"
    source_path.write_text("dummy", encoding="utf-8")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "rows": [
                    _industry_matrix_row(
                        company="中远海控",
                        query_id="SHIP-01",
                        document_key="cosco_doc",
                        local_path=str(source_path),
                        human_confirmed_pages=[15],
                        human_corrected_pages=[21],
                    ),
                    _industry_matrix_row(
                        company="中国海洋石油",
                        query_id="OIL-01",
                        document_key="cnooc_doc",
                        local_path=str(source_path),
                        human_confirmed_pages=[19],
                        human_missing_fields=["reserve_life"],
                    ),
                    _industry_matrix_row(
                        company="美团",
                        query_id="LOCAL-02",
                        document_key="meituan_doc",
                        local_path=str(source_path),
                        auto_anchor_status="needs_manual_probe",
                    ),
                    _industry_matrix_row(
                        company="比亚迪",
                        query_id="NEV-01",
                        document_key="byd_doc",
                        local_path=str(source_path),
                    ),
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = builder.build_manifest_report(matrix_path=matrix_path)
    manifest = load_smoke_v2_manifest_from_payload(report["manifest"], base_dir=tmp_path)

    assert report["summary"]["included_cases"] == 1
    assert report["summary"]["excluded_partial_field_gap"] == 1
    assert report["summary"]["excluded_no_hit_or_deferred"] == 1
    assert report["summary"]["excluded_missing_human_pages"] == 1
    assert [case.case_id for case in manifest.queries] == ["cosco_doc::SHIP-01"]
    assert manifest.queries[0].expected_pages == (15, 21)
    assert manifest.queries[0].mvp_status == "anchor_confirmed_draft"


def test_anchor_confirmed_manifest_builder_merges_multiple_matrices(tmp_path) -> None:
    builder = _load_manifest_builder_module()
    source_path = tmp_path / "filing.pdf"
    source_path.write_text("dummy", encoding="utf-8")
    industry_matrix_path = tmp_path / "industry_matrix.json"
    universal_matrix_path = tmp_path / "universal_matrix.json"
    industry_matrix_path.write_text(
        json.dumps(
            {
                "rows": [
                    _industry_matrix_row(
                        company="比亚迪",
                        query_id="NEV-01",
                        document_key="比亚迪_2025_annual_report-7906b664",
                        local_path=str(source_path),
                        human_confirmed_pages=[25],
                    )
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    universal_matrix_path.write_text(
        json.dumps(
            {
                "rows": [
                    _industry_matrix_row(
                        company="招商银行",
                        query_id="U-01",
                        document_key="cmb_doc",
                        local_path=str(source_path),
                        human_confirmed_pages=[8],
                    )
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = builder.build_manifest_report(
        matrix_paths=[industry_matrix_path, universal_matrix_path]
    )

    assert report["summary"]["included_cases"] == 2
    assert report["source_files"]["matrices"] == [
        builder._display_path(industry_matrix_path),
        builder._display_path(universal_matrix_path),
    ]
    assert report["manifest"]["metadata"]["source_matrix"] is None
    assert report["manifest"]["metadata"]["source_matrices"] == [
        builder._display_path(industry_matrix_path),
        builder._display_path(universal_matrix_path),
    ]
    assert [case["query_id"] for case in report["included_cases"]] == ["NEV-01", "U-01"]
    byd = next(case for case in report["manifest"]["queries"] if case["company"] == "比亚迪")
    assert byd["document_key"] == "比亚迪_2025_annual_report-7906b664"
    assert byd["expected_pages"] == [25]


def test_anchor_confirmed_manifest_builder_rejects_duplicate_case_id(tmp_path) -> None:
    builder = _load_manifest_builder_module()
    source_path = tmp_path / "filing.pdf"
    source_path.write_text("dummy", encoding="utf-8")
    first_matrix_path = tmp_path / "first.json"
    second_matrix_path = tmp_path / "second.json"
    row = _industry_matrix_row(
        company="招商银行",
        query_id="U-01",
        document_key="cmb_doc",
        local_path=str(source_path),
        human_confirmed_pages=[8],
    )
    first_matrix_path.write_text(json.dumps({"rows": [row]}, ensure_ascii=False), encoding="utf-8")
    second_matrix_path.write_text(json.dumps({"rows": [row]}, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(builder.SmokeManifestBuildError, match="Duplicate case_id"):
        builder.build_manifest_report(matrix_paths=[first_matrix_path, second_matrix_path])


def test_anchor_confirmed_manifest_expected_pages_do_not_use_candidates(tmp_path) -> None:
    builder = _load_manifest_builder_module()
    source_path = tmp_path / "filing.pdf"
    source_path.write_text("dummy", encoding="utf-8")
    matrix_path = tmp_path / "matrix.json"
    row = _industry_matrix_row(
        company="海尔智家",
        query_id="HA-03",
        document_key="haier_doc",
        local_path=str(source_path),
        candidate_pages=[25, 31, 32],
        human_confirmed_pages=[31],
        human_corrected_pages=[32, 31],
        human_supporting_pages=[81],
    )
    row["codex_anchor_pages"] = [25, 4, 31, 39, 24]
    row["codex_suggested_gold_pages"] = [25, 4, 99]
    matrix_path.write_text(json.dumps({"rows": [row]}, ensure_ascii=False), encoding="utf-8")

    report = builder.build_manifest_report(matrix_path=matrix_path)
    query = report["manifest"]["queries"][0]

    assert query["expected_pages"] == [31, 32]
    assert query["supporting_pages"] == [81]
    assert 25 not in query["expected_pages"]
    assert 81 not in query["expected_pages"]
    assert 99 not in query["expected_pages"]
    assert report["included_cases"][0]["expected_pages_source"] == (
        "human_confirmed_pages+human_corrected_pages"
    )


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
    return _load_run_smoke_v2_module().main


def _load_run_smoke_v2_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_smoke_v2_eval.py"
    spec = importlib.util.spec_from_file_location("run_smoke_v2_eval", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest_builder_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_golden_queries_v2_smoke_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("build_golden_queries_v2_smoke_manifest", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_page_text_shadow_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_smoke_v2_page_text_rrf_shadow.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_smoke_v2_page_text_rrf_shadow",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_page_text_grid_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_smoke_v2_page_text_hybrid_grid.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_smoke_v2_page_text_hybrid_grid",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_page_text_guard_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_smoke_v2_page_text_hybrid_guard.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_smoke_v2_page_text_hybrid_guard",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _industry_matrix_row(
    *,
    company: str,
    query_id: str,
    document_key: str,
    local_path: str,
    candidate_pages: list[int] | None = None,
    human_confirmed_pages: list[int] | None = None,
    human_corrected_pages: list[int] | None = None,
    human_supporting_pages: list[int] | None = None,
    human_missing_fields: list[str] | None = None,
    auto_anchor_status: str = "auto_anchor_high_confidence",
) -> dict:
    return {
        "case_id": f"{document_key}::{query_id}",
        "query_id": query_id,
        "tier": "smoke_v2",
        "company": company,
        "industry": "test_industry",
        "document_key": document_key,
        "local_path": local_path,
        "query": "测试问题",
        "expected_route": "document_only",
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence_kind": "table_row",
        "secondary_evidence_kinds": ["page_text"],
        "candidate_pages": candidate_pages or [],
        "expected_row_labels": [],
        "expected_metric_tags": [],
        "expected_section_types": [],
        "expected_answer_field_ids": ["revenue"],
        "forbidden_failure_modes": ["wrong evidence"],
        "answer_hygiene_checks": [
            "no_raw_metadata",
            "no_empty_parentheses",
            "unit_period_present",
        ],
        "manifest_readiness": "needs_anchor_confirmation",
        "auto_anchor_status": auto_anchor_status,
        "anchor_review_status": "human_reviewed"
        if human_confirmed_pages or human_corrected_pages or human_missing_fields
        else "not_reviewed",
        "human_confirmed_pages": human_confirmed_pages or [],
        "human_corrected_pages": human_corrected_pages or [],
        "human_supporting_pages": human_supporting_pages or [],
        "human_rejected_candidate_pages": [],
        "human_missing_fields": human_missing_fields or [],
        "human_review_notes": "test note",
    }


def _retrieved_chunk(
    *,
    chunk_id: str,
    text: str = "customer deposits 2025 RMB",
    page_number: int = 1,
    chunk_kind: str = "table_row",
    row_label: str | None = None,
    score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc",
        page_number=page_number,
        source_path=Path("source.pdf"),
        text=text,
        score=score,
        chunk_kind=chunk_kind,
        row_label=row_label,
        metric_tags=[],
    )


def _diagnosis_mode(*, hit: bool) -> dict:
    return {
        "hit": hit,
        "retrieved_pages": [31] if hit else [146],
        "top_results": [],
    }


def _page_text_shadow_case_result(
    *,
    case_id: str,
    query_id: str,
    baseline_status: str,
    baseline_page_hit: bool,
    mode_hits: dict[str, bool],
) -> dict:
    modes = {}
    for mode, hit in mode_hits.items():
        retrieved_pages = [30] if hit else [99]
        modes[mode] = {
            "hit": hit,
            "supporting_hit": False,
            "page_match_status": "primary_hit" if hit else "miss",
            "expected_pages": [30],
            "supporting_pages": [],
            "retrieved_pages": retrieved_pages,
            "hit_pages": [30] if hit else [],
            "supporting_hit_pages": [],
            "top_results": [
                {
                    "rank": 1,
                    "chunk_id": f"{mode}-chunk",
                    "chunk_kind": "page_text",
                    "page_number": retrieved_pages[0],
                    "row_label": None,
                    "section_title": None,
                    "section_type": None,
                    "metric_tags": [],
                    "score": 1.0,
                    "rank_sources": [{"source": "bm25", "rank": 1, "score": 1.0}],
                    "preview": f"top snippet for {query_id}",
                }
            ],
        }
    return {
        "id": case_id,
        "query_id": query_id,
        "query": f"query for {query_id}",
        "expected_pages": [30],
        "supporting_pages": [],
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence_kind": "table_row",
        "baseline": {
            "status": baseline_status,
            "page_hit_at_k": baseline_page_hit,
            "observed_pages": [99],
        },
        "modes": modes,
    }


def _grid_candidate(
    chunk_id: str,
    *,
    page_number: int,
    source: str,
    rank: int,
    score: float = 1.0,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=_retrieved_chunk(
            chunk_id=chunk_id,
            page_number=page_number,
            chunk_kind="page_text",
            score=score,
        ),
        score=score,
        rank_sources=(RankSource(source=source, rank=rank, score=score),),
    )


def _page_text_grid_case_result(
    *,
    case_id: str,
    query_id: str,
    baseline_status: str,
    baseline_page_hit: bool,
    hit: bool,
    supporting_hit: bool = False,
    page_match_status: str | None = None,
) -> dict:
    retrieved_pages = [30] if hit else [99]
    if supporting_hit and not hit:
        retrieved_pages = [81]
    page_match_status = page_match_status or (
        "primary_hit" if hit else "partial_support_only" if supporting_hit else "miss"
    )
    return {
        "id": case_id,
        "query_id": query_id,
        "company": "TestCo",
        "industry": "test_industry",
        "query": f"query for {query_id}",
        "expected_document_evidence_intent": "metric_value",
        "primary_evidence_kind": "table_row",
        "expected_pages": [30],
        "supporting_pages": [81] if supporting_hit else [],
        "baseline": {
            "status": baseline_status,
            "page_hit_at_k": baseline_page_hit,
            "observed_pages": [99],
        },
        "config_id": "s5_b5_k20_a0p5",
        "hit": hit,
        "supporting_hit": supporting_hit,
        "page_match_status": page_match_status,
        "retrieved_pages": retrieved_pages,
        "hit_pages": [30] if hit else [],
        "supporting_hit_pages": [81] if supporting_hit else [],
        "top_results": [
            {
                "rank": 1,
                "chunk_id": f"{query_id}-chunk",
                "chunk_kind": "page_text",
                "page_number": retrieved_pages[0],
                "row_label": None,
                "section_title": None,
                "section_type": None,
                "metric_tags": [],
                "score": 1.0,
                "rank_sources": [{"source": "semantic", "rank": 1, "score": 1.0}],
                "preview": f"top snippet for {query_id}",
            }
        ],
    }


class _CountingPageTextRetriever:
    def __init__(self) -> None:
        self.calls = 0
        self.call_args: list[dict[str, object]] = []

    def retrieve(
        self,
        *,
        document_id: str,
        question: str,
        top_k: int,
        chunk_kind: str,
    ) -> list[RetrievedChunk]:
        self.calls += 1
        self.call_args.append(
            {
                "document_id": document_id,
                "question": question,
                "top_k": top_k,
                "chunk_kind": chunk_kind,
            }
        )
        return [
            _retrieved_chunk(
                chunk_id=f"semantic-{index}",
                page_number=index,
                chunk_kind="page_text",
                score=1 / index,
            )
            for index in range(1, top_k + 1)
        ]


class _CountingBm25PageTextIndex:
    def __init__(self) -> None:
        self.calls = 0
        self.top_ks: list[int] = []

    def search(self, query: str, *, top_k: int) -> list[RetrievalCandidate]:
        self.calls += 1
        self.top_ks.append(top_k)
        return [
            _grid_candidate(
                f"bm25-{index}",
                page_number=index,
                source="bm25",
                rank=index,
                score=1 / index,
            )
            for index in range(1, top_k + 1)
        ]
