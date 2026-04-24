from __future__ import annotations

from filingdelta.eval.router_bakeoff import (
    ROUTER_BAKEOFF_CASES,
    build_router_result,
    result_to_json,
    summarize_router_results,
)
from filingdelta.schemas.chat import ChatRouteDecision


def test_router_bakeoff_cases_cover_core_routes() -> None:
    routes = {case.expected_route for case in ROUTER_BAKEOFF_CASES}

    assert {"document_only", "concept_only", "mixed", "unsupported"}.issubset(routes)


def test_summarize_router_results_counts_route_and_boolean_accuracy() -> None:
    case = ROUTER_BAKEOFF_CASES[0]
    result = build_router_result(
        router_name="test-router",
        case=case,
        decision=ChatRouteDecision(
            route=case.expected_route,
            needs_external_background=case.expected_needs_external_background,
            needs_risk_reasoning=case.expected_needs_risk_reasoning,
            rationale="Matched.",
        ),
        latency_ms=12.5,
    )

    summary = summarize_router_results([result])

    assert summary["test-router"]["route_accuracy"] == 1.0
    assert summary["test-router"]["full_accuracy"] == 1.0
    assert summary["test-router"]["avg_latency_ms"] == 12.5


def test_result_to_json_keeps_expected_and_actual_fields() -> None:
    case = ROUTER_BAKEOFF_CASES[0]
    result = build_router_result(
        router_name="test-router",
        case=case,
        decision=ChatRouteDecision(
            route="mixed",
            needs_external_background=True,
            needs_risk_reasoning=True,
            rationale="Bad route.",
        ),
        latency_ms=10.0,
    )

    payload = result_to_json(result)

    assert payload["expected"]["route"] == case.expected_route
    assert payload["actual"]["route"] == "mixed"
    assert payload["matches"]["full"] is False
