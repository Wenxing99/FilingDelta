from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from filingdelta.schemas.chat import ChatRouteDecision


RouterLabel = Literal["document_only", "concept_only", "mixed", "unsupported"]


@dataclass(frozen=True)
class RouterBakeoffCase:
    case_id: str
    document_name: str
    question: str
    expected_route: RouterLabel
    expected_needs_external_background: bool
    expected_needs_risk_reasoning: bool
    notes: str = ""


@dataclass(frozen=True)
class RouterBakeoffResult:
    router_name: str
    case_id: str
    document_name: str
    question: str
    expected_route: RouterLabel
    expected_needs_external_background: bool
    expected_needs_risk_reasoning: bool
    actual_route: str | None
    actual_needs_external_background: bool | None
    actual_needs_risk_reasoning: bool | None
    rationale: str
    latency_ms: float | None
    succeeded: bool
    error: str | None = None

    @property
    def route_matched(self) -> bool:
        return self.actual_route == self.expected_route

    @property
    def external_background_matched(self) -> bool:
        return self.actual_needs_external_background == self.expected_needs_external_background

    @property
    def risk_reasoning_matched(self) -> bool:
        return self.actual_needs_risk_reasoning == self.expected_needs_risk_reasoning

    @property
    def fully_matched(self) -> bool:
        return (
            self.succeeded
            and self.route_matched
            and self.external_background_matched
            and self.risk_reasoning_matched
        )


ROUTER_BAKEOFF_CASES: tuple[RouterBakeoffCase, ...] = (
    RouterBakeoffCase(
        case_id="cmb-real-estate-risk",
        document_name="招商银行2025年度报告.pdf",
        question="招商银行如何管控房地产风险？",
        expected_route="document_only",
        expected_needs_external_background=False,
        expected_needs_risk_reasoning=False,
        notes="Disclosure-only risk-management question.",
    ),
    RouterBakeoffCase(
        case_id="cmb-customer-deposits",
        document_name="招商银行2025年度报告.pdf",
        question="招商银行客户存款有什么变化？",
        expected_route="document_only",
        expected_needs_external_background=False,
        expected_needs_risk_reasoning=False,
        notes="Document-only balance sheet / deposit trend question.",
    ),
    RouterBakeoffCase(
        case_id="cmb-asset-quality",
        document_name="招商银行2025年度报告.pdf",
        question="招商银行资产质量有哪些主要变化？",
        expected_route="document_only",
        expected_needs_external_background=False,
        expected_needs_risk_reasoning=False,
        notes="Document-only asset-quality disclosure question.",
    ),
    RouterBakeoffCase(
        case_id="tcehy-ai-ads",
        document_name="腾讯控股2025年度报告.pdf",
        question="腾讯如何描述 AI 广告能力？",
        expected_route="document_only",
        expected_needs_external_background=False,
        expected_needs_risk_reasoning=False,
        notes="Document-only AI / advertising disclosure question.",
    ),
    RouterBakeoffCase(
        case_id="tcehy-capex",
        document_name="腾讯控股2025年度报告.pdf",
        question="腾讯2025年资本开支是多少？",
        expected_route="document_only",
        expected_needs_external_background=False,
        expected_needs_risk_reasoning=False,
        notes="Metric-heavy filing fact question.",
    ),
    RouterBakeoffCase(
        case_id="roe-concept",
        document_name="招商银行2025年度报告.pdf",
        question="什么是净资产收益率？",
        expected_route="concept_only",
        expected_needs_external_background=True,
        expected_needs_risk_reasoning=False,
        notes="Concept-only definition question.",
    ),
    RouterBakeoffCase(
        case_id="cmb-roe-mixed",
        document_name="招商银行2025年度报告.pdf",
        question="什么是净资产收益率？结合当前文档里的披露解释它说明什么。",
        expected_route="mixed",
        expected_needs_external_background=True,
        expected_needs_risk_reasoning=True,
        notes="Concept plus filing facts and interpretation.",
    ),
    RouterBakeoffCase(
        case_id="tcehy-capex-meaning",
        document_name="腾讯控股2025年度报告.pdf",
        question="资本开支增加通常意味着什么？结合腾讯这份文档回答。",
        expected_route="mixed",
        expected_needs_external_background=True,
        expected_needs_risk_reasoning=True,
        notes="Usual implication plus current filing facts.",
    ),
    RouterBakeoffCase(
        case_id="cmb-real-estate-background",
        document_name="招商银行2025年度报告.pdf",
        question="房地产风险通常如何影响银行资产质量？结合招商银行文档回答。",
        expected_route="mixed",
        expected_needs_external_background=True,
        expected_needs_risk_reasoning=True,
        notes="External risk background plus current filing facts.",
    ),
    RouterBakeoffCase(
        case_id="unsupported-investment-advice",
        document_name="腾讯控股2025年度报告.pdf",
        question="根据这份年报，我现在应该买入腾讯股票吗？",
        expected_route="unsupported",
        expected_needs_external_background=False,
        expected_needs_risk_reasoning=False,
        notes="Investment advice request is out of scope.",
    ),
)


def build_router_result(
    *,
    router_name: str,
    case: RouterBakeoffCase,
    decision: ChatRouteDecision | None,
    latency_ms: float | None,
    error: Exception | None = None,
) -> RouterBakeoffResult:
    return RouterBakeoffResult(
        router_name=router_name,
        case_id=case.case_id,
        document_name=case.document_name,
        question=case.question,
        expected_route=case.expected_route,
        expected_needs_external_background=case.expected_needs_external_background,
        expected_needs_risk_reasoning=case.expected_needs_risk_reasoning,
        actual_route=decision.route if decision is not None else None,
        actual_needs_external_background=(
            decision.needs_external_background if decision is not None else None
        ),
        actual_needs_risk_reasoning=decision.needs_risk_reasoning if decision is not None else None,
        rationale=decision.rationale if decision is not None else "",
        latency_ms=latency_ms,
        succeeded=decision is not None and error is None,
        error=f"{type(error).__name__}: {error}" if error is not None else None,
    )


def summarize_router_results(results: list[RouterBakeoffResult]) -> dict[str, object]:
    by_router: dict[str, list[RouterBakeoffResult]] = defaultdict(list)
    for result in results:
        by_router[result.router_name].append(result)

    summaries: dict[str, dict[str, object]] = {}
    for router_name, router_results in by_router.items():
        succeeded = [result for result in router_results if result.succeeded]
        latencies = [
            result.latency_ms
            for result in router_results
            if isinstance(result.latency_ms, int | float)
        ]
        route_matches = [result for result in succeeded if result.route_matched]
        fully_matched = [result for result in succeeded if result.fully_matched]
        summaries[router_name] = {
            "total_cases": len(router_results),
            "succeeded_count": len(succeeded),
            "failed_count": len(router_results) - len(succeeded),
            "route_accuracy": _safe_ratio(len(route_matches), len(router_results)),
            "full_accuracy": _safe_ratio(len(fully_matched), len(router_results)),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "route_mismatch_cases": [
                result.case_id for result in succeeded if not result.route_matched
            ],
            "boolean_mismatch_cases": [
                result.case_id
                for result in succeeded
                if result.route_matched
                and not (
                    result.external_background_matched and result.risk_reasoning_matched
                )
            ],
            "failed_cases": [result.case_id for result in router_results if not result.succeeded],
        }

    return summaries


def result_to_json(result: RouterBakeoffResult) -> dict[str, object]:
    return {
        "router_name": result.router_name,
        "case_id": result.case_id,
        "document_name": result.document_name,
        "question": result.question,
        "expected": {
            "route": result.expected_route,
            "needs_external_background": result.expected_needs_external_background,
            "needs_risk_reasoning": result.expected_needs_risk_reasoning,
        },
        "actual": {
            "route": result.actual_route,
            "needs_external_background": result.actual_needs_external_background,
            "needs_risk_reasoning": result.actual_needs_risk_reasoning,
            "rationale": result.rationale,
        },
        "matches": {
            "route": result.route_matched,
            "needs_external_background": result.external_background_matched,
            "needs_risk_reasoning": result.risk_reasoning_matched,
            "full": result.fully_matched,
        },
        "latency_ms": result.latency_ms,
        "succeeded": result.succeeded,
        "error": result.error,
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)
