from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from filingdelta.schemas.chat import ChatAnswer


ChatQualityRoute = Literal["document_only", "concept_only", "mixed", "unsupported"]

CMB_ANNUAL = "招商银行2025年度报告.pdf"

_EMPTY_CITATION_MARKER_RE = re.compile(r"（\s*(?:[、,，;；]\s*)*）|\(\s*(?:[,;]\s*)*\)")
_INTERNAL_MARKER_RE = re.compile(
    r"\b(?:DOC|WEB)_\d+\b|(?:\[|\()?Chunk\s+\d+(?:\]|\))?|source\s*=|score\s*=",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatQualityCase:
    case_id: str
    document_name: str
    question: str
    expected_route: ChatQualityRoute
    required_answer_terms: tuple[str, ...] = ()
    required_answer_term_groups: tuple[tuple[str, ...], ...] = ()
    forbidden_answer_terms: tuple[str, ...] = ()
    min_document_citations: int = 1
    min_external_citations: int = 0
    expected_document_pages_any: tuple[int, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ChatQualityCheck:
    check_id: str
    passed: bool
    message: str


@dataclass(frozen=True)
class ChatQualityResult:
    case_id: str
    document_name: str
    question: str
    expected_route: ChatQualityRoute
    actual_route: str | None
    succeeded: bool
    checks: tuple[ChatQualityCheck, ...] = field(default_factory=tuple)
    wall_ms: int | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.succeeded and all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[ChatQualityCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


CHAT_QUALITY_CASES: tuple[ChatQualityCase, ...] = (
    ChatQualityCase(
        case_id="cmb-real-estate-risk-quality",
        document_name=CMB_ANNUAL,
        question="招商银行如何管控房地产风险？",
        expected_route="document_only",
        required_answer_terms=("房地产", "风险", "白名单", "投贷后"),
        min_document_citations=2,
        expected_document_pages_any=(45, 47, 49, 64),
        notes="Narrative risk-management answer should stay in document evidence.",
    ),
    ChatQualityCase(
        case_id="cmb-customer-deposits-quality",
        document_name=CMB_ANNUAL,
        question="招商银行客户存款有什么变化？",
        expected_route="document_only",
        required_answer_terms=("98,361.30", "8.13", "活期"),
        required_answer_term_groups=(("50.79", "49.90", "49.40"),),
        forbidden_answer_terms=("如你需要明确只看", "请告知，我可按对应口径汇总", "16日均余额"),
        min_document_citations=2,
        expected_document_pages_any=(15, 19, 30, 47, 54, 222),
        notes=(
            "This intentionally guards the recent failure where the answer drifted to "
            "company-customer deposits instead of group-level customer deposits."
        ),
    ),
    ChatQualityCase(
        case_id="cmb-roe-mixed-quality",
        document_name=CMB_ANNUAL,
        question="什么是净资产收益率？结合当前文档里的披露解释它说明什么。",
        expected_route="mixed",
        required_answer_terms=("净资产收益率", "13.44"),
        min_document_citations=1,
        min_external_citations=1,
        expected_document_pages_any=(14, 15, 19, 249),
        notes="Mixed concept plus filing answer should keep citations clean.",
    ),
)


def evaluate_chat_quality(
    *,
    case: ChatQualityCase,
    answer: ChatAnswer | None,
    wall_ms: int | None = None,
    error: Exception | None = None,
) -> ChatQualityResult:
    if answer is None:
        return ChatQualityResult(
            case_id=case.case_id,
            document_name=case.document_name,
            question=case.question,
            expected_route=case.expected_route,
            actual_route=None,
            succeeded=False,
            checks=(),
            wall_ms=wall_ms,
            error=f"{type(error).__name__}: {error}" if error is not None else "No answer.",
        )

    text = _answer_text(answer)
    document_citations = [citation for citation in answer.citations if citation.source_type == "document"]
    external_citations = [citation for citation in answer.citations if citation.source_type == "external"]
    document_pages = [citation.page_number for citation in document_citations if citation.page_number]
    external_urls = [citation.url for citation in external_citations if citation.url]

    checks = [
        ChatQualityCheck(
            check_id="route",
            passed=answer.route == case.expected_route,
            message=f"expected {case.expected_route}, got {answer.route}",
        ),
        ChatQualityCheck(
            check_id="required_answer_terms",
            passed=_all_terms_present(text, case.required_answer_terms),
            message=_missing_terms_message(text, case.required_answer_terms),
        ),
        ChatQualityCheck(
            check_id="required_answer_term_groups",
            passed=_all_term_groups_present(text, case.required_answer_term_groups),
            message=_missing_term_groups_message(text, case.required_answer_term_groups),
        ),
        ChatQualityCheck(
            check_id="forbidden_answer_terms",
            passed=_all_terms_absent(text, case.forbidden_answer_terms),
            message=_present_terms_message(text, case.forbidden_answer_terms),
        ),
        ChatQualityCheck(
            check_id="empty_citation_markers",
            passed=_EMPTY_CITATION_MARKER_RE.search(text) is None,
            message="answer should not contain empty citation parentheses like （）",
        ),
        ChatQualityCheck(
            check_id="internal_markers",
            passed=_INTERNAL_MARKER_RE.search(text) is None,
            message="answer should not leak DOC_N, WEB_N, chunk ids, scores, or source tags",
        ),
        ChatQualityCheck(
            check_id="document_citation_count",
            passed=len(document_citations) >= case.min_document_citations,
            message=(
                f"expected at least {case.min_document_citations} document citations, "
                f"got {len(document_citations)}"
            ),
        ),
        ChatQualityCheck(
            check_id="external_citation_count",
            passed=len(external_citations) >= case.min_external_citations,
            message=(
                f"expected at least {case.min_external_citations} external citations, "
                f"got {len(external_citations)}"
            ),
        ),
        ChatQualityCheck(
            check_id="duplicate_document_pages",
            passed=len(document_pages) == len(set(document_pages)),
            message=f"document citation pages should be unique, got {document_pages}",
        ),
        ChatQualityCheck(
            check_id="duplicate_external_urls",
            passed=len(external_urls) == len(set(external_urls)),
            message=f"external citation URLs should be unique, got {external_urls}",
        ),
    ]

    if case.expected_document_pages_any:
        expected_pages = set(case.expected_document_pages_any)
        checks.append(
            ChatQualityCheck(
                check_id="expected_document_pages_any",
                passed=bool(expected_pages.intersection(document_pages)),
                message=f"expected one of pages {sorted(expected_pages)}, got {document_pages}",
            )
        )

    return ChatQualityResult(
        case_id=case.case_id,
        document_name=case.document_name,
        question=case.question,
        expected_route=case.expected_route,
        actual_route=answer.route,
        succeeded=True,
        checks=tuple(checks),
        wall_ms=wall_ms,
        error=None,
    )


def summarize_chat_quality_results(results: list[ChatQualityResult]) -> dict[str, object]:
    passed = [result for result in results if result.passed]
    failed = [result for result in results if not result.passed]
    return {
        "total_cases": len(results),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "failed_cases": [result.case_id for result in failed],
        "total_wall_ms": sum(result.wall_ms or 0 for result in results),
        "avg_wall_ms": round(
            sum(result.wall_ms or 0 for result in results) / len(results),
            2,
        )
        if results
        else 0.0,
    }


def chat_quality_result_to_json(result: ChatQualityResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "document_name": result.document_name,
        "question": result.question,
        "expected_route": result.expected_route,
        "actual_route": result.actual_route,
        "succeeded": result.succeeded,
        "passed": result.passed,
        "wall_ms": result.wall_ms,
        "error": result.error,
        "checks": [
            {
                "check_id": check.check_id,
                "passed": check.passed,
                "message": check.message,
            }
            for check in result.checks
        ],
        "failed_checks": [
            {
                "check_id": check.check_id,
                "message": check.message,
            }
            for check in result.failed_checks
        ],
    }


def _answer_text(answer: ChatAnswer) -> str:
    section_items = [
        item
        for section in answer.sections
        for item in section.items
    ]
    return "\n".join([answer.answer, *section_items])


def _all_terms_present(text: str, terms: tuple[str, ...]) -> bool:
    return not _missing_terms(text, terms)


def _all_terms_absent(text: str, terms: tuple[str, ...]) -> bool:
    return not _present_terms(text, terms)


def _all_term_groups_present(text: str, term_groups: tuple[tuple[str, ...], ...]) -> bool:
    return not _missing_term_groups(text, term_groups)


def _missing_terms_message(text: str, terms: tuple[str, ...]) -> str:
    missing = _missing_terms(text, terms)
    if not missing:
        return "all required terms are present"
    return f"missing required term(s): {', '.join(missing)}"


def _present_terms_message(text: str, terms: tuple[str, ...]) -> str:
    present = _present_terms(text, terms)
    if not present:
        return "no forbidden terms are present"
    return f"forbidden term(s) present: {', '.join(present)}"


def _missing_term_groups_message(text: str, term_groups: tuple[tuple[str, ...], ...]) -> str:
    missing = _missing_term_groups(text, term_groups)
    if not missing:
        return "all required term groups are represented"
    rendered = [" / ".join(group) for group in missing]
    return f"missing required term group(s): {', '.join(rendered)}"


def _missing_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term and not _term_present(text=text, term=term)]


def _present_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term and term in text]


def _missing_term_groups(text: str, term_groups: tuple[tuple[str, ...], ...]) -> list[tuple[str, ...]]:
    return [
        group
        for group in term_groups
        if group and not any(_term_present(text=text, term=term) for term in group)
    ]


def _term_present(*, text: str, term: str) -> bool:
    if term in text:
        return True
    if not any(char.isdigit() for char in term):
        return False
    compact_text = re.sub(r"[\s,，亿元百万元人民币%％]", "", text)
    compact_term = re.sub(r"[\s,，亿元百万元人民币%％]", "", term)
    return bool(compact_term) and compact_term in compact_text
