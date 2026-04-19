from __future__ import annotations

import asyncio
from pathlib import Path

from filingdelta.agents.reader import ReaderAgent
from filingdelta.agents.verifier import VerifierAgent
from filingdelta.core.config import Settings, get_settings
from filingdelta.ingestion.fact_extractors import get_filing_fact_extractor
from filingdelta.ingestion.pipeline import FilingIngestionPipeline
from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import Citation, FilingSource, SummaryItem
from filingdelta.schemas.workflow import (
    ReaderDraftResult,
    SingleFilingWorkflowResult,
    SummaryDraftPoint,
    SummaryDraftSection,
    VerificationIssue,
)
from filingdelta.services.citation_support import build_citation_from_evidence
from filingdelta.services.fact_citation_enrichment import enrich_headline_metric_citations


class ReviewFeedbackService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pipeline = FilingIngestionPipeline(settings=self._settings)
        self._reader = ReaderAgent(settings=self._settings)
        self._fact_extractor = get_filing_fact_extractor(settings=self._settings)
        self._verifier = VerifierAgent()

    async def approve_issue(
        self,
        *,
        result: SingleFilingWorkflowResult,
        item_key: str,
    ) -> SingleFilingWorkflowResult:
        next_result = result.model_copy(deep=True)
        issue = _find_issue(next_result, item_key)

        if issue.scope == "summary":
            _approve_summary_issue(next_result, issue)
        else:
            _approve_fact_issue(next_result, issue)

        _remove_issue(next_result, item_key)
        _refresh_summary_items(next_result)
        _recalculate_review_state(next_result)
        return next_result

    async def rerun_feedback_category(
        self,
        *,
        source: FilingSource,
        result: SingleFilingWorkflowResult,
        feedback_category: str,
    ) -> SingleFilingWorkflowResult:
        next_result = result.model_copy(deep=True)
        ingestion = await asyncio.to_thread(self._pipeline.run, source)
        parsed_filing = ingestion.parsed_filing
        chunks = ingestion.chunks

        if feedback_category == "summary":
            facts = await asyncio.to_thread(
                self._fact_extractor.extract,
                source,
                parsed_filing,
            )
            facts = await asyncio.to_thread(
                enrich_headline_metric_citations,
                parsed_filing,
                facts,
            )
            reader_drafts = await self._reader.read(parsed_filing, chunks)
            verification = self._verifier.verify(
                parsed_filing=parsed_filing,
                reader_drafts=reader_drafts,
                facts=facts,
            )
            next_result.reader_drafts = reader_drafts
            next_result.overview = verification.overview
            next_result.summary_sections = verification.summary_sections
            next_result.summary_items = verification.summary_items
            next_result.headline_metrics = facts
            next_result.verification_issues = verification.issues
        elif feedback_category == "numeric":
            previous_facts = next_result.headline_metrics.model_copy(deep=True)
            facts = await asyncio.to_thread(
                self._fact_extractor.extract,
                source,
                parsed_filing,
            )
            facts = await asyncio.to_thread(
                enrich_headline_metric_citations,
                parsed_filing,
                facts,
            )
            verification = self._verifier.verify(
                parsed_filing=parsed_filing,
                reader_drafts=_reader_drafts_from_result(next_result),
                facts=facts,
            )
            next_result.headline_metrics = facts
            next_result.verification_issues = _preserve_non_numeric_summary_and_non_fact_issues(
                next_result.verification_issues
            ) + _fact_issues(verification.issues)
            _sync_summary_review_flags_from_issues(next_result)
            changed_fields = _changed_fact_fields(previous_facts, facts)
            if changed_fields:
                _invalidate_fact_coupled_summary_points(next_result, changed_fields)
        elif feedback_category == "citation":
            reader_drafts = _reader_drafts_from_result(next_result)
            facts = next_result.headline_metrics.model_copy(deep=True)
            facts = await asyncio.to_thread(
                enrich_headline_metric_citations,
                parsed_filing,
                facts,
            )
            verification = self._verifier.verify(
                parsed_filing=parsed_filing,
                reader_drafts=reader_drafts,
                facts=facts,
            )
            next_result.reader_drafts = reader_drafts
            next_result.overview = verification.overview
            next_result.summary_sections = verification.summary_sections
            next_result.summary_items = verification.summary_items
            next_result.headline_metrics = facts
            next_result.verification_issues = _preserve_summary_numeric_issues(
                next_result.verification_issues
            ) + verification.issues
            _sync_summary_review_flags_from_issues(next_result)
        else:
            raise ValueError(f"Unsupported feedback category: {feedback_category}")

        next_result.document_id = parsed_filing.document.document_id
        next_result.source_path = parsed_filing.document.source_path
        next_result.parser_kind = parsed_filing.document.parser_kind
        next_result.total_pages = parsed_filing.document.total_pages
        next_result.chunk_count = len(chunks)
        _refresh_summary_items(next_result)
        _recalculate_review_state(next_result)
        return next_result

    async def rerun_issue(
        self,
        *,
        source: FilingSource,
        result: SingleFilingWorkflowResult,
        item_key: str,
    ) -> SingleFilingWorkflowResult:
        next_result = result.model_copy(deep=True)
        issue = _find_issue(next_result, item_key)

        ingestion = await asyncio.to_thread(self._pipeline.run, source)

        if issue.scope == "summary":
            reader_drafts = await self._reader.read(ingestion.parsed_filing, ingestion.chunks)
            next_result.reader_drafts = reader_drafts
            _rerun_summary_issue(
                next_result,
                issue,
                reader_drafts=reader_drafts,
                source_path=ingestion.parsed_filing.document.source_path,
                document_id=ingestion.parsed_filing.document.document_id,
                parsed_filing=ingestion.parsed_filing,
            )
        else:
            facts = await asyncio.to_thread(
                self._fact_extractor.extract,
                source,
                ingestion.parsed_filing,
            )
            facts = await asyncio.to_thread(
                enrich_headline_metric_citations,
                ingestion.parsed_filing,
                facts,
            )
            _rerun_fact_issue(
                next_result,
                issue,
                facts=facts,
            )

        _refresh_summary_items(next_result)
        _recalculate_review_state(next_result)
        return next_result


def _approve_summary_issue(result: SingleFilingWorkflowResult, issue: VerificationIssue) -> None:
    citation = _citation_from_issue(result, issue)
    if issue.item_key == "overview":
        if result.overview is None:
            raise KeyError("Overview is missing.")
        if citation:
            result.overview.citations = [citation]
        result.overview.needs_human_review = False
        return

    target = _find_summary_point(result, issue.item_key)
    if target is None:
        raise KeyError(f"Unknown summary point: {issue.item_key}")

    if citation:
        target.citations = [citation]
    target.needs_human_review = False
    target.verification_status = "verified"


def _approve_fact_issue(result: SingleFilingWorkflowResult, issue: VerificationIssue) -> None:
    field = getattr(result.headline_metrics, issue.item_key, None)
    if field is None:
        raise KeyError(f"Unknown fact field: {issue.item_key}")

    citation = _citation_from_issue(result, issue)
    if citation:
        field.citations = [citation]


def _rerun_summary_issue(
    result: SingleFilingWorkflowResult,
    issue: VerificationIssue,
    *,
    reader_drafts,
    source_path: Path,
    document_id: str,
    parsed_filing,
) -> None:
    if issue.item_key == "overview":
        replacement = reader_drafts.overview
        if replacement is None:
            return
        citation = build_citation_from_evidence(
            parsed_filing,
            evidence_page=replacement.evidence_page,
            evidence_quote=replacement.evidence_quote,
        )
        if citation is None or result.overview is None:
            issue.evidence_page = replacement.evidence_page
            issue.evidence_quote = replacement.evidence_quote
            return

        result.overview = SummaryItem(
            title="Overview",
            summary=replacement.text,
            citations=[citation],
            needs_human_review=False,
        )
        _remove_issue(result, issue.item_key)
        return

    section_title, current_text = _find_summary_context(result, issue.item_key)
    if section_title is None or current_text is None:
        raise KeyError(f"Unknown summary point: {issue.item_key}")

    replacement = _select_replacement_point(reader_drafts.sections, section_title, current_text)
    if replacement is None:
        return

    citation = build_citation_from_evidence(
        parsed_filing,
        evidence_page=replacement.evidence_page,
        evidence_quote=replacement.evidence_quote,
    )
    if citation is None:
        issue.evidence_page = replacement.evidence_page
        issue.evidence_quote = replacement.evidence_quote
        return

    result.reader_drafts = reader_drafts
    target = _find_summary_point(result, issue.item_key)
    if target is None:
        raise KeyError(f"Unknown summary point: {issue.item_key}")

    target.text = replacement.text
    target.citations = [citation]
    target.verification_status = "verified"
    target.needs_human_review = False
    _remove_issue(result, issue.item_key)


def _rerun_fact_issue(
    result: SingleFilingWorkflowResult,
    issue: VerificationIssue,
    *,
    facts: HeadlineMetricFacts,
) -> None:
    current_field = getattr(result.headline_metrics, issue.item_key, None)
    next_field = getattr(facts, issue.item_key, None)
    if current_field is None or next_field is None:
        raise KeyError(f"Unknown fact field: {issue.item_key}")

    current_field.value = next_field.value
    current_field.reasoning = next_field.reasoning
    current_field.confidence = next_field.confidence
    current_field.evidence_page = next_field.evidence_page
    current_field.evidence_quote = next_field.evidence_quote
    current_field.citations = next_field.citations

    if current_field.citations:
        _remove_issue(result, issue.item_key)
        return

    issue.evidence_page = current_field.evidence_page
    issue.evidence_quote = current_field.evidence_quote


def _find_issue(result: SingleFilingWorkflowResult, item_key: str) -> VerificationIssue:
    for issue in result.verification_issues:
        if issue.item_key == item_key and issue.severity == "review":
            return issue
    raise KeyError(f"Unknown pending issue: {item_key}")


def _find_summary_point(result: SingleFilingWorkflowResult, item_key: str):
    for section in result.summary_sections:
        for point in section.points:
            if point.point_id == item_key:
                return point
    return None


def _find_summary_context(result: SingleFilingWorkflowResult, item_key: str) -> tuple[str | None, str | None]:
    for section in result.summary_sections:
        for point in section.points:
            if point.point_id == item_key:
                return section.title, point.text
    return None, None


def _remove_issue(result: SingleFilingWorkflowResult, item_key: str) -> None:
    result.verification_issues = [
        issue for issue in result.verification_issues if not (issue.item_key == item_key and issue.severity == "review")
    ]


def _refresh_summary_items(result: SingleFilingWorkflowResult) -> None:
    summary_items: list[SummaryItem] = []
    for section in result.summary_sections:
        for point in section.points:
            summary_items.append(
                SummaryItem(
                    title=section.title,
                    summary=point.text,
                    citations=point.citations,
                    needs_human_review=point.needs_human_review,
                )
            )
    result.summary_items = summary_items


def _recalculate_review_state(result: SingleFilingWorkflowResult) -> None:
    verified_count = 0
    pending_confirmation_count = 0

    if result.overview:
        if result.overview.needs_human_review:
            pending_confirmation_count += 1
        else:
            verified_count += 1

    for section in result.summary_sections:
        section.needs_human_review = any(point.needs_human_review for point in section.points)
        for point in section.points:
            if point.needs_human_review:
                pending_confirmation_count += 1
            else:
                verified_count += 1

    for field_name in ("company_name", "fiscal_period", "unit", "revenue", "net_profit"):
        field = getattr(result.headline_metrics, field_name)
        if field.value is None:
            continue
        if field.citations:
            verified_count += 1
        else:
            pending_confirmation_count += 1

    result.needs_human_review = pending_confirmation_count > 0
    result.review.status = "needs_confirmation" if pending_confirmation_count > 0 else "passed"
    result.review.verified_count = verified_count
    result.review.pending_confirmation_count = pending_confirmation_count
    result.review.failed_count = 0


def _citation_from_issue(result: SingleFilingWorkflowResult, issue: VerificationIssue) -> Citation | None:
    return Citation(
        document_id=result.document_id,
        source_path=result.source_path,
        page_number=issue.evidence_page,
        quote=issue.evidence_quote or "用户已手动确认",
    )


def _select_replacement_point(
    sections,
    section_title: str,
    current_text: str,
) -> SummaryDraftPoint | None:
    same_section = next((section for section in sections if section.title == section_title), None)
    if same_section is None or not same_section.points:
        return None

    best_point: SummaryDraftPoint | None = None
    best_score = -1
    current_tokens = set(_normalize_tokens(current_text))

    for point in same_section.points:
        candidate_tokens = set(_normalize_tokens(point.text))
        overlap = len(current_tokens & candidate_tokens)
        if overlap > best_score:
            best_score = overlap
            best_point = point

    return best_point or same_section.points[0]


def _normalize_tokens(text: str) -> list[str]:
    return [token for token in "".join(text.lower().split()).replace("，", ",").split(",") if token]


def _apply_summary_fact_consistency(result: SingleFilingWorkflowResult) -> None:
    for field_name, keywords in _FACT_KEYWORDS.items():
        fact_field = getattr(result.headline_metrics, field_name)
        expected_value = _normalize_fact_value(fact_field.value)
        if expected_value is None:
            continue

        if result.overview and _text_mentions_metric(result.overview.summary, keywords):
            if _text_contains_digits(result.overview.summary) and expected_value not in _normalize_text(result.overview.summary):
                result.overview.needs_human_review = True
                _upsert_review_issue(
                    result,
                    VerificationIssue(
                        scope="summary",
                        item_key="overview",
                        item_label="Overview",
                        message="Overview may be inconsistent with refreshed fact values.",
                        severity="review",
                        review_reason="numeric_pending",
                        user_visible_reason="数字待确认",
                        evidence_page=result.overview.citations[0].page_number if result.overview.citations else None,
                        evidence_quote=result.overview.citations[0].quote if result.overview.citations else None,
                    ),
                )

        for section in result.summary_sections:
            for point in section.points:
                if not _text_mentions_metric(point.text, keywords):
                    continue
                if not _text_contains_digits(point.text):
                    continue
                if expected_value in _normalize_text(point.text):
                    continue
                point.needs_human_review = True
                point.verification_status = "review"
                _upsert_review_issue(
                    result,
                    VerificationIssue(
                        scope="summary",
                        item_key=point.point_id,
                        item_label=point.text,
                        message="Summary point may be inconsistent with refreshed fact values.",
                        severity="review",
                        review_reason="numeric_pending",
                        user_visible_reason="数字待确认",
                        evidence_page=point.citations[0].page_number if point.citations else None,
                        evidence_quote=point.citations[0].quote if point.citations else None,
                    ),
                )


def _invalidate_fact_coupled_summary_points(
    result: SingleFilingWorkflowResult,
    changed_fields: set[str],
) -> None:
    for section in result.summary_sections:
        for point in section.points:
            if not _point_depends_on_changed_facts(section.title, point.text, changed_fields):
                continue
            point.needs_human_review = True
            point.verification_status = "review"
            _upsert_review_issue(
                result,
                VerificationIssue(
                    scope="summary",
                    item_key=point.point_id,
                    item_label=point.text,
                    message="Summary point may be stale after fact refresh.",
                    severity="review",
                    review_reason="numeric_pending",
                    user_visible_reason="数字待确认",
                    evidence_page=point.citations[0].page_number if point.citations else None,
                    evidence_quote=point.citations[0].quote if point.citations else None,
                ),
            )

    if result.overview and _point_depends_on_changed_facts("overview", result.overview.summary, changed_fields):
        result.overview.needs_human_review = True
        _upsert_review_issue(
            result,
            VerificationIssue(
                scope="summary",
                item_key="overview",
                item_label="Overview",
                message="Overview may be stale after fact refresh.",
                severity="review",
                review_reason="numeric_pending",
                user_visible_reason="数字待确认",
                evidence_page=result.overview.citations[0].page_number if result.overview.citations else None,
                evidence_quote=result.overview.citations[0].quote if result.overview.citations else None,
            ),
        )


def _point_depends_on_changed_facts(section_title: str, text: str, changed_fields: set[str]) -> bool:
    for field_name in changed_fields:
        if field_name not in _FACT_KEYWORDS:
            continue
        if _text_mentions_metric(text, _FACT_KEYWORDS[field_name]):
            return True
    return False


def _changed_fact_fields(previous: HeadlineMetricFacts, current: HeadlineMetricFacts) -> set[str]:
    changed: set[str] = set()
    for field_name in ("company_name", "fiscal_period", "unit", "revenue", "net_profit"):
        previous_field = getattr(previous, field_name)
        current_field = getattr(current, field_name)
        if (
            previous_field.value != current_field.value
            or previous_field.evidence_page != current_field.evidence_page
            or previous_field.evidence_quote != current_field.evidence_quote
            or previous_field.citations != current_field.citations
        ):
            changed.add(field_name)
    return changed


def _normalize_fact_value(value: str | float | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if value.is_integer():
            raw = str(int(value))
        else:
            raw = str(value).replace(".", "")
    else:
        raw = str(value)
    return _normalize_text(raw)


def _text_mentions_metric(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = _normalize_text(text)
    return any(_normalize_text(keyword) in normalized for keyword in keywords)


def _text_contains_digits(text: str) -> bool:
    return any(character.isdigit() for character in text)


def _normalize_text(text: str) -> str:
    return "".join(text.lower().split()).replace(",", "").replace("，", "").replace(".", "")


def _upsert_review_issue(result: SingleFilingWorkflowResult, issue: VerificationIssue) -> None:
    for index, current in enumerate(result.verification_issues):
        if current.item_key == issue.item_key and current.severity == "review":
            result.verification_issues[index] = issue
            return
    result.verification_issues.append(issue)


_FACT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "revenue": ("营业收入", "收入", "revenue"),
    "net_profit": ("净利润", "归母净利润", "归属于本行股东的净利润", "profit attributable"),
    "unit": ("单位", "百万元", "百萬元", "rmb", "人民币"),
    "fiscal_period": ("报告期", "年度报告", "年报", "季度报告", "中期报告"),
}

_FACT_RELEVANT_FIELDS = {"revenue", "net_profit", "unit", "fiscal_period"}


def _preserve_non_numeric_summary_and_non_fact_issues(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    return [
        issue
        for issue in issues
        if issue.scope != "facts" and not (issue.scope == "summary" and issue.review_reason == "numeric_pending")
    ]


def _preserve_summary_numeric_issues(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    return [
        issue
        for issue in issues
        if issue.scope == "summary" and issue.review_reason == "numeric_pending"
    ]


def _sync_summary_review_flags_from_issues(result: SingleFilingWorkflowResult) -> None:
    summary_issue_keys = {
        issue.item_key
        for issue in result.verification_issues
        if issue.scope == "summary" and issue.severity == "review"
    }

    if result.overview is not None:
        result.overview.needs_human_review = "overview" in summary_issue_keys

    for section in result.summary_sections:
        for point in section.points:
            point.needs_human_review = point.point_id in summary_issue_keys
            point.verification_status = "review" if point.needs_human_review else "verified"
        section.needs_human_review = any(point.needs_human_review for point in section.points)


def _summary_issues(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    return [issue for issue in issues if issue.scope == "summary"]


def _fact_issues(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    return [issue for issue in issues if issue.scope == "facts"]


def _non_summary_issues(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    return [issue for issue in issues if issue.scope != "summary"]


def _non_fact_issues(issues: list[VerificationIssue]) -> list[VerificationIssue]:
    return [issue for issue in issues if issue.scope != "facts"]


def _reader_drafts_from_result(result: SingleFilingWorkflowResult) -> ReaderDraftResult:
    summary_issues = {
        issue.item_key: issue
        for issue in result.verification_issues
        if issue.scope == "summary" and issue.severity == "review"
    }

    overview = None
    if result.overview is not None:
        citation = result.overview.citations[0] if result.overview.citations else None
        overview_issue = summary_issues.get("overview")
        overview = SummaryDraftPoint(
            text=result.overview.summary,
            evidence_page=citation.page_number if citation else (overview_issue.evidence_page if overview_issue else None),
            evidence_quote=citation.quote if citation else (overview_issue.evidence_quote if overview_issue else None),
            confidence=1.0 if citation else None,
        )

    sections: list[SummaryDraftSection] = []
    for section in result.summary_sections:
        draft_points: list[SummaryDraftPoint] = []
        for point in section.points:
            citation = point.citations[0] if point.citations else None
            issue = summary_issues.get(point.point_id)
            draft_points.append(
                SummaryDraftPoint(
                    text=point.text,
                    evidence_page=citation.page_number if citation else (issue.evidence_page if issue else None),
                    evidence_quote=citation.quote if citation else (issue.evidence_quote if issue else None),
                    confidence=1.0 if citation else None,
                )
            )
        sections.append(SummaryDraftSection(title=section.title, points=draft_points))

    return ReaderDraftResult(overview=overview, sections=sections)
