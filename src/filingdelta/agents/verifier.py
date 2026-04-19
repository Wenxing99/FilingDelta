from __future__ import annotations

import re

from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import ParsedFiling, SummaryItem, SummaryPoint, SummarySection
from filingdelta.schemas.workflow import (
    ReaderDraftResult,
    ReviewStatusSummary,
    VerificationIssue,
    VerificationResult,
)
from filingdelta.services.citation_support import build_citation_from_evidence


_HEADLINE_FACT_FIELDS = (
    "company_name",
    "fiscal_period",
    "unit",
    "revenue",
    "net_profit",
)


class VerifierAgent:
    def verify(
        self,
        *,
        parsed_filing: ParsedFiling,
        reader_drafts: ReaderDraftResult,
        facts: HeadlineMetricFacts,
    ) -> VerificationResult:
        issues: list[VerificationIssue] = []
        overview_item: SummaryItem | None = None
        summary_sections: list[SummarySection] = []
        summary_items: list[SummaryItem] = []
        verified_count = 0
        pending_confirmation_count = 0

        if reader_drafts.overview:
            overview_item, overview_issue = _build_summary_item(
                parsed_filing,
                draft=reader_drafts.overview,
                item_key="overview",
                item_label="Overview",
                title="Overview",
                summary=reader_drafts.overview.text,
            )
            if overview_issue:
                issues.append(overview_issue)
                pending_confirmation_count += 1
            else:
                verified_count += 1

        if not reader_drafts.sections:
            issues.append(
                VerificationIssue(
                    scope="summary",
                    item_key="summary_output",
                    item_label="摘要生成结果",
                    message="Reader produced no summary sections.",
                    severity="review",
                    review_reason="summary_incomplete",
                    user_visible_reason="摘要待补充",
                )
            )
        elif sum(len(section.points) for section in reader_drafts.sections) < 6:
            issues.append(
                VerificationIssue(
                    scope="summary",
                    item_key="summary_output",
                    item_label="摘要生成结果",
                    message="Reader produced fewer than the target number of summary points.",
                    severity="warning",
                    review_reason="summary_incomplete",
                    user_visible_reason="摘要待补充",
                )
            )

        for section_index, section in enumerate(reader_drafts.sections, start=1):
            section_id = _slugify(f"section-{section_index}-{section.title}")
            section_points: list[SummaryPoint] = []

            for point_index, draft_point in enumerate(section.points, start=1):
                point_id = _slugify(f"{section_id}-point-{point_index}")
                point_item, point_issue = _build_summary_item(
                    parsed_filing,
                    draft=draft_point,
                    item_key=point_id,
                    item_label=draft_point.text,
                    title=section.title,
                    summary=draft_point.text,
                )
                if point_issue:
                    issues.append(point_issue)
                    pending_confirmation_count += 1
                else:
                    verified_count += 1

                point = SummaryPoint(
                    point_id=point_id,
                    text=draft_point.text,
                    citations=point_item.citations,
                    verification_status="review" if point_item.needs_human_review else "verified",
                    needs_human_review=point_item.needs_human_review,
                )
                section_points.append(point)
                summary_items.append(point_item)

            summary_sections.append(
                SummarySection(
                    section_id=section_id,
                    title=section.title,
                    points=section_points,
                    needs_human_review=any(point.needs_human_review for point in section_points),
                )
            )

        for field_name in _HEADLINE_FACT_FIELDS:
            fact_field = getattr(facts, field_name)
            if fact_field.value is None:
                continue
            if fact_field.citations:
                verified_count += 1
                continue
            issues.append(
                _build_fact_issue(field_name, fact_field.evidence_page, fact_field.evidence_quote)
            )
            pending_confirmation_count += 1

        needs_human_review = any(
            item.needs_human_review for item in summary_items
        ) or (
            overview_item.needs_human_review if overview_item is not None else False
        ) or any(issue.severity == "review" for issue in issues)

        review_status = "needs_confirmation" if needs_human_review else "passed"

        return VerificationResult(
            overview=overview_item,
            summary_sections=summary_sections,
            summary_items=summary_items,
            issues=issues,
            needs_human_review=needs_human_review,
            review=ReviewStatusSummary(
                status=review_status,
                verified_count=verified_count,
                pending_confirmation_count=pending_confirmation_count,
                failed_count=0,
            ),
        )


def _build_summary_item(
    parsed_filing: ParsedFiling,
    *,
    draft,
    item_key: str,
    item_label: str,
    title: str,
    summary: str,
) -> tuple[SummaryItem, VerificationIssue | None]:
    citation = build_citation_from_evidence(
        parsed_filing,
        evidence_page=draft.evidence_page,
        evidence_quote=draft.evidence_quote,
    )
    citations = [citation] if citation else []
    needs_review = not citations
    issue: VerificationIssue | None = None

    if needs_review:
        issue = VerificationIssue(
            scope="summary",
            item_key=item_key,
            item_label=item_label,
            message="Summary point is missing a validated citation.",
            severity="review",
            review_reason="citation_pending",
            user_visible_reason="引用待确认",
            evidence_page=draft.evidence_page,
            evidence_quote=draft.evidence_quote,
        )

    return (
        SummaryItem(
            title=title,
            summary=summary,
            citations=citations,
            needs_human_review=needs_review,
        ),
        issue,
    )


def _slugify(value: str) -> str:
    normalized = value.lower()
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "summary"


def _build_fact_issue(
    field_name: str,
    evidence_page: int | None,
    evidence_quote: str | None,
) -> VerificationIssue:
    label_map = {
        "company_name": "公司名称",
        "fiscal_period": "报告期",
        "unit": "单位",
        "revenue": "营业收入",
        "net_profit": "归属于股东净利润",
    }
    numeric_fields = {"revenue", "net_profit"}
    is_numeric = field_name in numeric_fields
    return VerificationIssue(
        scope="facts",
        item_key=field_name,
        item_label=label_map.get(field_name, field_name),
        message="Fact field has a value but no validated citation.",
        severity="review",
        review_reason="numeric_pending" if is_numeric else "citation_pending",
        user_visible_reason="数字待确认" if is_numeric else "引用待确认",
        evidence_page=evidence_page,
        evidence_quote=evidence_quote,
    )
