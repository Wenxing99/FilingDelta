from __future__ import annotations

from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import ParsedFiling, SummaryItem
from filingdelta.schemas.workflow import (
    ReaderDraftResult,
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
        summary_items: list[SummaryItem] = []

        if not reader_drafts.items:
            issues.append(
                VerificationIssue(
                    scope="summary",
                    item_key="summary_output",
                    message="Reader produced no summary items.",
                    severity="review",
                )
            )
        elif len(reader_drafts.items) < 3:
            issues.append(
                VerificationIssue(
                    scope="summary",
                    item_key="summary_output",
                    message="Reader produced fewer than the target number of summary items.",
                    severity="warning",
                )
            )

        for index, draft in enumerate(reader_drafts.items, start=1):
            citation = build_citation_from_evidence(
                parsed_filing,
                evidence_page=draft.evidence_page,
                evidence_quote=draft.evidence_quote,
            )
            citations = [citation] if citation else []
            needs_review = not citations
            if needs_review:
                issues.append(
                    VerificationIssue(
                        scope="summary",
                        item_key=f"summary_{index}",
                        message="Summary item is missing a validated citation.",
                        severity="review",
                        evidence_page=draft.evidence_page,
                        evidence_quote=draft.evidence_quote,
                    )
                )

            summary_items.append(
                SummaryItem(
                    title=draft.title,
                    summary=draft.summary,
                    citations=citations,
                    needs_human_review=needs_review,
                )
            )

        for field_name in _HEADLINE_FACT_FIELDS:
            fact_field = getattr(facts, field_name)
            if fact_field.value is None:
                continue
            if fact_field.citations:
                continue
            issues.append(
                VerificationIssue(
                    scope="facts",
                    item_key=field_name,
                    message="Fact field has a value but no validated citation.",
                    severity="review",
                )
            )

        needs_human_review = any(
            item.needs_human_review for item in summary_items
        ) or any(issue.severity == "review" for issue in issues)

        return VerificationResult(
            summary_items=summary_items,
            issues=issues,
            needs_human_review=needs_human_review,
        )
