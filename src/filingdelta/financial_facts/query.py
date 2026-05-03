from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from filingdelta.financial_facts.catalog import CANONICAL_METRICS
from filingdelta.financial_facts.schemas import FinancialFact
from filingdelta.financial_facts.store import SQLiteFinancialFactStore


FactQueryStatus = Literal["success", "partial", "unsupported", "unavailable"]


class FinancialFactTopKSummary(BaseModel):
    selected_docs: int = 0
    candidate_count: int = 0
    verified_annual_candidates: int = 0
    after_citation_filter: int = 0
    after_company_dedupe: int = 0
    excluded_non_annual_count: int = 0
    excluded_duplicate_company_count: int = 0
    returned_rows: int = 0


class FinancialFactTopKResult(BaseModel):
    status: FactQueryStatus
    metric_id: str
    fiscal_year: int
    limit: int
    facts: list[FinancialFact] = Field(default_factory=list)
    summary: FinancialFactTopKSummary = Field(default_factory=FinancialFactTopKSummary)
    notes: list[str] = Field(default_factory=list)


class FinancialFactsQueryService:
    def __init__(self, db_path: Path | str = Path("data/indexes/financial_facts.sqlite")) -> None:
        self.db_path = Path(db_path)
        self._store = SQLiteFinancialFactStore(self.db_path)

    def top_metric_by_year(
        self,
        *,
        metric_id: str,
        fiscal_year: int,
        limit: int = 3,
        document_ids: Iterable[str] | None = None,
    ) -> FinancialFactTopKResult:
        selected_document_ids = _dedupe_preserve_order(document_ids)
        if metric_id not in CANONICAL_METRICS:
            return _result(
                status="unsupported",
                metric_id=metric_id,
                fiscal_year=fiscal_year,
                limit=limit,
                selected_docs=len(selected_document_ids) if selected_document_ids is not None else 0,
                notes=[f"unsupported_metric_id={metric_id}"],
            )
        if limit <= 0:
            return _result(
                status="unsupported",
                metric_id=metric_id,
                fiscal_year=fiscal_year,
                limit=limit,
                selected_docs=len(selected_document_ids) if selected_document_ids is not None else 0,
                notes=["limit_must_be_positive"],
            )
        if not self.db_path.exists():
            return _result(
                status="unavailable",
                metric_id=metric_id,
                fiscal_year=fiscal_year,
                limit=limit,
                selected_docs=len(selected_document_ids) if selected_document_ids is not None else 0,
                notes=[f"db_missing={self.db_path}"],
            )

        try:
            candidates = self._store.list_facts(
                metric_id=metric_id,
                fiscal_year=fiscal_year,
                review_status="verified",
            )
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).casefold():
                return _result(
                    status="unavailable",
                    metric_id=metric_id,
                    fiscal_year=fiscal_year,
                    limit=limit,
                    selected_docs=len(selected_document_ids) if selected_document_ids is not None else 0,
                    notes=["financial_facts_table_missing"],
                )
            raise

        if selected_document_ids is not None:
            allowed = set(selected_document_ids)
            candidates = [fact for fact in candidates if fact.document_id in allowed]

        annual_candidates: list[FinancialFact] = []
        excluded_non_annual_count = 0
        for fact in candidates:
            if _is_annual_fiscal_period(fact.fiscal_period):
                annual_candidates.append(fact)
            else:
                excluded_non_annual_count += 1

        cited_candidates = [
            fact
            for fact in annual_candidates
            if fact.has_page_quote_citation and fact.normalized_value is not None
        ]
        deduped, excluded_duplicate_company_count = _dedupe_company_facts(cited_candidates)
        summary = FinancialFactTopKSummary(
            selected_docs=len(selected_document_ids) if selected_document_ids is not None else 0,
            candidate_count=len(candidates),
            verified_annual_candidates=len(annual_candidates),
            after_citation_filter=len(cited_candidates),
            after_company_dedupe=len(deduped),
            excluded_non_annual_count=excluded_non_annual_count,
            excluded_duplicate_company_count=excluded_duplicate_company_count,
        )

        if not deduped:
            return FinancialFactTopKResult(
                status="partial",
                metric_id=metric_id,
                fiscal_year=fiscal_year,
                limit=limit,
                summary=summary,
                notes=["no_verified_annual_cited_facts"],
            )

        comparable_unit, unit_notes = _comparable_normalized_unit(deduped)
        if comparable_unit is None:
            return FinancialFactTopKResult(
                status="unsupported",
                metric_id=metric_id,
                fiscal_year=fiscal_year,
                limit=limit,
                summary=summary,
                notes=unit_notes,
            )

        sorted_facts = sorted(
            deduped,
            key=lambda fact: float(fact.normalized_value or 0),
            reverse=True,
        )
        returned = sorted_facts[:limit]
        summary.returned_rows = len(returned)
        status: FactQueryStatus = "success" if len(returned) == limit else "partial"
        notes = [
            "TopK keeps verified annual facts with page/quote citations only.",
            f"normalized_unit={comparable_unit}",
        ]
        if status == "partial":
            notes.append("coverage_below_requested_limit")
        return FinancialFactTopKResult(
            status=status,
            metric_id=metric_id,
            fiscal_year=fiscal_year,
            limit=limit,
            facts=returned,
            summary=summary,
            notes=notes,
        )


def _result(
    *,
    status: FactQueryStatus,
    metric_id: str,
    fiscal_year: int,
    limit: int,
    selected_docs: int,
    notes: list[str],
) -> FinancialFactTopKResult:
    return FinancialFactTopKResult(
        status=status,
        metric_id=metric_id,
        fiscal_year=fiscal_year,
        limit=limit,
        summary=FinancialFactTopKSummary(selected_docs=selected_docs),
        notes=notes,
    )


def _dedupe_company_facts(facts: list[FinancialFact]) -> tuple[list[FinancialFact], int]:
    sorted_facts = sorted(
        facts,
        key=lambda fact: float(fact.normalized_value or 0),
        reverse=True,
    )
    deduped: list[FinancialFact] = []
    seen: set[str] = set()
    excluded_duplicate_company_count = 0
    for fact in sorted_facts:
        key = _company_dedupe_key(fact)
        if key in seen:
            excluded_duplicate_company_count += 1
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped, excluded_duplicate_company_count


def _comparable_normalized_unit(facts: list[FinancialFact]) -> tuple[str | None, list[str]]:
    unit_values = [(fact.normalized_unit or fact.currency or "").strip() for fact in facts]
    if any(not unit for unit in unit_values):
        return None, ["missing_normalized_unit_or_currency"]
    units = set(unit_values)
    if len(units) != 1:
        return None, [f"incomparable_normalized_units={','.join(sorted(units))}"]
    return next(iter(units)), []


def _is_annual_fiscal_period(fiscal_period: str) -> bool:
    normalized = _normalize_period_label(fiscal_period)
    if not normalized:
        return False
    if any(marker in normalized for marker in _NON_ANNUAL_PERIOD_MARKERS):
        return False
    return any(marker in normalized for marker in _ANNUAL_PERIOD_MARKERS) or bool(
        re.fullmatch(r"(?:19|20)\d{2}年?", normalized)
    )


def _company_dedupe_key(fact: FinancialFact) -> str:
    company_name = (fact.company_name or "").strip()
    if company_name:
        return f"company:{company_name.casefold()}"
    return f"document:{fact.document_id}"


def _normalize_period_label(value: str) -> str:
    return "".join(value.casefold().split()).replace("_", "-")


def _dedupe_preserve_order(items: Iterable[str] | None) -> list[str] | None:
    if items is None:
        return None
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


_ANNUAL_PERIOD_MARKERS = (
    "annualreport",
    "full-year",
    "fullyear",
    "yearended",
    "fortheyear",
    "年度报告",
    "年度報告",
    "年报",
    "年報",
    "全年",
)

_NON_ANNUAL_PERIOD_MARKERS = (
    "q1",
    "q2",
    "q3",
    "quarter",
    "quarterly",
    "interim",
    "half-year",
    "halfyear",
    "sixmonths",
    "ninemonths",
    "threemonths",
    "1-3",
    "1-6",
    "1-9",
    "季度",
    "一季",
    "三季",
    "前三季",
    "中期",
    "半年",
    "半年度",
    "1-3月",
    "1-6月",
    "1-9月",
    "1至3月",
    "1至6月",
    "1至9月",
)
