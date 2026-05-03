from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from filingdelta.financial_facts.catalog import PeriodType


ReviewStatus = Literal["verified", "needs_review", "rejected"]


class FinancialFact(BaseModel):
    fact_id: str
    document_id: str
    company_name: str | None = None
    ticker: str | None = None
    source_path: Path
    metric_id: str
    metric_label: str
    source_metric_name: str
    period_type: PeriodType
    fiscal_period: str
    fiscal_year: int | None = None
    value: float
    unit_raw: str
    currency: str | None = None
    scale: float | None = None
    normalized_value: float | None = None
    normalized_unit: str | None = None
    evidence_page: int | None = None
    evidence_quote: str = ""
    review_status: ReviewStatus = "needs_review"
    source: str = "headline_metrics"
    notes: str | None = None

    @property
    def has_page_quote_citation(self) -> bool:
        return self.evidence_page is not None and bool(self.evidence_quote.strip())


class FinancialFactQueryResult(BaseModel):
    facts: list[FinancialFact] = Field(default_factory=list)
