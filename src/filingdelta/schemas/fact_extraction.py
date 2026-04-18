from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TextFactEvidence(BaseModel):
    value: str | None = Field(default=None)
    evidence_page: int | None = Field(default=None)
    evidence_quote: str | None = Field(default=None)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class NumericFactEvidence(BaseModel):
    value: float | None = Field(default=None)
    evidence_page: int | None = Field(default=None)
    evidence_quote: str | None = Field(default=None)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class HeadlineMetricsStructuredExtraction(BaseModel):
    company_name: TextFactEvidence = Field(default_factory=TextFactEvidence)
    fiscal_period: TextFactEvidence = Field(default_factory=TextFactEvidence)
    unit: TextFactEvidence = Field(default_factory=TextFactEvidence)
    revenue: NumericFactEvidence = Field(default_factory=NumericFactEvidence)
    net_profit: NumericFactEvidence = Field(default_factory=NumericFactEvidence)


class CandidatePageSelection(BaseModel):
    shared_pages: list[int] = Field(default_factory=list)
    field_pages: dict[str, list[int]] = Field(default_factory=dict)

    def pages_for(self, field_name: str) -> list[int]:
        return _dedupe_preserve_order(
            [*self.shared_pages, *self.field_pages.get(field_name, [])]
        )

    def all_pages(self) -> list[int]:
        pages: list[int] = []
        for field_name in self.field_pages:
            pages.extend(self.pages_for(field_name))
        return _dedupe_preserve_order(pages or self.shared_pages)


def _dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    deduped: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
