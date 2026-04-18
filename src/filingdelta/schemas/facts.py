from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from filingdelta.schemas.filing import Citation


class HeadlineMetricsExtractionSchema(BaseModel):
    company_name: str | None = Field(
        default=None,
        description=(
            "Official company name as shown in the filing cover, header, or core report metadata."
        ),
    )
    fiscal_period: str | None = Field(
        default=None,
        description=(
            "The reporting period for the filing, such as 2024 annual report, 2024 H1, "
            "2024 Q3, or year ended December 31, 2024."
        ),
    )
    unit: str | None = Field(
        default=None,
        description=(
            "Unit for financial figures, such as RMB, CNY, HKD, USD, yuan, thousand RMB, "
            "million HKD, or 人民币百万元."
        ),
    )
    revenue: float | None = Field(
        default=None,
        description=(
            "Total operating revenue for the main reporting period. Extract the numeric value "
            "only, without commas or unit text."
        ),
    )
    net_profit: float | None = Field(
        default=None,
        description=(
            "Net profit attributable to shareholders, owners, or the parent company for the "
            "main reporting period. Prefer the attributable measure over a generic total net "
            "profit line. Extract the numeric value only, without commas or unit text."
        ),
    )


class ExtractedFactField(BaseModel):
    value: str | float | int | None = None
    reasoning: str | None = None
    confidence: float | None = None
    evidence_page: int | None = None
    evidence_quote: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class HeadlineMetricFacts(BaseModel):
    document_id: str
    source_path: Path
    company_name: ExtractedFactField = Field(default_factory=ExtractedFactField)
    fiscal_period: ExtractedFactField = Field(default_factory=ExtractedFactField)
    unit: ExtractedFactField = Field(default_factory=ExtractedFactField)
    revenue: ExtractedFactField = Field(default_factory=ExtractedFactField)
    net_profit: ExtractedFactField = Field(default_factory=ExtractedFactField)
