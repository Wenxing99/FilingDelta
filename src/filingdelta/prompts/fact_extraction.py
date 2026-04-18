from __future__ import annotations

from llama_index.core.prompts import PromptTemplate


HEADLINE_METRICS_EXTRACTION_PROMPT = PromptTemplate(
    """You are extracting high-confidence headline financial facts from a public filing.

Use only the supplied pages. Do not infer a value if the evidence is weak or missing.

Output rules:
- Return every field in the schema.
- If a field cannot be found, set its value to null and leave evidence fields null.
- `evidence_page` must be one of the provided page numbers.
- `evidence_quote` must be a short verbatim quote copied from the same page.
- For numeric fields, return only the numeric value without commas or unit text.
- Prefer cover pages, financial highlights, summary tables, and result tables over narrative prose when both are available.

Field guidance:
- `company_name`: the official company name shown in the filing.
- `fiscal_period`: the reporting period label used by the filing.
- `unit`: the unit for the headline financial figures, such as RMB million or 人民币百万元.
- `revenue`: the total operating revenue / total revenues for the main reporting period.
- `net_profit`: net profit attributable to shareholders / owners / parent company for the main reporting period.

Filing hints:
- company_name_hint: {company_name}
- ticker_hint: {ticker}
- market_hint: {market}
- doc_type_hint: {doc_type}
- fiscal_period_hint: {fiscal_period}

Candidate pages by field:
- company_name: {company_name_pages}
- fiscal_period: {fiscal_period_pages}
- unit: {unit_pages}
- revenue: {revenue_pages}
- net_profit: {net_profit_pages}

Page excerpts:
{page_context}
"""
)
