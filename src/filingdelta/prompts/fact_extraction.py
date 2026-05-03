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
- For numeric fields, return only the numeric value without commas, percent signs, or unit text.
- For `roe`, always store the literal percentage-point number shown in the filing. For example, if the page says `14.34%`, return `14.34`, not `0.1434`.
- Prefer cover pages, financial highlights, summary tables, and result tables over narrative prose when both are available.
- For `net_profit`, prefer the attributable measure over a generic total net profit measure.
- For `total_assets` and `total_liabilities`, use explicit balance sheet / statement of financial position / key accounting data rows only. Do not calculate liabilities from assets minus equity.
- For `roe`, prefer the annualized, excluding non-recurring items, attributable to ordinary/common shareholders, weighted average ROE measure when available. If that exact wording is unavailable, fall back to the closest explicitly disclosed weighted average ROE measure.
- Do not infer or calculate `roe`. If the supplied pages do not explicitly disclose a ROE / ROAE / return on equity measure, return null.
- Do not use ROAA / return on assets, revenue growth rates, net interest margin, or any other nearby percentage as `roe`.
- Do not infer or calculate `total_assets` or `total_liabilities`. If the supplied pages do not explicitly disclose a total assets / total liabilities row or label, return null.

Field guidance:
- `company_name`: the official company name shown in the filing.
- `fiscal_period`: the reporting period label used by the filing.
- `unit`: the unit for the headline financial figures, such as RMB million or 人民币百万元.
- `revenue`: the total operating revenue / total revenues for the main reporting period.
- `net_profit`: net profit attributable to shareholders / owners / parent company for the main reporting period.
- `total_assets`: total assets at period end. This is an end-of-period metric and must come from an explicit total assets row or label.
- `total_liabilities`: total liabilities at period end. This is an end-of-period metric and must come from an explicit total liabilities row or label.
- `roe`: return on equity / weighted average return on equity for the main reporting period. Prefer the annualized, excluding non-recurring items, attributable to ordinary/common shareholders version, and keep the literal percentage-point number from the page. Only extract a value when the page explicitly labels the metric as ROE / ROAE / return on equity / 净资产收益率.
- Do not use a generic "net profit" line if an attributable net profit line is present on the supplied pages.

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
- total_assets: {total_assets_pages}
- total_liabilities: {total_liabilities_pages}
- roe: {roe_pages}

Page excerpts:
{page_context}
"""
)
