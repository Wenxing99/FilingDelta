from __future__ import annotations

from llama_index.core.prompts import PromptTemplate


READER_SUMMARY_PROMPT = PromptTemplate(
    """You are reading a public company filing and preparing a dense, traceable structured summary.

Use only the supplied page excerpts. Do not invent facts or numbers.

Output rules:
- Return:
  - an optional `overview`
  - 4 to 10 summary sections when the document supports them
- Use only section titles from this taxonomy when they are supported by the document:
{section_taxonomy}
- `overview` should be one concise sentence that helps a human quickly understand the filing at a glance.
- `overview` must also include:
  - `text`
  - `evidence_page`
  - `evidence_quote`
- Each section should contain 1 to 4 `points`.
- Each point must have:
  - `text`: one concise bullet-style sentence
  - `evidence_page`: one of the provided page numbers
  - `evidence_quote`: a short verbatim quote from the same page
- Each point should focus on one concrete fact or one clear statement only.
- If the evidence comes from a table, copy one continuous row or one continuous clause.
- Do not combine multiple rows, multiple clauses, or multiple unrelated metrics into one `evidence_quote`.
- Prefer sections and points that help a human quickly understand the filing:
  - financial performance
  - dividends
  - operating / business metrics
  - strategy / outlook
  - shareholder information
  - business review
  - asset quality / risk
  - sustainability
- Avoid repeating the exact same point across different sections.
- If evidence is weak, drop the point instead of guessing.
- If the overview would be weak or generic, return `overview = null`.
- Prioritize information density over prose. It should feel like a useful filing brief, not a narrative article.

Filing hints:
- company_name_hint: {company_name}
- ticker_hint: {ticker}
- market_hint: {market}
- doc_type_hint: {doc_type}
- fiscal_period_hint: {fiscal_period}

Candidate pages:
{page_numbers}

Page excerpts:
{page_context}
"""
)
