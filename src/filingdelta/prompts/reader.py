from __future__ import annotations

from llama_index.core.prompts import PromptTemplate


READER_SUMMARY_PROMPT = PromptTemplate(
    """You are reading a public company filing and preparing a concise, traceable summary.

Use only the supplied page excerpts. Do not invent facts or numbers.

Output rules:
- Return 3 to 4 summary items.
- Each item must have:
  - `title`: a short headline
  - `summary`: one concise sentence
  - `evidence_page`: one of the provided page numbers
  - `evidence_quote`: a short verbatim quote from the same page
- Each summary item should focus on one main point only.
- If the evidence comes from a table, copy one continuous row or one continuous clause.
- Do not combine multiple rows, multiple clauses, or multiple unrelated metrics into one `evidence_quote`.
- Prefer items that help a human quickly understand the filing:
  - company performance
  - major business changes
  - risk / outlook
  - management discussion or result highlights
- Avoid repeating the exact same point in multiple items.
- If evidence is weak, drop the item instead of guessing.

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
