from __future__ import annotations

from llama_index.llms.openai import OpenAI

from filingdelta.core.config import Settings, get_settings
from filingdelta.ingestion.page_locators import CandidatePageLocator
from filingdelta.prompts.reader import READER_SUMMARY_PROMPT
from filingdelta.schemas.filing import FilingChunk, ParsedFiling
from filingdelta.schemas.workflow import ReaderDraftResult, SummaryDraftItem


class ReaderAgent:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = OpenAI(
            model=self._settings.filingdelta_llm_model,
            temperature=0,
            api_key=self._settings.require_openai_api_key(),
            api_base=self._settings.openai_base_url,
            strict=True,
        )

    async def read(
        self,
        parsed_filing: ParsedFiling,
        chunks: list[FilingChunk],
    ) -> ReaderDraftResult:
        page_numbers = _select_summary_pages(parsed_filing)
        page_context = _build_page_context(parsed_filing, page_numbers)
        chunk_count = len(chunks)

        result = await self._llm.astructured_predict(
            ReaderDraftResult,
            READER_SUMMARY_PROMPT,
            company_name=parsed_filing.document.company_name,
            ticker=parsed_filing.document.ticker or "",
            market=parsed_filing.document.market.value,
            doc_type=parsed_filing.document.doc_type.value,
            fiscal_period=parsed_filing.document.fiscal_period or "",
            page_numbers=", ".join(str(page_number) for page_number in page_numbers) or "none",
            page_context=(
                f"[Document metadata]\n"
                f"- total_pages: {parsed_filing.document.total_pages}\n"
                f"- chunk_count: {chunk_count}\n\n"
                f"{page_context}"
            ),
        )
        result.items = _dedupe_summary_items(result.items)[:4]
        return result


def _select_summary_pages(parsed_filing: ParsedFiling) -> list[int]:
    locator = CandidatePageLocator()
    selection = locator.locate(parsed_filing)
    page_numbers = [
        page.page_number for page in parsed_filing.pages[: min(4, len(parsed_filing.pages))]
    ]
    page_numbers.extend(selection.shared_pages)
    page_numbers.extend(selection.pages_for("revenue")[:2])
    page_numbers.extend(selection.pages_for("net_profit")[:2])
    return _dedupe_preserve_order(page_numbers)


def _build_page_context(parsed_filing: ParsedFiling, page_numbers: list[int]) -> str:
    page_lookup = {page.page_number: page for page in parsed_filing.pages}
    parts: list[str] = []
    for page_number in page_numbers:
        page = page_lookup.get(page_number)
        if not page:
            continue
        page_text = (page.markdown or page.text).strip()
        if not page_text:
            continue
        parts.append(f"[Page {page_number}]\n{_truncate_text(page_text)}")
    return "\n\n".join(parts)


def _truncate_text(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _dedupe_preserve_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _dedupe_summary_items(items: list[SummaryDraftItem]) -> list[SummaryDraftItem]:
    seen: set[tuple[str, str]] = set()
    deduped: list[SummaryDraftItem] = []
    for item in items:
        title_key = _normalize_text_key(item.title)
        summary_key = _normalize_text_key(item.summary)
        dedupe_key = (title_key, summary_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(item)
    return deduped


def _normalize_text_key(text: str) -> str:
    return " ".join(text.lower().split())
