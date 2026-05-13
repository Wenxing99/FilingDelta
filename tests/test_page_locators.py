from __future__ import annotations

from pathlib import Path

from filingdelta.ingestion.page_locators import CandidatePageLocator, select_summary_pages
from filingdelta.ingestion.table_metrics import extract_table_headline_metrics
from filingdelta.schemas.fact_extraction import CandidatePageSelection
from filingdelta.schemas.filing import (
    FilingDocType,
    FilingDocument,
    FilingSource,
    Market,
    ParsedFiling,
    ParsedPage,
    ParserKind,
)


def test_locate_returns_candidate_page_selection() -> None:
    parsed = _parsed_filing(
        [
            (1, "Cover page"),
            (2, "Financial summary\nRevenue\n100"),
        ]
    )

    selection = CandidatePageLocator().locate(parsed)

    assert isinstance(selection, CandidatePageSelection)
    assert selection.shared_pages[:2] == [1, 2]


def test_pages_for_shared_then_field_dedupes_stably() -> None:
    selection = CandidatePageSelection(
        shared_pages=[2, 1],
        field_pages={"revenue": [1, 3, 2, 4]},
    )

    assert selection.pages_for("revenue") == [2, 1, 3, 4]


def test_all_pages_duck_typed_by_table_metrics() -> None:
    parsed = _parsed_filing(
        [
            (1, "Cover page"),
            (
                5,
                """
                Selected financial data
                RMB million
                2025
                Revenue
                100
                Profit attributable to shareholders
                20
                Total assets
                500
                Total liabilities
                200
                Return on equity
                10
                """,
            ),
        ],
        source_path=Path("dummy.pdf"),
    )
    selection = CandidatePageLocator().locate(parsed)

    result = extract_table_headline_metrics(
        source=_source(),
        parsed_filing=parsed,
        selection=selection,
    )

    assert selection.all_pages()
    assert result.has_table_signal is True
    assert result.structured.revenue.value == 100


def test_trace_debug_stays_out_of_candidate_page_selection_dump() -> None:
    parsed = _parsed_filing(
        [
            (1, "Cover page"),
            (3, "Financial summary\nRevenue was RMB 100 million in fiscal year 2025."),
        ]
    )

    trace = CandidatePageLocator().locate_with_trace(parsed)
    entry = trace.field_debug["revenue"][0]
    dumped_selection = trace.selection.model_dump()

    assert entry.page_number == 3
    assert entry.score > 0
    assert entry.matched_terms
    assert "Revenue" in entry.snippet or "revenue" in entry.snippet.lower()
    assert entry.reasons
    assert "headline_summary_context" in entry.reasons
    assert "shared_debug" not in dumped_selection
    assert "field_debug" not in dumped_selection


def test_headline_summary_context_uses_parsed_order_not_page_number_cutoff() -> None:
    parsed = _parsed_filing(
        [
            (
                101,
                "Financial summary\nRevenue was RMB 100 million in fiscal year 2025.",
            ),
            *[
                (page_number, f"Intro page {page_number}")
                for page_number in range(102, 131)
            ],
            (
                15,
                "Financial summary\nRevenue was RMB 200 million in fiscal year 2025.",
            ),
        ]
    )

    trace = CandidatePageLocator().locate_with_trace(parsed)
    entries_by_page = {
        entry.page_number: entry for entry in trace.field_debug["revenue"]
    }

    assert "headline_summary_context" in entries_by_page[101].reasons
    assert "headline_summary_context" not in entries_by_page[15].reasons


def test_scored_pages_sort_by_score_desc_then_page_number_asc() -> None:
    parsed = _parsed_filing(
        [
            (100, "Cover page"),
            (101, "Contents"),
            (10, "Revenue\n100"),
            (7, "Revenue\n90"),
            (5, "Financial highlights\nRevenue\n120"),
        ]
    )

    trace = CandidatePageLocator().locate_with_trace(parsed)
    revenue_entries = [
        entry.page_number
        for entry in trace.field_debug["revenue"]
        if entry.page_number in {5, 7, 10}
    ]

    assert revenue_entries == [5, 7, 10]


def test_field_pages_keep_front_priority_before_scored_pages() -> None:
    parsed = _parsed_filing(
        [
            (30, "Cover page"),
            (10, "Contents"),
            (20, "Financial highlights\nRevenue\n120"),
        ]
    )

    selection = CandidatePageLocator().locate(parsed)

    assert selection.field_pages["revenue"][:3] == [30, 10, 20]


def test_liabilities_only_page_is_not_assets_candidate() -> None:
    parsed = _parsed_filing(
        [
            (1, "Cover page"),
            (2, "Contents"),
            (
                9,
                """
                Operating results
                RMB million
                Annual report
                Total liabilities
                200
                """,
            ),
            (10, "Balance sheet\nTotal assets\n500"),
        ]
    )

    selection = CandidatePageLocator().locate(parsed)

    assert 10 in selection.field_pages["total_assets"]
    assert 9 not in selection.field_pages["total_assets"]
    assert 9 in selection.field_pages["total_liabilities"]


def test_headline_summary_context_keeps_early_liabilities_summary_in_budget() -> None:
    parsed = _parsed_filing(
        [
            (1, "Cover page"),
            (2, "Contents"),
            *[
                (page_number, f"Intro page {page_number}")
                for page_number in range(3, 15)
            ],
            (
                15,
                "\u7ecf\u8425\u4e1a\u7ee9\n"
                "\u5e74\u5ea6\u62a5\u544a\n"
                "\u4eba\u6c11\u5e01 \u767e\u4e07\u5143\n"
                "\u603b\u8d1f\u503a 200\n"
                "\u603b\u8d44\u4ea7 500",
            ),
            (
                271,
                "\u8d44\u4ea7\u8d1f\u503a\u8868\n"
                "\u8d1f\u503a\u5408\u8ba1 210\n"
                "\u8d44\u4ea7\u5408\u8ba1 510",
            ),
            (
                272,
                "\u8d44\u4ea7\u8d1f\u503a\u8868\n"
                "\u8d1f\u503a\u5408\u8ba1 220\n"
                "\u8d44\u4ea7\u5408\u8ba1 520",
            ),
            (
                273,
                "\u8d44\u4ea7\u8d1f\u503a\u8868\n"
                "\u8d1f\u503a\u5408\u8ba1 230\n"
                "\u8d44\u4ea7\u5408\u8ba1 530",
            ),
            (
                274,
                "\u8d44\u4ea7\u8d1f\u503a\u8868\n"
                "\u8d1f\u503a\u5408\u8ba1 240\n"
                "\u8d44\u4ea7\u5408\u8ba1 540",
            ),
            (
                275,
                "\u8d44\u4ea7\u8d1f\u503a\u8868\n"
                "\u8d1f\u503a\u5408\u8ba1 250\n"
                "\u8d44\u4ea7\u5408\u8ba1 550",
            ),
        ]
    )

    trace = CandidatePageLocator().locate_with_trace(parsed)
    selection = trace.selection
    liabilities_pages = selection.pages_for("total_liabilities")
    headline_entry = next(
        entry
        for entry in trace.field_debug["total_liabilities"]
        if entry.page_number == 15
    )

    assert 15 in liabilities_pages
    assert 271 in liabilities_pages
    assert 272 in liabilities_pages
    assert 275 not in selection.field_pages["total_liabilities"]
    assert "headline_summary_context" in headline_entry.reasons
    dumped_selection = selection.model_dump()
    assert "shared_debug" not in dumped_selection
    assert "field_debug" not in dumped_selection


def test_headline_summary_context_requires_own_field_keyword() -> None:
    parsed = _parsed_filing(
        [
            (1, "Cover page"),
            (2, "Contents"),
            (
                8,
                "\u7ecf\u8425\u4e1a\u7ee9\n"
                "\u5e74\u5ea6\u62a5\u544a\n"
                "\u4eba\u6c11\u5e01 \u767e\u4e07\u5143\n"
                "General management discussion without the target metric.",
            ),
            (
                271,
                "\u8d44\u4ea7\u8d1f\u503a\u8868\n"
                "\u8d1f\u503a\u5408\u8ba1 210\n"
                "\u8d44\u4ea7\u5408\u8ba1 510",
            ),
        ]
    )

    trace = CandidatePageLocator().locate_with_trace(parsed)

    assert 8 not in trace.selection.field_pages["total_liabilities"]
    assert all(
        entry.page_number != 8 for entry in trace.field_debug["total_liabilities"]
    )


def test_summary_helper_does_not_change_fact_field_pages_and_limits_pages() -> None:
    parsed = _parsed_filing(
        [
            (page_number, f"Page {page_number}")
            for page_number in range(1, 18)
        ]
        + [
            (30, "Financial highlights\nRevenue\n100"),
            (31, "Strategy outlook and plan"),
        ]
    )
    locator = CandidatePageLocator()
    before = locator.locate(parsed).field_pages

    summary_pages = select_summary_pages(
        parsed,
        section_keyword_groups=(("strategy outlook",),),
    )
    after = locator.locate(parsed).field_pages

    assert before == after
    assert len(summary_pages) <= 14


def test_non_contiguous_parsed_page_numbers_use_real_page_numbers() -> None:
    parsed = _parsed_filing(
        [
            (10, "Cover page"),
            (20, "Contents"),
            (30, "Financial highlights\nRevenue\n100"),
        ]
    )

    selection = CandidatePageLocator().locate(parsed)
    summary_pages = select_summary_pages(parsed)

    assert selection.shared_pages[:2] == [10, 20]
    assert selection.field_pages["revenue"][:3] == [10, 20, 30]
    assert summary_pages[:3] == [10, 20, 30]
    assert 1 not in selection.all_pages()
    assert 2 not in selection.all_pages()


def test_document_level_company_and_ticker_do_not_affect_company_name_locator() -> None:
    pages = [
        (1, "Cover page"),
        (2, "Contents"),
        (3, "Corporate information"),
        (8, "Alpha Holdings ALPHA appears only in page text."),
    ]
    alpha = _parsed_filing(pages, company_name="Alpha Holdings", ticker="ALPHA")
    beta = _parsed_filing(pages, company_name="Beta Holdings", ticker="BETA")

    alpha_trace = CandidatePageLocator().locate_with_trace(alpha)
    beta_trace = CandidatePageLocator().locate_with_trace(beta)

    assert alpha_trace.selection.field_pages["company_name"] == [1, 2, 3]
    assert beta_trace.selection.field_pages["company_name"] == [1, 2, 3]
    assert alpha_trace.selection.field_pages == beta_trace.selection.field_pages
    assert [
        (entry.page_number, entry.score, entry.matched_terms)
        for entry in alpha_trace.field_debug["company_name"]
    ] == [
        (entry.page_number, entry.score, entry.matched_terms)
        for entry in beta_trace.field_debug["company_name"]
    ]


def _source() -> FilingSource:
    return FilingSource(
        source_path=Path("dummy.pdf"),
        company_name="Example Holdings",
        market=Market.H_SHARE,
        doc_type=FilingDocType.ANNUAL_REPORT,
        fiscal_period="2025 annual report",
    )


def _parsed_filing(
    pages: list[tuple[int, str]],
    *,
    source_path: Path = Path("dummy.txt"),
    company_name: str = "Example Holdings",
    ticker: str | None = "EX",
) -> ParsedFiling:
    return ParsedFiling(
        document=FilingDocument(
            document_id="dummy",
            company_name=company_name,
            ticker=ticker,
            market=Market.H_SHARE,
            doc_type=FilingDocType.ANNUAL_REPORT,
            fiscal_period="2025 annual report",
            source_path=source_path,
            parser_kind=ParserKind.PYMUPDF,
            total_pages=len(pages),
        ),
        pages=[
            ParsedPage(
                page_number=page_number,
                text=_strip_test_text(text),
                markdown=_strip_test_text(text),
            )
            for page_number, text in pages
        ],
    )


def _strip_test_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines())
