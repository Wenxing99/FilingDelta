from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from filingdelta.schemas.filing import FilingDocType, FilingSource, Market
from filingdelta.workflows.single_filing import SingleFilingWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FilingDelta single-document workflow for a filing."
    )
    parser.add_argument("source_path", type=Path, help="Path to the source filing.")
    parser.add_argument("--company-name", required=True, help="Company name for this filing.")
    parser.add_argument("--ticker", default=None, help="Ticker symbol if available.")
    parser.add_argument(
        "--market",
        default=Market.OTHER.value,
        choices=[market.value for market in Market],
        help="Listing market for this filing.",
    )
    parser.add_argument(
        "--doc-type",
        default=FilingDocType.OTHER.value,
        choices=[doc_type.value for doc_type in FilingDocType],
        help="Document type for this filing.",
    )
    parser.add_argument(
        "--fiscal-period",
        default=None,
        help="Fiscal period label, for example 2025 annual report.",
    )
    parser.add_argument(
        "--language",
        default="zh",
        help="Document language. Defaults to zh.",
    )
    return parser


async def _run() -> None:
    args = build_parser().parse_args()

    source = FilingSource(
        source_path=args.source_path.resolve(),
        company_name=args.company_name,
        ticker=args.ticker,
        market=Market(args.market),
        doc_type=FilingDocType(args.doc_type),
        fiscal_period=args.fiscal_period,
        language=args.language,
    )

    workflow = SingleFilingWorkflow(verbose=False)
    result = await workflow.run(source=source)

    print("document_id:", result.document_id)
    print("pages:", result.total_pages)
    print("chunks:", result.chunk_count)
    print("needs_human_review:", result.needs_human_review)

    print("headline_metrics:")
    print("  company_name:", result.headline_metrics.company_name.value)
    print("  fiscal_period:", result.headline_metrics.fiscal_period.value)
    print("  unit:", result.headline_metrics.unit.value)
    print("  revenue:", result.headline_metrics.revenue.value)
    print("  net_profit:", result.headline_metrics.net_profit.value)

    print("overview:")
    if result.overview:
        print(f"  text: {result.overview.summary}")
        if result.overview.citations:
            citation = result.overview.citations[0]
            print(f"  citation: page={citation.page_number} quote={citation.quote}")
        else:
            print("  citation: none")
    else:
        print("  none")

    print("summary_sections:")
    for section in result.summary_sections:
        print(f"  [{section.title}]")
        for point in section.points:
            print(f"    - {point.text}")
            if point.citations:
                citation = point.citations[0]
                print(f"      citation: page={citation.page_number} quote={citation.quote}")
            else:
                print("      citation: none")

    print("verification_issues:", len(result.verification_issues))
    for issue in result.verification_issues:
        print(f"  - [{issue.scope}] {issue.item_key}: {issue.message}")
        if issue.evidence_page is not None:
            print(f"    evidence_page: {issue.evidence_page}")
        if issue.evidence_quote:
            print(f"    evidence_quote: {issue.evidence_quote}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    asyncio.run(_run())


if __name__ == "__main__":
    main()
