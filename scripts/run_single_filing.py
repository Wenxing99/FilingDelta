from __future__ import annotations

import argparse
import sys
from pathlib import Path

from filingdelta.schemas.filing import FilingDocType, FilingSource, Market
from filingdelta.services.single_filing import SingleFilingProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FilingDelta parsing and fact extraction for a single filing."
    )
    parser.add_argument("source_path", type=Path, help="Path to the source filing PDF.")
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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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

    result = SingleFilingProcessor().run(source)

    print("document_id:", result.ingestion.parsed_filing.document.document_id)
    print("pages:", len(result.ingestion.parsed_filing.pages))
    print("chunks:", len(result.ingestion.chunks))
    print("parsed_output:", result.artifacts.parsed_path)
    print("facts_output:", result.artifacts.facts_path)

    print("headline_metrics:")
    print("  company_name:", result.facts.company_name.value)
    print("  fiscal_period:", result.facts.fiscal_period.value)
    print("  unit:", result.facts.unit.value)
    print("  revenue:", result.facts.revenue.value)
    print("  net_profit:", result.facts.net_profit.value)


if __name__ == "__main__":
    main()
