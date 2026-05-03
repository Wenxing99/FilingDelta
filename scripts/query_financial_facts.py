from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from filingdelta.core.config import REPO_ROOT
from filingdelta.financial_facts import FinancialFactsQueryService


DEFAULT_DB_PATH = Path("data/indexes/financial_facts.sqlite")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only SQL queries against the financial fact store.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--metric-id", default="revenue")
    parser.add_argument("--top-revenue-year", type=int, default=2025)
    parser.add_argument("--fiscal-year", type=int)
    parser.add_argument("--limit", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    fiscal_year = args.fiscal_year or args.top_revenue_year
    service = FinancialFactsQueryService(_resolve_repo_path(args.db))
    result = service.top_metric_by_year(
        metric_id=args.metric_id,
        fiscal_year=fiscal_year,
        limit=args.limit,
    )
    payload = {
        "query": f"{fiscal_year} {args.metric_id} top {args.limit}",
        "source": "sqlite_financial_fact_store",
        "status": result.status,
        "summary": result.summary.model_dump(mode="json"),
        "notes": [
            "Top metric query keeps verified annual/full-year facts only.",
            "Rows are deduped by company_name when available, otherwise document_id.",
            *result.notes,
        ],
        "facts": [fact.model_dump(mode="json") for fact in result.facts],
    }
    _write_json_stdout(payload)
    return payload


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _write_json_stdout(payload: dict[str, Any]) -> None:
    _configure_stdout_utf8()
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _configure_stdout_utf8() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        return


if __name__ == "__main__":
    main()
