from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from filingdelta.core.config import REPO_ROOT
from filingdelta.financial_facts import (
    SQLiteFinancialFactStore,
    convert_headline_metric_facts,
)
from filingdelta.schemas.facts import HeadlineMetricFacts


DEFAULT_DB_PATH = Path("data/indexes/financial_facts.sqlite")
DEFAULT_INPUT_DIR = Path("data/outputs")
DEFAULT_REPORT_PATH = Path("data/outputs/financial_fact_store_coverage.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the local SQLite financial fact store from headline metric artifacts.",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--glob", default="*.headline_metrics.json")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--top-revenue-year", type=int, default=2025)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    db_path = _resolve_repo_path(args.db)
    input_dir = _resolve_repo_path(args.input_dir)
    report_path = _resolve_repo_path(args.report)

    headline_metrics = load_headline_metrics(input_dir=input_dir, glob_pattern=args.glob)
    facts = [
        fact
        for headline_metric_facts in headline_metrics
        for fact in convert_headline_metric_facts(headline_metric_facts)
    ]

    store = SQLiteFinancialFactStore(db_path)
    store.upsert_facts(facts)
    top_revenue = store.top_revenue_by_year(fiscal_year=args.top_revenue_year, limit=3)
    top_revenue_summary = store.top_revenue_by_year_stats(
        fiscal_year=args.top_revenue_year,
        limit=3,
    )

    report = build_coverage_report(
        input_dir=input_dir,
        db_path=db_path,
        source_file_count=len(headline_metrics),
        facts=facts,
        top_revenue_year=args.top_revenue_year,
        top_revenue=top_revenue,
        top_revenue_summary=top_revenue_summary,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "financial fact store built: "
        f"{len(facts)} facts from {len(headline_metrics)} headline metric files -> {db_path}"
    )
    print(f"coverage report: {report_path}")
    return report


def load_headline_metrics(*, input_dir: Path, glob_pattern: str) -> list[HeadlineMetricFacts]:
    if not input_dir.exists():
        return []
    results: list[HeadlineMetricFacts] = []
    for path in sorted(input_dir.rglob(glob_pattern)):
        if not path.is_file():
            continue
        results.append(HeadlineMetricFacts.model_validate_json(path.read_text(encoding="utf-8")))
    return results


def build_coverage_report(
    *,
    input_dir: Path,
    db_path: Path,
    source_file_count: int,
    facts: list[Any],
    top_revenue_year: int,
    top_revenue: list[Any],
    top_revenue_summary: dict[str, int],
) -> dict[str, Any]:
    by_status = Counter(fact.review_status for fact in facts)
    by_metric = Counter(fact.metric_id for fact in facts)
    return {
        "schema_version": "financial_fact_store_coverage.v1",
        "input_dir": str(input_dir),
        "db_path": str(db_path),
        "source_file_count": source_file_count,
        "fact_count": len(facts),
        "by_review_status": dict(sorted(by_status.items())),
        "by_metric": dict(sorted(by_metric.items())),
        "top_revenue_year": top_revenue_year,
        "top_revenue_summary": top_revenue_summary,
        "top_revenue": [fact.model_dump(mode="json") for fact in top_revenue],
    }


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    main()
