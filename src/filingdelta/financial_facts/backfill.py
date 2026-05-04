from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from filingdelta.core.config import REPO_ROOT
from filingdelta.financial_facts import CANONICAL_METRICS, convert_headline_metric_facts
from filingdelta.financial_facts.schemas import FinancialFact
from filingdelta.ingestion.raw_registry import RawDocumentRegistry, RawRegistryEntry
from filingdelta.schemas.facts import HeadlineMetricFacts
from filingdelta.schemas.filing import FilingDocType, FilingSource, Market


DEFAULT_REGISTRY_PATH = Path("data/outputs/eval/raw_document_registry.json")
DEFAULT_DB_PATH = Path("data/indexes/financial_facts.sqlite")
DEFAULT_ARTIFACT_DIR = Path("data/outputs/financial_facts")
DEFAULT_REPORT_PATH = Path("data/outputs/eval/financial_facts_backfill_current_kb_report.json")
DEFAULT_TOPK_FISCAL_YEAR = 2025
WRAPPER_SCHEMA_VERSION = "financial_facts_backfill_v1"
REPORT_SCHEMA_VERSION = "financial_facts_backfill_current_kb_report_v1"
SELECTION_MODE_DEFAULT_V1D_ALLOWLIST = "default_v1d_allowlist"
SELECTION_MODE_EXPLICIT_ALLOWLIST = "explicit_allowlist"
SELECTION_MODE_ALL_ELIGIBLE_ANNUAL_REPORTS = "all_eligible_annual_reports"

DEFAULT_V1D_ALLOWLIST = (
    "\u62db\u5546\u94f6\u884c_2025_annual_report-b849785a",
    "\u817e\u8baf\u63a7\u80a1_2025_annual_report-d19f1834",
    "\u8d35\u5dde\u8305\u53f0_2025_annual_report-474905de",
    "\u6bd4\u4e9a\u8fea_2025_annual_report-7906b664",
    "\u4e2d\u56fd\u5e73\u5b89_2025_annual_report-860c455b",
)

ALLOWED_METRIC_IDS = frozenset(
    {
        "revenue",
        "net_profit_attributable",
        "total_assets",
        "total_liabilities",
    }
)
EXPLICIT_HEADLINE_METRIC_KEYS = frozenset(
    {
        "document_id",
        "source_path",
        "company_name",
        "fiscal_period",
        "unit",
        "revenue",
        "net_profit",
        "total_assets",
        "total_liabilities",
        "roe",
    }
)
HARD_EXCLUSION_PATTERNS = (
    ("\u6458\u8981", re.compile("\u6458\u8981", re.IGNORECASE)),
    ("summary", re.compile(r"\bsummary\b", re.IGNORECASE)),
    ("quarter", re.compile(r"\bquarter(?:ly)?\b|\bq[1-4]\b", re.IGNORECASE)),
    ("\u5b63\u5ea6", re.compile("\u5b63\u5ea6|\u4e00\u5b63|\u4e09\u5b63", re.IGNORECASE)),
    ("interim", re.compile(r"\binterim\b", re.IGNORECASE)),
    ("\u4e2d\u671f", re.compile("\u4e2d\u671f|\u534a\u5e74", re.IGNORECASE)),
    ("earnings release", re.compile(r"\bearnings\s+release\b", re.IGNORECASE)),
    ("20f", re.compile(r"\b(?:form\s*)?20[\s-]*f\b", re.IGNORECASE)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill selected annual-report financial facts from raw filings into SQLite.",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--allowlist-file", type=Path)
    parser.add_argument(
        "--use-default-v1d-allowlist",
        action="store_true",
        help="Use the fixed first-batch v1D annual-report document_key allowlist.",
    )
    parser.add_argument(
        "--select-all-eligible-annual-reports",
        action="store_true",
        help="Select every current raw registry annual report that passes source/checksum guards.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually parse/extract/convert/write facts. Omit for dry-run selection only.",
    )
    parser.add_argument(
        "--reuse-existing-artifacts",
        action="store_true",
        help="Reuse validated financial_facts_backfill wrapper artifacts instead of rebuilding.",
    )
    parser.add_argument("--top-revenue-year", type=int, default=DEFAULT_TOPK_FISCAL_YEAR)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    registry_path = _resolve_repo_path(args.registry)
    artifact_dir = _resolve_repo_path(args.artifact_dir)
    report_path = _resolve_repo_path(args.report)
    db_path = _resolve_repo_path(args.db)
    selection_mode, allowlist = _load_selection(
        allowlist_file=args.allowlist_file,
        use_default_v1d=args.use_default_v1d_allowlist,
        select_all_eligible_annual_reports=args.select_all_eligible_annual_reports,
    )

    registry = _load_registry(registry_path)
    selection = select_registry_documents(
        registry=registry,
        selection_mode=selection_mode,
        allowlist=allowlist,
    )
    report = _build_base_report(
        mode="execute" if args.execute else "dry_run",
        registry_path=registry_path,
        db_path=db_path,
        artifact_dir=artifact_dir,
        selection_mode=selection_mode,
        allowlist=allowlist,
        selection=selection,
        topk_fiscal_year=args.top_revenue_year,
    )

    if args.execute:
        execute_summary = _execute_selection(
            selection=selection,
            artifact_dir=artifact_dir,
            db_path=db_path,
            reuse_existing_artifacts=args.reuse_existing_artifacts,
            topk_fiscal_year=args.top_revenue_year,
        )
        report.update(execute_summary)
    else:
        report["execution"] = {
            "executed": False,
            "note": "dry-run only: parser/extractor/store/indexer were not instantiated",
        }

    report["db_scope_check"] = _db_scope_check(
        db_path=db_path,
        selected_document_keys=selection["selected_document_keys"],
    )
    _write_json(report_path, report)
    print(f"financial facts backfill {'executed' if args.execute else 'dry-run'} report: {report_path}")
    return report


def select_registry_documents(
    *,
    registry: RawDocumentRegistry,
    selection_mode: str,
    allowlist: Iterable[str] | None = None,
) -> dict[str, Any]:
    by_key = {entry.document_key: entry for entry in registry.documents}
    entries: list[dict[str, Any]] = []
    selected: list[RawRegistryEntry] = []
    hard_exclusion_samples: list[dict[str, Any]] = []

    allowlist_ordered = _dedupe_preserve_order(allowlist or [])
    if selection_mode == SELECTION_MODE_ALL_ELIGIBLE_ANNUAL_REPORTS:
        candidate_entries = list(registry.documents)
        missing_allowlist_count = 0
    else:
        candidate_entries = []
        missing_allowlist_count = 0
        for document_key in allowlist_ordered:
            entry = by_key.get(document_key)
            if entry is None:
                entries.append(
                    {
                        "document_key": document_key,
                        "status": "skipped",
                        "reasons": ["not_in_registry"],
                    }
                )
                missing_allowlist_count += 1
                continue
            candidate_entries.append(entry)

    for entry in candidate_entries:
        entry_report = _registry_entry_report(entry)
        entries.append(entry_report)
        if entry_report["hard_exclusion_terms"]:
            hard_exclusion_samples.append(entry_report)
        if entry_report["status"] == "selected":
            selected.append(entry)

    return {
        "selection_mode": selection_mode,
        "allowlist": allowlist_ordered,
        "entries": entries,
        "selected_entries": selected,
        "selected_document_keys": [entry.document_key for entry in selected],
        "selected_count": len(selected),
        "skipped_count": len(entries) - len(selected),
        "selected_by_fiscal_year": _count_selected_by_fiscal_year(selected),
        "registry_skipped_not_selected_count": max(
            0,
            len(registry.documents) - len(candidate_entries),
        ),
        "allowlist_missing_count": missing_allowlist_count,
        "hard_exclusion_samples": hard_exclusion_samples[:10],
    }


def _registry_entry_report(entry: RawRegistryEntry) -> dict[str, Any]:
    outcome = _evaluate_registry_entry(entry)
    return {
        "document_key": entry.document_key,
        "status": "selected" if outcome["selected"] else "skipped",
        "reasons": outcome["reasons"],
        "local_path": entry.local_path,
        "resolved_source_path": str(_resolve_repo_path(Path(entry.local_path))),
        "inferred_doc_type": entry.inferred_doc_type,
        "inferred_fiscal_year": entry.inferred_fiscal_year,
        "source_exists": outcome["source_exists"],
        "checksum_match": outcome["checksum_match"],
        "hard_exclusion_terms": outcome["hard_exclusion_terms"],
    }


def _execute_selection(
    *,
    selection: dict[str, Any],
    artifact_dir: Path,
    db_path: Path,
    reuse_existing_artifacts: bool,
    topk_fiscal_year: int,
) -> dict[str, Any]:
    from filingdelta.financial_facts import FinancialFactsQueryService, SQLiteFinancialFactStore

    store = SQLiteFinancialFactStore(db_path)
    document_reports: list[dict[str, Any]] = []
    all_written_facts: list[FinancialFact] = []
    allowed_metric_ids = set(CANONICAL_METRICS).intersection(ALLOWED_METRIC_IDS)

    for entry in selection["selected_entries"]:
        document_report: dict[str, Any] = {
            "document_key": entry.document_key,
            "status": "pending",
            "artifact_action": "none",
            "facts_written": 0,
            "pruned_fact_count": 0,
            "metrics": _empty_metric_report(),
        }
        try:
            facts = _load_or_build_headline_metrics(
                entry=entry,
                artifact_dir=artifact_dir,
                reuse_existing_artifacts=reuse_existing_artifacts,
                document_report=document_report,
            )
            converted_facts = convert_headline_metric_facts(facts)
            _assert_allowed_metric_ids(converted_facts, allowed_metric_ids=allowed_metric_ids)
            if not converted_facts:
                document_report["status"] = "no_facts_extracted"
                document_report["metrics"] = _empty_metric_report()
                document_reports.append(document_report)
                continue
            replace_result = store.replace_facts_for_document(entry.document_key, converted_facts)
            document_report["status"] = "written"
            document_report["facts_written"] = replace_result["upserted"]
            document_report["pruned_fact_count"] = replace_result["deleted"]
            document_report["metrics"] = _metric_report(converted_facts)
            all_written_facts.extend(converted_facts)
        except Exception as exc:  # pragma: no cover - covered by behavior tests with fakes
            document_report["status"] = "failed"
            document_report["error_type"] = exc.__class__.__name__
            document_report["error"] = str(exc)
        document_reports.append(document_report)

    selected_document_ids = selection["selected_document_keys"]
    written_document_count = sum(
        1 for result in document_reports if result.get("status") == "written"
    )
    query_service = FinancialFactsQueryService(db_path)
    per_metric_top3_status = _per_metric_top3_status(
        query_service=query_service,
        metric_ids=sorted(ALLOWED_METRIC_IDS),
        fiscal_year=topk_fiscal_year,
        document_ids=selected_document_ids,
    )
    revenue_result = per_metric_top3_status["revenue"]

    return {
        "execution": {
            "executed": True,
            "reuse_existing_artifacts": reuse_existing_artifacts,
            "document_results": document_reports,
            "facts_written_total": sum(result["facts_written"] for result in document_reports),
            "pruned_fact_count_total": sum(
                result["pruned_fact_count"] for result in document_reports
            ),
            "facts_written_by_metric": dict(
                sorted(Counter(fact.metric_id for fact in all_written_facts).items())
            ),
            "review_status_by_metric": _aggregate_metric_reports(document_reports),
        },
        "per_metric_top3_status": per_metric_top3_status,
        "selected_scope_revenue_top3": revenue_result["facts"],
        "selected_scope_revenue_top3_stats": revenue_result["summary"],
        "selected_scope_revenue_top3_status": _topk_status(
            status=revenue_result["status"],
            summary=revenue_result["summary"],
            notes=revenue_result["notes"],
            written_document_count=written_document_count,
        ),
        "current_kb_scope_note": (
            "Coverage is limited to current raw registry eligible annual reports and "
            "the current SQLite KB."
        ),
    }


def _load_or_build_headline_metrics(
    *,
    entry: RawRegistryEntry,
    artifact_dir: Path,
    reuse_existing_artifacts: bool,
    document_report: dict[str, Any],
) -> HeadlineMetricFacts:
    wrapper_path = _wrapper_path(artifact_dir=artifact_dir, document_key=entry.document_key)
    if reuse_existing_artifacts and wrapper_path.exists():
        reusable, reuse_failure_reason = _try_load_reusable_wrapper(
            wrapper_path=wrapper_path,
            entry=entry,
        )
        if reusable is not None:
            document_report["artifact_action"] = "reused"
            document_report["artifact_path"] = str(wrapper_path)
            return reusable
        document_report["artifact_action"] = "rebuilt"
        document_report["artifact_rebuilt_reason"] = reuse_failure_reason

    facts = _run_parse_extract(entry)
    _write_wrapper(wrapper_path=wrapper_path, entry=entry, facts=facts)
    document_report["artifact_action"] = "written"
    document_report["artifact_path"] = str(wrapper_path)
    return facts


def _run_parse_extract(entry: RawRegistryEntry) -> HeadlineMetricFacts:
    from filingdelta.ingestion.fact_extractors import get_filing_fact_extractor
    from filingdelta.ingestion.parsers import get_filing_parser
    from filingdelta.services.fact_citation_enrichment import enrich_headline_metric_citations
    from filingdelta.core.config import get_settings

    source = _filing_source_from_registry_entry(entry)
    parsed_filing = get_filing_parser(get_settings()).parse(source)
    parsed_filing.document.document_id = entry.document_key
    facts = get_filing_fact_extractor().extract(source, parsed_filing)
    facts = facts.model_copy(update={"document_id": entry.document_key})
    return enrich_headline_metric_citations(parsed_filing, facts)


def _try_load_reusable_wrapper(
    *,
    wrapper_path: Path,
    entry: RawRegistryEntry,
) -> tuple[HeadlineMetricFacts | None, str | None]:
    try:
        payload = json.loads(wrapper_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "wrapper_read_failed"
    if not _wrapper_payload_matches_entry(payload=payload, entry=entry):
        return None, "wrapper_metadata_mismatch"
    headline_metrics = payload.get("headline_metrics")
    if not isinstance(headline_metrics, dict):
        return None, "headline_metrics_missing"
    if not EXPLICIT_HEADLINE_METRIC_KEYS.issubset(headline_metrics):
        return None, "headline_metric_keys_missing"
    try:
        facts = HeadlineMetricFacts.model_validate(headline_metrics)
    except ValueError:
        return None, "headline_metrics_validation_failed"
    if facts.document_id != entry.document_key:
        return None, "headline_document_id_mismatch"
    source_path = str(facts.source_path)
    if not (
        _same_path_text(source_path, entry.local_path)
        or _same_path_text(source_path, str(_resolve_repo_path(Path(entry.local_path))))
    ):
        return None, "headline_source_path_mismatch"
    return facts, None


def _wrapper_payload_matches_entry(*, payload: dict[str, Any], entry: RawRegistryEntry) -> bool:
    return (
        payload.get("schema_version") == WRAPPER_SCHEMA_VERSION
        and payload.get("document_key") == entry.document_key
        and _same_path_text(payload.get("registry_local_path"), entry.local_path)
        and _same_path_text(
            payload.get("resolved_source_path"),
            str(_resolve_repo_path(Path(entry.local_path))),
        )
        and payload.get("checksum_sha256") == entry.checksum_sha256
    )


def _write_wrapper(
    *,
    wrapper_path: Path,
    entry: RawRegistryEntry,
    facts: HeadlineMetricFacts,
) -> None:
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": WRAPPER_SCHEMA_VERSION,
        "document_key": entry.document_key,
        "registry_local_path": entry.local_path,
        "resolved_source_path": str(_resolve_repo_path(Path(entry.local_path))),
        "checksum_sha256": entry.checksum_sha256,
        "headline_metrics": facts.model_dump(mode="json"),
    }
    _write_json(wrapper_path, payload)


def _evaluate_registry_entry(entry: RawRegistryEntry) -> dict[str, Any]:
    source_path = _resolve_repo_path(Path(entry.local_path))
    source_exists = source_path.exists() and source_path.is_file()
    checksum_match = False
    if source_exists:
        checksum_match = _sha256_file(source_path) == entry.checksum_sha256

    hard_exclusion_terms = _hard_exclusion_terms(entry)
    reasons: list[str] = []
    if entry.inferred_doc_type != FilingDocType.ANNUAL_REPORT.value:
        reasons.append("not_annual_report")
    if hard_exclusion_terms:
        reasons.append("hard_exclusion_terms")
    if not source_exists:
        reasons.append("source_missing")
    if source_exists and not checksum_match:
        reasons.append("checksum_mismatch")

    return {
        "selected": not reasons,
        "reasons": reasons or ["selected"],
        "source_exists": source_exists,
        "checksum_match": checksum_match,
        "hard_exclusion_terms": hard_exclusion_terms,
    }


def _hard_exclusion_terms(entry: RawRegistryEntry) -> list[str]:
    text = " ".join(
        value
        for value in (
            entry.document_key,
            entry.local_path,
            entry.filename,
            entry.company_id,
            entry.inferred_company_name,
            entry.ticker,
            entry.source_url,
            entry.inferred_doc_type,
            entry.notes,
        )
        if value
    )
    return [label for label, pattern in HARD_EXCLUSION_PATTERNS if pattern.search(text)]


def _filing_source_from_registry_entry(entry: RawRegistryEntry) -> FilingSource:
    return FilingSource(
        source_path=_resolve_repo_path(Path(entry.local_path)),
        company_name=entry.inferred_company_name or entry.company_id or entry.document_key,
        ticker=entry.ticker,
        market=_enum_or_default(Market, entry.inferred_market, Market.OTHER),
        doc_type=_enum_or_default(FilingDocType, entry.inferred_doc_type, FilingDocType.OTHER),
        fiscal_period=entry.fiscal_period or str(entry.inferred_fiscal_year or ""),
        language=entry.language,
    )


def _build_base_report(
    *,
    mode: str,
    registry_path: Path,
    db_path: Path,
    artifact_dir: Path,
    selection_mode: str,
    allowlist: list[str],
    selection: dict[str, Any],
    topk_fiscal_year: int,
) -> dict[str, Any]:
    selected_document_keys = selection["selected_document_keys"]
    entries = selection["entries"]
    selected_docs_for_topk_year = [
        entry.document_key
        for entry in selection["selected_entries"]
        if entry.inferred_fiscal_year == topk_fiscal_year
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": mode,
        "selection_mode": selection_mode,
        "registry_path": str(registry_path),
        "db_path": str(db_path),
        "artifact_dir": str(artifact_dir),
        "allowlist": allowlist,
        "selected_document_keys": selected_document_keys,
        "selected_count": selection["selected_count"],
        "selected_by_fiscal_year": selection["selected_by_fiscal_year"],
        "topk_fiscal_year": topk_fiscal_year,
        "selected_docs_for_topk_year": selected_docs_for_topk_year,
        "skipped_count": selection["skipped_count"],
        "skipped_reasons": dict(
            sorted(
                Counter(
                    reason
                    for entry in entries
                    if entry["status"] == "skipped"
                    for reason in entry.get("reasons", [])
                ).items()
            )
        ),
        "selection": entries,
        "hard_exclusion_samples": selection["hard_exclusion_samples"],
        "top_revenue_year": topk_fiscal_year,
        "selected_scope_revenue_top3": [],
        "selected_scope_revenue_top3_stats": {
            "selected_docs": len(selected_document_keys),
            "candidate_count": 0,
            "verified_annual_candidates": 0,
            "after_citation_filter": 0,
            "after_company_dedupe": 0,
            "returned_rows": 0,
        },
        "selected_scope_revenue_top3_status": {
            "status": "not_run",
            "reasons": ["dry_run"],
        },
        "per_metric_top3_status": {
            metric_id: {
                "status": "not_run",
                "metric_id": metric_id,
                "fiscal_year": topk_fiscal_year,
                "limit": 3,
                "facts": [],
                "summary": {
                    "selected_docs": len(selected_document_keys),
                    "candidate_count": 0,
                    "verified_annual_candidates": 0,
                    "after_citation_filter": 0,
                    "after_company_dedupe": 0,
                    "excluded_non_annual_count": 0,
                    "excluded_duplicate_company_count": 0,
                    "returned_rows": 0,
                },
                "notes": ["dry_run"],
                "reasons": ["dry_run"],
            }
            for metric_id in sorted(ALLOWED_METRIC_IDS)
        },
        "current_kb_scope_note": (
            "Coverage is limited to current raw registry eligible annual reports and "
            "the current SQLite KB."
        ),
        "db_scope_check": {
            "status": "not_run",
            "note": "db scope is checked after dry-run/execute report assembly",
        },
    }


def _metric_report(facts: list[FinancialFact]) -> dict[str, dict[str, int]]:
    by_metric_status: dict[str, Counter[str]] = defaultdict(Counter)
    for fact in facts:
        by_metric_status[fact.metric_id][fact.review_status] += 1
    report = _empty_metric_report()
    for metric_id, counter in by_metric_status.items():
        for status, count in counter.items():
            report[metric_id][status] = count
        report[metric_id]["missing"] = 0
    return report


def _empty_metric_report() -> dict[str, dict[str, int]]:
    return {
        metric_id: {"verified": 0, "needs_review": 0, "rejected": 0, "missing": 1}
        for metric_id in sorted(ALLOWED_METRIC_IDS)
    }


def _aggregate_metric_reports(
    document_reports: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    summary = {
        metric_id: {"verified": 0, "needs_review": 0, "rejected": 0, "missing": 0}
        for metric_id in sorted(ALLOWED_METRIC_IDS)
    }
    for document_report in document_reports:
        metrics = document_report.get("metrics", {})
        for metric_id, metric_summary in metrics.items():
            if metric_id not in summary:
                continue
            for status in summary[metric_id]:
                summary[metric_id][status] += int(metric_summary.get(status, 0))
    return summary


def _per_metric_top3_status(
    *,
    query_service: Any,
    metric_ids: Iterable[str],
    fiscal_year: int,
    document_ids: list[str],
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for metric_id in metric_ids:
        result = query_service.top_metric_by_year(
            metric_id=metric_id,
            fiscal_year=fiscal_year,
            limit=3,
            document_ids=document_ids,
        )
        metric_report = _serialize_topk_result(result)
        metric_report["reasons"] = _topk_reasons(
            status=metric_report["status"],
            summary=metric_report["summary"],
            notes=metric_report["notes"],
        )
        report[metric_id] = metric_report
    return report


def _serialize_topk_result(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "metric_id": result.metric_id,
        "fiscal_year": result.fiscal_year,
        "limit": result.limit,
        "facts": [fact.model_dump(mode="json") for fact in result.facts],
        "summary": result.summary.model_dump(mode="json"),
        "notes": list(result.notes),
    }


def _topk_status(
    *,
    status: str,
    summary: dict[str, int],
    notes: list[str],
    written_document_count: int,
) -> dict[str, Any]:
    reasons = _topk_reasons(status=status, summary=summary, notes=notes)
    if written_document_count == 0:
        reasons.append("no_written_documents")
    return {
        "status": status,
        "reasons": _dedupe_preserve_order(reasons),
        "notes": notes,
        "summary": summary,
    }


def _topk_reasons(
    *,
    status: str,
    summary: dict[str, int],
    notes: list[str],
) -> list[str]:
    reasons: list[str] = []
    if status != "success":
        reasons.append(f"query_status={status}")
    if summary.get("candidate_count", 0) == 0:
        reasons.append("candidate_count")
    if summary.get("verified_annual_candidates", 0) == 0:
        reasons.append("verified_annual_candidates")
    if summary.get("after_citation_filter", 0) == 0:
        reasons.append("after_citation_filter")
    if summary.get("after_company_dedupe", 0) == 0:
        reasons.append("after_company_dedupe")
    if summary.get("returned_rows", 0) < summary.get("limit", 3):
        reasons.append("returned_rows")
    reasons.extend(notes)
    return _dedupe_preserve_order(reasons)


def _count_selected_by_fiscal_year(selected: Iterable[RawRegistryEntry]) -> dict[str, int]:
    counts = Counter(
        str(entry.inferred_fiscal_year) if entry.inferred_fiscal_year is not None else "unknown"
        for entry in selected
    )
    return dict(sorted(counts.items()))


def _db_scope_check(*, db_path: Path, selected_document_keys: list[str]) -> dict[str, Any]:
    selected = set(selected_document_keys)
    if not db_path.exists():
        return {
            "status": "unavailable",
            "selected_document_count": len(selected),
            "db_document_count": 0,
            "non_selected_document_ids": [],
            "notes": [f"db_missing={db_path}"],
        }
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT document_id FROM financial_facts ORDER BY document_id"
            ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).casefold():
            return {
                "status": "unavailable",
                "selected_document_count": len(selected),
                "db_document_count": 0,
                "non_selected_document_ids": [],
                "notes": ["financial_facts_table_missing"],
            }
        raise

    db_document_ids = [str(row[0]) for row in rows]
    non_selected = [document_id for document_id in db_document_ids if document_id not in selected]
    return {
        "status": "warning" if non_selected else "ok",
        "selected_document_count": len(selected),
        "db_document_count": len(db_document_ids),
        "non_selected_document_ids": non_selected,
        "notes": (
            [
                "current SQLite KB contains document ids outside this selection; "
                "Ask Filing defaults to reading the whole DB."
            ]
            if non_selected
            else []
        ),
    }


def _assert_allowed_metric_ids(
    facts: Iterable[FinancialFact],
    *,
    allowed_metric_ids: set[str],
) -> None:
    disallowed = sorted({fact.metric_id for fact in facts} - allowed_metric_ids)
    if disallowed:
        raise ValueError(f"Converted facts contain disallowed metric_ids: {', '.join(disallowed)}")


def _wrapper_path(*, artifact_dir: Path, document_key: str) -> Path:
    return artifact_dir / f"{document_key}.financial_facts_backfill.json"


def _load_selection(
    *,
    allowlist_file: Path | None,
    use_default_v1d: bool,
    select_all_eligible_annual_reports: bool,
) -> tuple[str, list[str]]:
    selected_modes = [
        allowlist_file is not None,
        use_default_v1d,
        select_all_eligible_annual_reports,
    ]
    if sum(1 for selected in selected_modes if selected) != 1:
        raise ValueError(
            "Provide exactly one of --use-default-v1d-allowlist, --allowlist-file, "
            "or --select-all-eligible-annual-reports."
        )
    if select_all_eligible_annual_reports:
        return SELECTION_MODE_ALL_ELIGIBLE_ANNUAL_REPORTS, []
    if allowlist_file is None:
        return SELECTION_MODE_DEFAULT_V1D_ALLOWLIST, list(DEFAULT_V1D_ALLOWLIST)

    path = _resolve_repo_path(allowlist_file)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return (
                SELECTION_MODE_EXPLICIT_ALLOWLIST,
                _dedupe_preserve_order(str(item) for item in payload),
            )
        if isinstance(payload, dict):
            values = payload.get("document_keys") or payload.get("allowlist")
            if isinstance(values, list):
                return (
                    SELECTION_MODE_EXPLICIT_ALLOWLIST,
                    _dedupe_preserve_order(str(item) for item in values),
                )
        raise ValueError(f"Unsupported allowlist JSON shape: {path}")
    return (
        SELECTION_MODE_EXPLICIT_ALLOWLIST,
        _dedupe_preserve_order(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ),
    )


def _load_registry(path: Path) -> RawDocumentRegistry:
    return RawDocumentRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path_text(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return Path(left).as_posix().casefold() == Path(right).as_posix().casefold()


def _enum_or_default(enum_type: type[Any], value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return enum_type(value)
    except ValueError:
        return default


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


if __name__ == "__main__":
    main()
