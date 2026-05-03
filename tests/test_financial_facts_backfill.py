from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from filingdelta.financial_facts import SQLiteFinancialFactStore
from filingdelta.financial_facts.schemas import FinancialFact
from filingdelta.schemas.facts import ExtractedFactField, HeadlineMetricFacts


def test_backfill_dry_run_selects_exact_allowlist_and_has_no_execute_cost(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backfill = _load_backfill_module()
    selected_file = tmp_path / "selected.pdf"
    summary_file = tmp_path / "summary.pdf"
    interim_file = tmp_path / "interim.pdf"
    mismatch_file = tmp_path / "mismatch.pdf"
    for path in (selected_file, summary_file, interim_file, mismatch_file):
        path.write_text(path.name, encoding="utf-8")

    selected_key = "selected-doc"
    summary_key = "summary-doc"
    interim_key = "interim-doc"
    mismatch_key = "mismatch-doc"
    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        [
            _registry_doc(selected_key, selected_file, checksum=_sha256(selected_file)),
            _registry_doc(
                summary_key,
                summary_file,
                checksum=_sha256(summary_file),
                filename="company annual summary.pdf",
            ),
            _registry_doc(
                interim_key,
                interim_file,
                checksum=_sha256(interim_file),
                inferred_doc_type="interim_report",
            ),
            _registry_doc(mismatch_key, mismatch_file, checksum="bad-checksum"),
        ],
    )
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(
        json.dumps([selected_key, summary_key, interim_key, mismatch_key, "missing-doc"]),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        backfill,
        "_execute_selection",
        lambda **_: (_ for _ in ()).throw(AssertionError("execute path should not run")),
    )

    report = backfill.main(
        [
            "--registry",
            str(registry_path),
            "--allowlist-file",
            str(allowlist_path),
            "--report",
            str(report_path),
        ]
    )

    assert report["mode"] == "dry_run"
    assert report["selected_document_keys"] == [selected_key]
    assert report["skipped_reasons"] == {
        "checksum_mismatch": 1,
        "hard_exclusion_terms": 2,
        "not_annual_report": 1,
        "not_in_registry": 1,
    }
    assert report["execution"]["executed"] is False
    assert report_path.exists()


def test_backfill_requires_explicit_allowlist_choice(tmp_path: Path) -> None:
    backfill = _load_backfill_module()
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, [])

    with pytest.raises(ValueError, match="--use-default-v1d-allowlist"):
        backfill.main(["--registry", str(registry_path), "--report", str(tmp_path / "report.json")])


def test_backfill_hard_exclusion_checks_company_fields(tmp_path: Path) -> None:
    backfill = _load_backfill_module()
    raw_file = tmp_path / "annual.pdf"
    raw_file.write_text("annual", encoding="utf-8")
    document_key = "doc-a"
    registry_path = tmp_path / "registry.json"
    payload = _registry_doc(document_key, raw_file, checksum=_sha256(raw_file))
    payload["inferred_company_name"] = "Example Summary Holdings"
    _write_registry(registry_path, [payload])
    allowlist_path = tmp_path / "allowlist.txt"
    allowlist_path.write_text(document_key, encoding="utf-8")

    report = backfill.main(
        [
            "--registry",
            str(registry_path),
            "--allowlist-file",
            str(allowlist_path),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    assert report["selected_count"] == 0
    assert report["skipped_reasons"]["hard_exclusion_terms"] == 1


def test_backfill_execute_reuses_validated_wrapper_and_writes_only_canonical_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backfill = _load_backfill_module()
    raw_file = tmp_path / "annual.pdf"
    raw_file.write_text("annual", encoding="utf-8")
    checksum = _sha256(raw_file)
    document_key = "doc-a"
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, [_registry_doc(document_key, raw_file, checksum=checksum)])
    allowlist_path = tmp_path / "allowlist.txt"
    allowlist_path.write_text(document_key, encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    wrapper_path = artifact_dir / f"{document_key}.financial_facts_backfill.json"
    facts = _headline_metrics(document_key=document_key, source_path=raw_file)
    _write_wrapper(
        wrapper_path,
        backfill=backfill,
        document_key=document_key,
        raw_file=raw_file,
        checksum=checksum,
        facts=facts,
    )
    monkeypatch.setattr(
        backfill,
        "_run_parse_extract",
        lambda _entry: (_ for _ in ()).throw(AssertionError("wrapper should be reused")),
    )

    report = backfill.main(
        [
            "--execute",
            "--reuse-existing-artifacts",
            "--registry",
            str(registry_path),
            "--allowlist-file",
            str(allowlist_path),
            "--artifact-dir",
            str(artifact_dir),
            "--db",
            str(tmp_path / "facts.sqlite"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    store = SQLiteFinancialFactStore(tmp_path / "facts.sqlite")
    rows = store.list_facts()
    assert [row.metric_id for row in rows] == [
        "net_profit_attributable",
        "revenue",
        "total_assets",
        "total_liabilities",
    ]
    assert "roe" not in {row.metric_id for row in rows}
    document_result = report["execution"]["document_results"][0]
    assert document_result["artifact_action"] == "reused"
    assert document_result["facts_written"] == 4
    assert report["execution"]["review_status_by_metric"]["revenue"]["verified"] == 1
    assert report["selected_scope_revenue_top3_status"]["status"] == "partial"
    assert "selected_docs" in report["selected_scope_revenue_top3_status"]["reasons"]
    assert list(artifact_dir.rglob("*.headline_metrics.json")) == []


def test_backfill_wrapper_reuse_requires_explicit_metric_keys(tmp_path: Path) -> None:
    backfill = _load_backfill_module()
    raw_file = tmp_path / "annual.pdf"
    raw_file.write_text("annual", encoding="utf-8")
    checksum = _sha256(raw_file)
    entry = backfill.RawRegistryEntry.model_validate(
        _registry_doc("doc-a", raw_file, checksum=checksum)
    )
    wrapper_path = tmp_path / "doc-a.financial_facts_backfill.json"
    facts = _headline_metrics(document_key="doc-a", source_path=raw_file).model_dump(mode="json")
    facts.pop("total_liabilities")
    wrapper_path.write_text(
        json.dumps(
            {
                "schema_version": backfill.WRAPPER_SCHEMA_VERSION,
                "document_key": "doc-a",
                "registry_local_path": str(raw_file),
                "resolved_source_path": str(raw_file.resolve()),
                "checksum_sha256": checksum,
                "headline_metrics": facts,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    reusable, reason = backfill._try_load_reusable_wrapper(wrapper_path=wrapper_path, entry=entry)

    assert reusable is None
    assert reason == "headline_metric_keys_missing"


def test_backfill_wrapper_reuse_rejects_nested_headline_document_mismatch(
    tmp_path: Path,
) -> None:
    backfill = _load_backfill_module()
    raw_file = tmp_path / "annual.pdf"
    raw_file.write_text("annual", encoding="utf-8")
    checksum = _sha256(raw_file)
    entry = backfill.RawRegistryEntry.model_validate(
        _registry_doc("doc-a", raw_file, checksum=checksum)
    )
    wrapper_path = tmp_path / "doc-a.financial_facts_backfill.json"
    stale_facts = _headline_metrics(document_key="path-stem-doc", source_path=raw_file)
    _write_wrapper(
        wrapper_path,
        backfill=backfill,
        document_key="doc-a",
        raw_file=raw_file,
        checksum=checksum,
        facts=stale_facts,
    )

    reusable, reason = backfill._try_load_reusable_wrapper(wrapper_path=wrapper_path, entry=entry)

    assert reusable is None
    assert reason == "headline_document_id_mismatch"


def test_backfill_metric_guard_rejects_non_canonical_financial_facts() -> None:
    backfill = _load_backfill_module()
    disallowed = _financial_fact("doc-a", metric_id="roe")

    with pytest.raises(ValueError, match="disallowed metric_ids"):
        backfill._assert_allowed_metric_ids(
            [disallowed],
            allowed_metric_ids=set(backfill.ALLOWED_METRIC_IDS),
        )


def test_backfill_report_counts_missing_metrics_not_as_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backfill = _load_backfill_module()
    raw_a = tmp_path / "a.pdf"
    raw_b = tmp_path / "b.pdf"
    raw_a.write_text("a", encoding="utf-8")
    raw_b.write_text("b", encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    _write_registry(
        registry_path,
        [
            _registry_doc("doc-a", raw_a, checksum=_sha256(raw_a)),
            _registry_doc("doc-b", raw_b, checksum=_sha256(raw_b)),
        ],
    )
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(json.dumps(["doc-a", "doc-b"]), encoding="utf-8")
    artifact_dir = tmp_path / "artifacts"
    for document_key, raw_file, include_revenue in (
        ("doc-a", raw_a, True),
        ("doc-b", raw_b, False),
    ):
        _write_wrapper(
            artifact_dir / f"{document_key}.financial_facts_backfill.json",
            backfill=backfill,
            document_key=document_key,
            raw_file=raw_file,
            checksum=_sha256(raw_file),
            facts=_headline_metrics(
                document_key=document_key,
                source_path=raw_file,
                include_revenue=include_revenue,
            ),
        )
    monkeypatch.setattr(
        backfill,
        "_run_parse_extract",
        lambda _entry: (_ for _ in ()).throw(AssertionError("wrapper should be reused")),
    )

    report = backfill.main(
        [
            "--execute",
            "--reuse-existing-artifacts",
            "--registry",
            str(registry_path),
            "--allowlist-file",
            str(allowlist_path),
            "--artifact-dir",
            str(artifact_dir),
            "--db",
            str(tmp_path / "facts.sqlite"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    revenue_summary = report["execution"]["review_status_by_metric"]["revenue"]
    assert revenue_summary["verified"] == 1
    assert revenue_summary["missing"] == 1
    assert report["selected_scope_revenue_top3_status"]["status"] == "partial"
    assert "verified_annual_candidates" in report["selected_scope_revenue_top3_status"]["reasons"]


def _load_backfill_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "backfill_financial_facts_from_raw.py"
    )
    spec = importlib.util.spec_from_file_location("backfill_financial_facts_from_raw", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_registry(path: Path, documents: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "raw_document_registry.v1",
                "raw_dir": str(path.parent),
                "documents": documents,
                "summary": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _registry_doc(
    document_key: str,
    raw_file: Path,
    *,
    checksum: str,
    filename: str | None = None,
    inferred_doc_type: str = "annual_report",
) -> dict[str, object]:
    return {
        "document_key": document_key,
        "local_path": str(raw_file),
        "filename": filename or raw_file.name,
        "suffix": raw_file.suffix,
        "file_size": raw_file.stat().st_size,
        "checksum_sha256": checksum,
        "inferred_company_name": f"{document_key} company",
        "inferred_fiscal_year": 2025,
        "fiscal_period": "2025 annual report",
        "inferred_doc_type": inferred_doc_type,
        "inferred_market": "a_share",
        "language": "zh",
        "warnings": [],
    }


def _write_wrapper(
    wrapper_path: Path,
    *,
    backfill: ModuleType,
    document_key: str,
    raw_file: Path,
    checksum: str,
    facts: HeadlineMetricFacts,
) -> None:
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(
        json.dumps(
            {
                "schema_version": backfill.WRAPPER_SCHEMA_VERSION,
                "document_key": document_key,
                "registry_local_path": str(raw_file),
                "resolved_source_path": str(raw_file.resolve()),
                "checksum_sha256": checksum,
                "headline_metrics": facts.model_dump(mode="json"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _headline_metrics(
    *,
    document_key: str,
    source_path: Path,
    include_revenue: bool = True,
) -> HeadlineMetricFacts:
    return HeadlineMetricFacts(
        document_id=document_key,
        source_path=source_path,
        company_name=ExtractedFactField(value=f"{document_key} company"),
        fiscal_period=ExtractedFactField(value="2025 annual report"),
        unit=ExtractedFactField(value="RMB million"),
        revenue=_field(100, "Revenue 100") if include_revenue else ExtractedFactField(),
        net_profit=_field(50, "Profit attributable 50"),
        total_assets=_field(1000, "Total assets 1000"),
        total_liabilities=_field(500, "Total liabilities 500"),
        roe=_field(12.3, "ROE 12.3"),
    )


def _field(value: float, quote: str) -> ExtractedFactField:
    return ExtractedFactField(value=value, evidence_page=8, evidence_quote=quote)


def _financial_fact(document_id: str, *, metric_id: str) -> FinancialFact:
    return FinancialFact(
        fact_id=f"{document_id}:{metric_id}",
        document_id=document_id,
        company_name="A",
        source_path=Path("a.pdf"),
        metric_id=metric_id,
        metric_label=metric_id,
        source_metric_name=metric_id,
        period_type="period",
        fiscal_period="2025 annual report",
        fiscal_year=2025,
        value=1,
        unit_raw="RMB million",
        currency="CNY",
        scale=1_000_000,
        normalized_value=1_000_000,
        normalized_unit="CNY",
        evidence_page=1,
        evidence_quote="quote",
        review_status="verified",
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
