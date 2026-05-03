from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from filingdelta.financial_facts import (
    CANONICAL_METRICS,
    SQLiteFinancialFactStore,
    canonicalize_metric_id,
    convert_headline_metric_facts,
)
from filingdelta.financial_facts.normalization import normalize_numeric_value, normalize_unit
from filingdelta.financial_facts.schemas import FinancialFact
from filingdelta.schemas.facts import ExtractedFactField, HeadlineMetricFacts
from filingdelta.schemas.filing import Citation


def test_metric_aliases_and_period_types_are_canonical() -> None:
    assert canonicalize_metric_id("营业收入") == "revenue"
    assert canonicalize_metric_id("net_profit") == "net_profit_attributable"
    assert canonicalize_metric_id("归属于母公司股东的净利润") == "net_profit_attributable"
    assert canonicalize_metric_id("total assets") == "total_assets"
    assert canonicalize_metric_id("总负债") == "total_liabilities"

    assert CANONICAL_METRICS["revenue"].period_type == "period"
    assert CANONICAL_METRICS["net_profit_attributable"].period_type == "period"
    assert CANONICAL_METRICS["total_assets"].period_type == "end_of_period"
    assert CANONICAL_METRICS["total_liabilities"].period_type == "end_of_period"


def test_unit_currency_scale_and_value_normalization() -> None:
    unit = normalize_unit("人民币百万元，特别注明除外")
    assert unit.currency == "CNY"
    assert unit.scale == 1_000_000
    assert unit.normalized_unit == "CNY"

    hkd_unit = normalize_unit("HKD million")
    assert hkd_unit.currency == "HKD"
    assert hkd_unit.scale == 1_000_000

    assert normalize_numeric_value("337,532") == 337_532
    assert normalize_numeric_value("bad") is None


def test_headline_metric_conversion_creates_verified_rows_with_page_quote_citations() -> None:
    facts = _headline_metrics(
        revenue=ExtractedFactField(
            value=337_532,
            evidence_page=8,
            evidence_quote="营业收入 337,532 人民币百万元",
        ),
        net_profit=ExtractedFactField(
            value=150_181,
            citations=[
                Citation(
                    document_id="doc-a",
                    source_path=Path("sample.pdf"),
                    page_number=8,
                    quote="归属于本行股东的净利润 150,181 人民币百万元",
                )
            ],
        ),
        total_assets=ExtractedFactField(
            value=12_345_678,
            evidence_page=9,
            evidence_quote="Total assets 12,345,678 RMB million",
        ),
        total_liabilities=ExtractedFactField(
            value=6_543_210,
            evidence_page=9,
            evidence_quote="Total liabilities 6,543,210 RMB million",
        ),
    )

    rows = convert_headline_metric_facts(facts)

    assert [row.metric_id for row in rows] == [
        "revenue",
        "net_profit_attributable",
        "total_assets",
        "total_liabilities",
    ]
    assert all(row.review_status == "verified" for row in rows)
    assert rows[0].period_type == "period"
    assert rows[1].period_type == "period"
    assert rows[2].period_type == "end_of_period"
    assert rows[3].period_type == "end_of_period"
    assert rows[0].fiscal_year == 2025
    assert rows[0].normalized_value == 337_532_000_000
    assert rows[0].has_page_quote_citation is True
    assert rows[1].evidence_page == 8
    assert "150,181" in rows[1].evidence_quote
    assert rows[2].metric_id == "total_assets"
    assert rows[3].metric_id == "total_liabilities"


def test_quote_local_unit_overrides_global_unit_for_bound_value() -> None:
    facts = _headline_metrics(
        unit="\u4eba\u6c11\u5e63\u5343\u5143",
        revenue=ExtractedFactField(value=None),
        net_profit=ExtractedFactField(
            value=32619,
            evidence_page=26,
            evidence_quote=(
                "\u6b78\u5c6c\u65bc\u4e0a\u5e02\u516c\u53f8\u80a1\u6771\u7684\u6de8\u5229\u6f64"
                "\u7d04\u4eba\u6c11\u5e63 32,619\u767e\u842c\u5143,\u540c\u6bd4\u4e0b\u964d18.97%."
            ),
        ),
    )

    rows = convert_headline_metric_facts(facts)

    assert len(rows) == 1
    row = rows[0]
    assert row.metric_id == "net_profit_attributable"
    assert row.review_status == "verified"
    assert row.currency == "CNY"
    assert row.scale == 1_000_000
    assert row.unit_raw == "\u4eba\u6c11\u5e63 \u767e\u842c\u5143"
    assert row.normalized_value == 32_619_000_000
    assert "unit_source=quote" in (row.notes or "")
    assert "global_unit_ignored=\u4eba\u6c11\u5e63\u5343\u5143" in (row.notes or "")


def test_quote_local_unit_uses_longest_non_overlapping_unit_token() -> None:
    facts = _headline_metrics(
        unit="\u4eba\u6c11\u5e01\u5343\u5143",
        revenue=ExtractedFactField(
            value=32619,
            evidence_page=5,
            evidence_quote="\u8425\u4e1a\u6536\u5165 32,619\u767e\u4e07\u5143",
        ),
        net_profit=ExtractedFactField(
            value=1234,
            evidence_page=6,
            evidence_quote="\u5f52\u6bcd\u51c0\u5229\u6da6 1,234\u4ebf\u5143",
        ),
    )

    rows = convert_headline_metric_facts(facts)
    revenue = _row_by_metric(rows, "revenue")
    net_profit = _row_by_metric(rows, "net_profit_attributable")

    assert revenue.review_status == "verified"
    assert revenue.scale == 1_000_000
    assert revenue.unit_raw == "\u767e\u4e07\u5143"
    assert revenue.normalized_value == 32_619_000_000
    assert net_profit.review_status == "verified"
    assert net_profit.scale == 100_000_000
    assert net_profit.unit_raw == "\u4ebf\u5143"
    assert net_profit.normalized_value == 123_400_000_000


def test_quote_local_unit_normalizes_fullwidth_numbers_and_punctuation() -> None:
    facts = _headline_metrics(
        unit="\u4eba\u6c11\u5e63\u5343\u5143",
        revenue=ExtractedFactField(value=None),
        net_profit=ExtractedFactField(
            value=32619,
            evidence_page=26,
            evidence_quote="\u6b78\u6bcd\u6de8\u5229\u6f64 \uff13\uff12\uff0c\uff16\uff11\uff19\u767e\u842c\u5143",
        ),
    )

    row = convert_headline_metric_facts(facts)[0]

    assert row.review_status == "verified"
    assert row.normalized_value == 32_619_000_000
    assert "unit_source=quote" in (row.notes or "")


def test_quote_local_unit_allows_multiple_same_unit_matches() -> None:
    facts = _headline_metrics(
        unit="\u4eba\u6c11\u5e01\u767e\u4e07\u5143",
        revenue=ExtractedFactField(
            value=100,
            evidence_page=8,
            evidence_quote=(
                "\u8425\u4e1a\u6536\u5165 100\u767e\u4e07\u5143;"
                "\u8c03\u6574\u540e\u8425\u4e1a\u6536\u5165 100\u767e\u4e07\u5143"
            ),
        ),
        net_profit=ExtractedFactField(value=None),
    )

    row = convert_headline_metric_facts(facts)[0]

    assert row.review_status == "verified"
    assert row.normalized_value == 100_000_000


def test_quote_local_unit_resolves_hkd_units() -> None:
    facts = _headline_metrics(
        unit=None,
        revenue=ExtractedFactField(
            value=1234,
            evidence_page=9,
            evidence_quote="Revenue 1,234 HKD million",
        ),
        net_profit=ExtractedFactField(
            value=32619,
            evidence_page=10,
            evidence_quote="Profit attributable 32,619\u6e2f\u5143",
        ),
    )

    rows = convert_headline_metric_facts(facts)
    revenue = _row_by_metric(rows, "revenue")
    net_profit = _row_by_metric(rows, "net_profit_attributable")

    assert revenue.review_status == "verified"
    assert revenue.currency == "HKD"
    assert revenue.scale == 1_000_000
    assert revenue.normalized_value == 1_234_000_000
    assert net_profit.review_status == "verified"
    assert net_profit.currency == "HKD"
    assert net_profit.scale == 1
    assert net_profit.normalized_value == 32_619


def test_quote_local_unit_falls_back_to_global_unit_when_quote_has_no_unit() -> None:
    facts = _headline_metrics(
        unit="\u4eba\u6c11\u5e01\u767e\u4e07\u5143",
        revenue=ExtractedFactField(
            value=100,
            evidence_page=8,
            evidence_quote="\u8425\u4e1a\u6536\u5165 100",
        ),
        net_profit=ExtractedFactField(value=None),
    )

    rows = convert_headline_metric_facts(facts)

    assert len(rows) == 1
    assert rows[0].review_status == "verified"
    assert rows[0].currency == "CNY"
    assert rows[0].scale == 1_000_000
    assert rows[0].normalized_value == 100_000_000
    assert rows[0].notes is None


def test_quote_local_unit_marks_unbound_or_conflicting_units_for_review() -> None:
    unbound = _headline_metrics(
        unit="\u4eba\u6c11\u5e01\u5343\u5143",
        revenue=ExtractedFactField(
            value=32619,
            evidence_page=5,
            evidence_quote="\u8425\u4e1a\u6536\u5165 32,620\u767e\u4e07\u5143",
        ),
        net_profit=ExtractedFactField(value=None),
    )
    conflict = _headline_metrics(
        unit="\u4eba\u6c11\u5e01\u767e\u4e07\u5143",
        revenue=ExtractedFactField(
            value=32619,
            evidence_page=5,
            evidence_quote="\u8425\u4e1a\u6536\u5165 32,619\u6e2f\u5143",
        ),
        net_profit=ExtractedFactField(value=None),
    )

    unbound_row = convert_headline_metric_facts(unbound)[0]
    conflict_row = convert_headline_metric_facts(conflict)[0]

    assert unbound_row.review_status == "needs_review"
    assert unbound_row.normalized_value is None
    assert "value_unbound" in (unbound_row.notes or "")
    assert conflict_row.review_status == "needs_review"
    assert conflict_row.normalized_value is None
    assert "currency_conflict" in (conflict_row.notes or "")


def test_quote_local_unit_marks_different_units_for_same_value_as_ambiguous() -> None:
    facts = _headline_metrics(
        unit="\u4eba\u6c11\u5e01\u767e\u4e07\u5143",
        revenue=ExtractedFactField(
            value=100,
            evidence_page=8,
            evidence_quote=(
                "\u8425\u4e1a\u6536\u5165 100\u767e\u4e07\u5143;"
                "\u8c03\u6574\u540e\u8425\u4e1a\u6536\u5165 100\u5343\u5143"
            ),
        ),
        net_profit=ExtractedFactField(value=None),
    )

    row = convert_headline_metric_facts(facts)[0]

    assert row.review_status == "needs_review"
    assert row.normalized_value is None
    assert "unit_ambiguous" in (row.notes or "")


def test_quote_local_unit_uses_citation_quote_path() -> None:
    facts = _headline_metrics(
        unit="\u4eba\u6c11\u5e63\u5343\u5143",
        revenue=ExtractedFactField(value=None),
        net_profit=ExtractedFactField(
            value=32619,
            citations=[
                Citation(
                    document_id="doc-a",
                    source_path=Path("sample.pdf"),
                    page_number=26,
                    quote="\u6b78\u6bcd\u6de8\u5229\u6f64 32,619\u767e\u842c\u5143",
                )
            ],
        ),
    )

    row = convert_headline_metric_facts(facts)[0]

    assert row.review_status == "verified"
    assert row.evidence_page == 26
    assert row.normalized_value == 32_619_000_000
    assert "unit_source=quote" in (row.notes or "")


def test_headline_metric_conversion_marks_missing_evidence_or_unit_for_review() -> None:
    facts = _headline_metrics(
        fiscal_period="2025 annual report",
        unit=None,
        revenue=ExtractedFactField(value=10),
        net_profit=ExtractedFactField(value=None),
    )

    rows = convert_headline_metric_facts(facts)

    assert len(rows) == 1
    assert rows[0].metric_id == "revenue"
    assert rows[0].review_status == "needs_review"
    assert "unit_currency_or_scale" in (rows[0].notes or "")
    assert "evidence_page" in (rows[0].notes or "")


def test_headline_metric_conversion_requires_extractable_fiscal_year() -> None:
    facts = _headline_metrics(
        fiscal_period="annual report",
        revenue=ExtractedFactField(
            value=10,
            evidence_page=1,
            evidence_quote="Revenue 10",
        ),
        net_profit=ExtractedFactField(value=None),
    )

    rows = convert_headline_metric_facts(facts)

    assert len(rows) == 1
    assert rows[0].review_status == "needs_review"
    assert rows[0].fiscal_year is None
    assert "fiscal_year" in (rows[0].notes or "")


def test_sqlite_store_upserts_and_filters_by_review_status(tmp_path: Path) -> None:
    store = SQLiteFinancialFactStore(tmp_path / "facts.sqlite")
    verified = _fact(
        fact_id="verified",
        document_id="doc-a",
        company_name="A",
        normalized_value=300,
        review_status="verified",
    )
    needs_review = _fact(
        fact_id="needs-review",
        document_id="doc-b",
        company_name="B",
        normalized_value=500,
        review_status="needs_review",
    )

    assert store.upsert_facts([verified, needs_review]) == 2
    updated = verified.model_copy(update={"value": 4.0, "normalized_value": 400.0})
    assert store.upsert_facts([updated]) == 1

    verified_rows = store.list_facts(review_status="verified")

    assert len(verified_rows) == 1
    assert verified_rows[0].fact_id == "verified"
    assert verified_rows[0].normalized_value == 400


def test_top3_revenue_query_sorts_verified_rows_and_requires_citations(tmp_path: Path) -> None:
    store = SQLiteFinancialFactStore(tmp_path / "facts.sqlite")
    store.upsert_facts(
        [
            _fact("doc-a", "A", 100, fiscal_period="2025 annual report"),
            _fact("doc-b", "B", 300, fiscal_period="year ended December 31, 2025"),
            _fact("doc-c", "C", 200, fiscal_period="2025 annual report"),
            _fact("doc-d", "D", 999, review_status="needs_review"),
            _fact("doc-e", "E", 800, evidence_quote=""),
            _fact("doc-f", "F", 700, fiscal_year=2024),
        ]
    )

    top = store.top_revenue_by_year(fiscal_year=2025, limit=3)

    assert [fact.document_id for fact in top] == ["doc-b", "doc-c", "doc-a"]
    assert all(fact.review_status == "verified" for fact in top)
    assert all(fact.has_page_quote_citation for fact in top)


def test_top3_revenue_query_excludes_unit_resolution_needs_review_rows(tmp_path: Path) -> None:
    store = SQLiteFinancialFactStore(tmp_path / "facts.sqlite")
    store.upsert_facts(
        [
            _fact("doc-a", "A", 300, fiscal_period="2025 annual report"),
            _fact(
                "doc-b",
                "B",
                500,
                fiscal_period="2025 annual report",
                review_status="needs_review",
            ),
            _fact("doc-c", "C", 200, fiscal_period="2025 annual report"),
        ]
    )

    top = store.top_revenue_by_year(fiscal_year=2025, limit=3)

    assert [fact.document_id for fact in top] == ["doc-a", "doc-c"]


def test_top3_revenue_query_filters_annual_period_and_dedupes_company(tmp_path: Path) -> None:
    store = SQLiteFinancialFactStore(tmp_path / "facts.sqlite")
    cmb = "\u62db\u5546\u94f6\u884c"
    tencent = "\u817e\u8baf\u63a7\u80a1"
    moutai = "\u8d35\u5dde\u8305\u53f0"
    store.upsert_facts(
        [
            _fact("cmb-annual-a", cmb, 300, fiscal_period="2025 annual report"),
            _fact("cmb-q3", cmb, 999, fiscal_period="2025\u5e741-9\u6708"),
            _fact("cmb-annual-b", cmb, 320, fiscal_period="2025\u5e74\u5ea6\u62a5\u544a"),
            _fact("tencent-annual", tencent, 250, fiscal_period="year ended December 31, 2025"),
            _fact("moutai-interim", moutai, 500, fiscal_period="2025 interim report"),
            _fact("moutai-annual", moutai, 200, fiscal_period="2025 annual report"),
        ]
    )

    top = store.top_revenue_by_year(fiscal_year=2025, limit=3)
    stats = store.top_revenue_by_year_stats(fiscal_year=2025, limit=3)

    assert [fact.company_name for fact in top] == [cmb, tencent, moutai]
    assert [fact.document_id for fact in top] == [
        "cmb-annual-b",
        "tencent-annual",
        "moutai-annual",
    ]
    assert stats == {
        "selected_docs": 0,
        "candidate_count": 6,
        "verified_annual_candidates": 4,
        "after_citation_filter": 4,
        "after_company_dedupe": 3,
        "excluded_non_annual_count": 2,
        "excluded_duplicate_company_count": 1,
        "returned_rows": 3,
    }


def test_replace_facts_for_document_is_transactional_and_rejects_cross_document_rows(
    tmp_path: Path,
) -> None:
    store = SQLiteFinancialFactStore(tmp_path / "facts.sqlite")
    original = _fact("doc-a", "A", 100, fact_id="doc-a:old")
    store.upsert_facts([original])

    mismatched = [
        _fact("doc-a", "A", 200, fact_id="doc-a:new"),
        _fact("doc-b", "B", 300, fact_id="doc-b:wrong"),
    ]

    try:
        store.replace_facts_for_document("doc-a", mismatched)
    except ValueError as exc:
        assert "doc-b" in str(exc)
    else:
        raise AssertionError("replace_facts_for_document should reject cross-document facts")

    rows = store.list_facts(metric_id="revenue")
    assert [row.fact_id for row in rows] == ["doc-a:old"]
    assert rows[0].normalized_value == 100

    result = store.replace_facts_for_document(
        "doc-a",
        [_fact("doc-a", "A", 400, fact_id="doc-a:new")],
    )

    assert result == {"deleted": 1, "upserted": 1}
    rows = store.list_facts(metric_id="revenue")
    assert [row.fact_id for row in rows] == ["doc-a:new"]
    assert rows[0].normalized_value == 400


def test_top_metric_by_year_can_limit_to_selected_documents(tmp_path: Path) -> None:
    store = SQLiteFinancialFactStore(tmp_path / "facts.sqlite")
    store.upsert_facts(
        [
            _fact("selected-a", "A", 200, fiscal_period="2025 annual report"),
            _fact("selected-b", "B", 100, fiscal_period="2025 annual report"),
            _fact("outside", "Outside", 999, fiscal_period="2025 annual report"),
        ]
    )

    top = store.top_metric_by_year(
        metric_id="revenue",
        fiscal_year=2025,
        limit=3,
        document_ids=["selected-a", "selected-b"],
    )
    stats = store.top_metric_by_year_stats(
        metric_id="revenue",
        fiscal_year=2025,
        limit=3,
        document_ids=["selected-a", "selected-b"],
    )

    assert [fact.document_id for fact in top] == ["selected-a", "selected-b"]
    assert stats["selected_docs"] == 2
    assert stats["candidate_count"] == 2
    assert stats["returned_rows"] == 2


def test_query_script_reconfigures_stdout_for_readable_chinese_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    query_module = _load_query_script_module()
    store = SQLiteFinancialFactStore(tmp_path / "facts.sqlite")
    company = "\u62db\u5546\u94f6\u884c"
    store.upsert_facts([_fact("cmb-annual", company, 300, fiscal_period="2025 annual report")])
    stdout = _EncodingGuardedStdout(encoding="cp1252")
    monkeypatch.setattr(query_module.sys, "stdout", stdout)

    payload = query_module.main(
        [
            "--db",
            str(tmp_path / "facts.sqlite"),
            "--top-revenue-year",
            "2025",
            "--limit",
            "3",
        ]
    )

    rendered = "".join(stdout.writes)
    assert stdout.encoding == "utf-8"
    assert company in rendered
    assert "\\u62db\\u5546" not in rendered
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["returned_rows"] == 1


def _headline_metrics(
    *,
    revenue: ExtractedFactField,
    net_profit: ExtractedFactField,
    total_assets: ExtractedFactField | None = None,
    total_liabilities: ExtractedFactField | None = None,
    fiscal_period: str | None = "2025 annual report",
    unit: str | None = "人民币百万元",
) -> HeadlineMetricFacts:
    return HeadlineMetricFacts(
        document_id="doc-a",
        source_path=Path("sample.pdf"),
        company_name=ExtractedFactField(value="招商银行"),
        fiscal_period=ExtractedFactField(value=fiscal_period),
        unit=ExtractedFactField(value=unit),
        revenue=revenue,
        net_profit=net_profit,
        total_assets=total_assets or ExtractedFactField(),
        total_liabilities=total_liabilities or ExtractedFactField(),
    )


def _fact(
    document_id: str = "doc-a",
    company_name: str = "A",
    normalized_value: float = 100,
    *,
    fact_id: str | None = None,
    review_status: str = "verified",
    evidence_quote: str = "营业收入 100",
    fiscal_year: int = 2025,
    fiscal_period: str | None = None,
) -> FinancialFact:
    return FinancialFact(
        fact_id=fact_id or f"{document_id}:revenue",
        document_id=document_id,
        company_name=company_name,
        source_path=Path(f"{document_id}.pdf"),
        metric_id="revenue",
        metric_label="营业收入",
        source_metric_name="revenue",
        period_type="period",
        fiscal_period=fiscal_period or f"{fiscal_year} annual report",
        fiscal_year=fiscal_year,
        value=normalized_value / 1_000_000,
        unit_raw="人民币百万元",
        currency="CNY",
        scale=1_000_000,
        normalized_value=normalized_value,
        normalized_unit="CNY",
        evidence_page=8,
        evidence_quote=evidence_quote,
        review_status=review_status,
    )


def _row_by_metric(rows: list[FinancialFact], metric_id: str) -> FinancialFact:
    for row in rows:
        if row.metric_id == metric_id:
            return row
    raise AssertionError(f"metric row not found: {metric_id}")


def _load_query_script_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "query_financial_facts.py"
    spec = importlib.util.spec_from_file_location("query_financial_facts", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _EncodingGuardedStdout:
    def __init__(self, *, encoding: str) -> None:
        self.encoding = encoding
        self.writes: list[str] = []

    def reconfigure(self, *, encoding: str) -> None:
        self.encoding = encoding

    def write(self, text: str) -> int:
        text.encode(self.encoding)
        self.writes.append(text)
        return len(text)
