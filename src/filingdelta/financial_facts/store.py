from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Iterable

from filingdelta.financial_facts.schemas import FinancialFact

_UPSERT_SQL = """
    INSERT INTO financial_facts (
        fact_id, document_id, company_name, ticker, source_path,
        metric_id, metric_label, source_metric_name, period_type,
        fiscal_period, fiscal_year, value, unit_raw, currency,
        scale, normalized_value, normalized_unit, evidence_page,
        evidence_quote, review_status, source, notes
    )
    VALUES (
        :fact_id, :document_id, :company_name, :ticker, :source_path,
        :metric_id, :metric_label, :source_metric_name, :period_type,
        :fiscal_period, :fiscal_year, :value, :unit_raw, :currency,
        :scale, :normalized_value, :normalized_unit, :evidence_page,
        :evidence_quote, :review_status, :source, :notes
    )
    ON CONFLICT(fact_id) DO UPDATE SET
        document_id=excluded.document_id,
        company_name=excluded.company_name,
        ticker=excluded.ticker,
        source_path=excluded.source_path,
        metric_id=excluded.metric_id,
        metric_label=excluded.metric_label,
        source_metric_name=excluded.source_metric_name,
        period_type=excluded.period_type,
        fiscal_period=excluded.fiscal_period,
        fiscal_year=excluded.fiscal_year,
        value=excluded.value,
        unit_raw=excluded.unit_raw,
        currency=excluded.currency,
        scale=excluded.scale,
        normalized_value=excluded.normalized_value,
        normalized_unit=excluded.normalized_unit,
        evidence_page=excluded.evidence_page,
        evidence_quote=excluded.evidence_quote,
        review_status=excluded.review_status,
        source=excluded.source,
        notes=excluded.notes
"""


class SQLiteFinancialFactStore:
    def __init__(self, db_path: Path | str = Path("data/indexes/financial_facts.sqlite")) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_facts (
                    fact_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    company_name TEXT,
                    ticker TEXT,
                    source_path TEXT NOT NULL,
                    metric_id TEXT NOT NULL,
                    metric_label TEXT NOT NULL,
                    source_metric_name TEXT NOT NULL,
                    period_type TEXT NOT NULL,
                    fiscal_period TEXT NOT NULL,
                    fiscal_year INTEGER,
                    value REAL NOT NULL,
                    unit_raw TEXT NOT NULL,
                    currency TEXT,
                    scale REAL,
                    normalized_value REAL,
                    normalized_unit TEXT,
                    evidence_page INTEGER,
                    evidence_quote TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_financial_facts_metric_year_status
                ON financial_facts(metric_id, fiscal_year, review_status, normalized_value)
                """
            )

    def upsert_facts(self, facts: Iterable[FinancialFact]) -> int:
        facts_list = list(facts)
        self.initialize()
        if not facts_list:
            return 0
        with self._connect() as conn:
            _upsert_fact_rows(conn, facts_list)
        return len(facts_list)

    def replace_facts_for_document(
        self,
        document_id: str,
        facts: Iterable[FinancialFact],
    ) -> dict[str, int]:
        facts_list = list(facts)
        mismatched = [fact.document_id for fact in facts_list if fact.document_id != document_id]
        if mismatched:
            raise ValueError(
                "replace_facts_for_document received facts for other documents: "
                + ", ".join(sorted(set(mismatched)))
            )

        self.initialize()
        with self._connect() as conn:
            deleted = conn.execute(
                "DELETE FROM financial_facts WHERE document_id = ?",
                (document_id,),
            ).rowcount
            if facts_list:
                _upsert_fact_rows(conn, facts_list)
        return {"deleted": int(deleted), "upserted": len(facts_list)}

    def list_facts(
        self,
        *,
        metric_id: str | None = None,
        fiscal_year: int | None = None,
        review_status: str | None = None,
    ) -> list[FinancialFact]:
        clauses: list[str] = []
        params: dict[str, object] = {}
        if metric_id is not None:
            clauses.append("metric_id = :metric_id")
            params["metric_id"] = metric_id
        if fiscal_year is not None:
            clauses.append("fiscal_year = :fiscal_year")
            params["fiscal_year"] = fiscal_year
        if review_status is not None:
            clauses.append("review_status = :review_status")
            params["review_status"] = review_status

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM financial_facts
            {where}
            ORDER BY document_id, metric_id, fiscal_period
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_fact(row) for row in rows]

    def top_revenue_by_year(
        self,
        *,
        fiscal_year: int = 2025,
        limit: int = 3,
        document_ids: Iterable[str] | None = None,
    ) -> list[FinancialFact]:
        return self.top_metric_by_year(
            metric_id="revenue",
            fiscal_year=fiscal_year,
            limit=limit,
            document_ids=document_ids,
        )

    def top_revenue_by_year_stats(
        self,
        *,
        fiscal_year: int = 2025,
        limit: int = 3,
        document_ids: Iterable[str] | None = None,
    ) -> dict[str, int]:
        return self.top_metric_by_year_stats(
            metric_id="revenue",
            fiscal_year=fiscal_year,
            limit=limit,
            document_ids=document_ids,
        )

    def top_metric_by_year(
        self,
        *,
        metric_id: str,
        fiscal_year: int,
        limit: int,
        document_ids: Iterable[str] | None = None,
    ) -> list[FinancialFact]:
        return self._top_metric_by_year_result(
            metric_id=metric_id,
            fiscal_year=fiscal_year,
            limit=limit,
            document_ids=document_ids,
        )[0]

    def top_metric_by_year_stats(
        self,
        *,
        metric_id: str,
        fiscal_year: int,
        limit: int,
        document_ids: Iterable[str] | None = None,
    ) -> dict[str, int]:
        return self._top_metric_by_year_result(
            metric_id=metric_id,
            fiscal_year=fiscal_year,
            limit=limit,
            document_ids=document_ids,
        )[1]

    def _top_metric_by_year_result(
        self,
        *,
        metric_id: str,
        fiscal_year: int,
        limit: int,
        document_ids: Iterable[str] | None,
    ) -> tuple[list[FinancialFact], dict[str, int]]:
        selected_document_ids = _dedupe_document_ids(document_ids)
        candidates = self._metric_candidates_by_year(
            metric_id=metric_id,
            fiscal_year=fiscal_year,
            document_ids=selected_document_ids,
        )
        annual_candidates: list[FinancialFact] = []
        excluded_non_annual_count = 0
        for fact in candidates:
            if _is_annual_fiscal_period(fact.fiscal_period):
                annual_candidates.append(fact)
            else:
                excluded_non_annual_count += 1

        cited_candidates = [fact for fact in annual_candidates if fact.has_page_quote_citation]

        deduped: list[FinancialFact] = []
        seen_companies: set[str] = set()
        excluded_duplicate_company_count = 0
        for fact in cited_candidates:
            company_key = _company_dedupe_key(fact)
            if company_key in seen_companies:
                excluded_duplicate_company_count += 1
                continue
            seen_companies.add(company_key)
            deduped.append(fact)

        returned = deduped[:limit]
        return returned, {
            "selected_docs": len(selected_document_ids) if selected_document_ids is not None else 0,
            "candidate_count": len(candidates),
            "verified_annual_candidates": len(annual_candidates),
            "after_citation_filter": len(cited_candidates),
            "after_company_dedupe": len(deduped),
            "excluded_non_annual_count": excluded_non_annual_count,
            "excluded_duplicate_company_count": excluded_duplicate_company_count,
            "returned_rows": len(returned),
        }

    def _metric_candidates_by_year(
        self,
        *,
        metric_id: str,
        fiscal_year: int,
        document_ids: list[str] | None,
    ) -> list[FinancialFact]:
        if document_ids == []:
            return []
        params: dict[str, object] = {
            "metric_id": metric_id,
            "fiscal_year": fiscal_year,
        }
        document_filter = ""
        if document_ids is not None:
            placeholders = []
            for index, document_id in enumerate(document_ids):
                param_name = f"document_id_{index}"
                placeholders.append(f":{param_name}")
                params[param_name] = document_id
            document_filter = f"AND document_id IN ({', '.join(placeholders)})"

        query = """
            SELECT * FROM financial_facts
            WHERE metric_id = :metric_id
              AND fiscal_year = :fiscal_year
              AND review_status = 'verified'
              AND normalized_value IS NOT NULL
              {document_filter}
            ORDER BY normalized_value DESC
        """.format(document_filter=document_filter)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_fact(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _fact_to_row(fact: FinancialFact) -> dict[str, object]:
    payload = fact.model_dump(mode="json")
    payload["source_path"] = str(fact.source_path)
    return payload


def _upsert_fact_rows(conn: sqlite3.Connection, facts: list[FinancialFact]) -> None:
    conn.executemany(_UPSERT_SQL, [_fact_to_row(fact) for fact in facts])


def _row_to_fact(row: sqlite3.Row) -> FinancialFact:
    return FinancialFact(
        fact_id=row["fact_id"],
        document_id=row["document_id"],
        company_name=row["company_name"],
        ticker=row["ticker"],
        source_path=Path(row["source_path"]),
        metric_id=row["metric_id"],
        metric_label=row["metric_label"],
        source_metric_name=row["source_metric_name"],
        period_type=row["period_type"],
        fiscal_period=row["fiscal_period"],
        fiscal_year=row["fiscal_year"],
        value=row["value"],
        unit_raw=row["unit_raw"],
        currency=row["currency"],
        scale=row["scale"],
        normalized_value=row["normalized_value"],
        normalized_unit=row["normalized_unit"],
        evidence_page=row["evidence_page"],
        evidence_quote=row["evidence_quote"],
        review_status=row["review_status"],
        source=row["source"],
        notes=row["notes"],
    )


def _is_annual_fiscal_period(fiscal_period: str) -> bool:
    normalized = _normalize_period_label(fiscal_period)
    if not normalized:
        return False
    if any(marker in normalized for marker in _NON_ANNUAL_PERIOD_MARKERS):
        return False
    return any(marker in normalized for marker in _ANNUAL_PERIOD_MARKERS) or bool(
        re.fullmatch(r"(?:19|20)\d{2}年?", normalized)
    )


def _company_dedupe_key(fact: FinancialFact) -> str:
    company_name = (fact.company_name or "").strip()
    if company_name:
        return f"company:{company_name.casefold()}"
    return f"document:{fact.document_id}"


def _normalize_period_label(value: str) -> str:
    return "".join(value.casefold().split()).replace("_", "-")


def _dedupe_document_ids(document_ids: Iterable[str] | None) -> list[str] | None:
    if document_ids is None:
        return None
    seen: set[str] = set()
    deduped: list[str] = []
    for document_id in document_ids:
        if document_id in seen:
            continue
        seen.add(document_id)
        deduped.append(document_id)
    return deduped


_ANNUAL_PERIOD_MARKERS = (
    "annualreport",
    "full-year",
    "fullyear",
    "yearended",
    "fortheyear",
    "\u5e74\u5ea6\u62a5\u544a",
    "\u5e74\u5ea6\u5831\u544a",
    "\u5e74\u62a5",
    "\u5e74\u5831",
    "\u5168\u5e74",
)

_NON_ANNUAL_PERIOD_MARKERS = (
    "q1",
    "q2",
    "q3",
    "quarter",
    "quarterly",
    "interim",
    "half-year",
    "halfyear",
    "sixmonths",
    "ninemonths",
    "threemonths",
    "1-3",
    "1-6",
    "1-9",
    "\u5b63\u5ea6",
    "\u4e00\u5b63",
    "\u4e09\u5b63",
    "\u524d\u4e09\u5b63",
    "\u4e2d\u671f",
    "\u534a\u5e74",
    "\u534a\u5e74\u5ea6",
    "1-3\u6708",
    "1-6\u6708",
    "1-9\u6708",
    "1\u81f33\u6708",
    "1\u81f36\u6708",
    "1\u81f39\u6708",
)
