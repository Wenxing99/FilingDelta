from __future__ import annotations

from pathlib import Path

from filingdelta.financial_facts import FinancialFactsQueryService, SQLiteFinancialFactStore
from filingdelta.financial_facts.schemas import FinancialFact
from filingdelta.services.kb_financial_facts import (
    KbFinancialFactsChatService,
    parse_kb_metric_rank_question,
)


def test_parse_kb_metric_rank_question_supports_revenue_and_assets() -> None:
    revenue = parse_kb_metric_rank_question("2025年哪三家公司营业收入最高？")
    assets = parse_kb_metric_rank_question("2025年总资产最高的三家公司是谁？")

    assert revenue.recognized is True
    assert revenue.metric_id == "revenue"
    assert revenue.fiscal_year == 2025
    assert revenue.limit == 3
    assert assets.metric_id == "total_assets"
    assert assets.limit == 3


def test_parse_kb_metric_rank_question_supports_kb_topk_without_company_scope() -> None:
    cases = {
        "current-KB revenue Top3": ("revenue", None, 3),
        "2025 revenue Top3 in current KB": ("revenue", 2025, 3),
        "2025年当前KB营业收入Top3": ("revenue", 2025, 3),
        "2025年营业收入Top3": ("revenue", 2025, 3),
        "2025 total assets Top3 in current KB": ("total_assets", 2025, 3),
        "2025 total liabilities Top3 in current KB": ("total_liabilities", 2025, 3),
        "2025 net profit attributable Top3 in current KB": (
            "net_profit_attributable",
            2025,
            3,
        ),
        "2025 attributable net profit Top3 in fact store": (
            "net_profit_attributable",
            2025,
            3,
        ),
    }

    for question, (metric_id, fiscal_year, limit) in cases.items():
        parsed = parse_kb_metric_rank_question(question)

        assert parsed.recognized is True, question
        assert parsed.metric_id == metric_id, question
        assert parsed.fiscal_year == fiscal_year, question
        assert parsed.limit == limit, question


def test_parse_kb_metric_rank_question_treats_cross_company_highest_as_top1() -> None:
    cases = {
        "2025年归母净利润最高是哪家企业？": "net_profit_attributable",
        "2025年总资产最高是哪家公司？": "total_assets",
    }

    for question, metric_id in cases.items():
        parsed = parse_kb_metric_rank_question(question)

        assert parsed.recognized is True, question
        assert parsed.metric_id == metric_id, question
        assert parsed.fiscal_year == 2025, question
        assert parsed.limit == 1, question
        assert parsed.unsupported_reason is None, question


def test_parse_kb_metric_rank_question_does_not_intercept_document_highest_business() -> None:
    questions = (
        "2025年招商银行营业收入最高的业务是什么？",
        "这份报告里营业收入最高的业务是什么？",
    )

    for question in questions:
        parsed = parse_kb_metric_rank_question(question)

        assert parsed.recognized is False, question


def test_parse_kb_metric_rank_question_rejects_unsupported_year_without_rag_fallback() -> None:
    parsed = parse_kb_metric_rank_question("2024年哪三家公司营业收入最高？")

    assert parsed.recognized is True
    assert parsed.unsupported_reason is not None
    assert "2025" in parsed.unsupported_reason


def test_kb_financial_facts_unsupported_year_returns_no_rag_fallback(tmp_path: Path) -> None:
    service = KbFinancialFactsChatService(FinancialFactsQueryService(tmp_path / "missing.sqlite"))

    answer = service.answer_if_supported(
        document_id="ui-doc",
        session_id="session",
        question="2024年哪三家公司营业收入Top3？",
    )

    assert answer is not None
    assert answer.route == "unsupported"
    assert answer.retrieval_mode == "kb_financial_facts"
    assert answer.citations == []


def test_kb_financial_facts_answer_has_no_clickable_citations(tmp_path: Path) -> None:
    db_path = tmp_path / "facts.sqlite"
    store = SQLiteFinancialFactStore(db_path)
    store.upsert_facts(
        [
            _fact("doc-a", "A", 300),
            _fact("doc-b", "B", 200),
            _fact("doc-c", "C", 100),
        ]
    )
    service = KbFinancialFactsChatService(FinancialFactsQueryService(db_path))

    answer = service.answer_if_supported(
        document_id="ui-doc",
        session_id="session",
        question="2025年哪三家公司营业收入最高？",
    )

    assert answer is not None
    assert answer.route == "document_only"
    assert answer.retrieval_mode == "kb_financial_facts"
    assert answer.citations == []
    assert "doc-a" in answer.answer
    assert "第 8 页" in answer.answer


def test_kb_financial_facts_answer_formats_quote_for_markdown_table(tmp_path: Path) -> None:
    db_path = tmp_path / "facts.sqlite"
    store = SQLiteFinancialFactStore(db_path)
    store.upsert_facts(
        [
            _fact("doc-a", "A", 300, evidence_quote="資產總額\n2,038,986"),
            _fact("doc-b", "B", 200, evidence_quote="收入 | 成本"),
        ]
    )
    service = KbFinancialFactsChatService(FinancialFactsQueryService(db_path))

    answer = service.answer_if_supported(
        document_id="ui-doc",
        session_id="session",
        question="2025 revenue Top2 in current KB",
    )

    assert answer is not None
    assert "資產總額 2,038,986" in answer.answer
    assert "資產總額\n2,038,986" not in answer.answer
    assert "收入 \\| 成本" in answer.answer


def test_kb_financial_facts_unsupported_question_returns_answer_not_none(tmp_path: Path) -> None:
    service = KbFinancialFactsChatService(FinancialFactsQueryService(tmp_path / "missing.sqlite"))

    answer = service.answer_if_supported(
        document_id="ui-doc",
        session_id="session",
        question="2025年经营现金流Top3",
    )

    assert answer is not None
    assert answer.route == "unsupported"
    assert answer.retrieval_mode == "kb_financial_facts"


def test_kb_financial_facts_non_rank_question_is_not_intercepted(tmp_path: Path) -> None:
    service = KbFinancialFactsChatService(FinancialFactsQueryService(tmp_path / "missing.sqlite"))

    answer = service.answer_if_supported(
        document_id="ui-doc",
        session_id="session",
        question="招商银行营业收入是多少？",
    )

    assert answer is None


def _fact(
    document_id: str,
    company_name: str,
    normalized_value: float,
    *,
    evidence_quote: str = "营业收入 100",
) -> FinancialFact:
    return FinancialFact(
        fact_id=f"{document_id}:revenue",
        document_id=document_id,
        company_name=company_name,
        source_path=Path(f"{document_id}.pdf"),
        metric_id="revenue",
        metric_label="营业收入",
        source_metric_name="revenue",
        period_type="period",
        fiscal_period="2025 annual report",
        fiscal_year=2025,
        value=normalized_value / 1_000_000,
        unit_raw="人民币百万元",
        currency="CNY",
        scale=1_000_000,
        normalized_value=normalized_value,
        normalized_unit="CNY",
        evidence_page=8,
        evidence_quote=evidence_quote,
        review_status="verified",
    )
