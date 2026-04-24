from __future__ import annotations

from pathlib import Path

from filingdelta.ingestion.chunking import build_chunks
from filingdelta.ingestion.evidence_builder import build_evidence_units
from filingdelta.ingestion.table_row_evidence import build_table_row_evidence
from filingdelta.retrieval.indexer import evidence_to_node
from filingdelta.schemas.filing import (
    EvidenceKind,
    FilingDocument,
    Market,
    ParsedFiling,
    ParsedPage,
    ParserKind,
)


def test_build_table_row_evidence_extracts_customer_deposit_rows() -> None:
    parsed = _parsed_filing(
        """
        客户存款
        2025年12月31日
        2024年12月31日
        较上年末增减
        客户存款总额
        98,361.30
        90,962.00
        8.13%
        活期存款占比
        50.79%
        52.24%
        下降1.45个百分点
        公司客户存款
        53,402.16
        50,646.00
        5.44%
        """,
        fiscal_period="2025年度报告",
    )

    units = build_table_row_evidence(parsed)

    deposit_unit = next(unit for unit in units if unit.metadata.row_label == "客户存款")
    demand_unit = next(unit for unit in units if unit.metadata.row_label == "活期存款")

    assert deposit_unit.metadata.chunk_kind == EvidenceKind.TABLE_ROW
    assert deposit_unit.metadata.metric_tags == ["customer_deposits", "deposits"]
    assert deposit_unit.metadata.period_hint == "fy2025"
    assert "98,361.30" in deposit_unit.text
    assert "8.13%" in deposit_unit.text
    assert "50.79%" in demand_unit.text


def test_build_evidence_units_includes_table_row_evidence() -> None:
    parsed = _parsed_filing(
        """
        主要会计数据和财务指标
        2025年
        2024年
        营业收入
        337,532
        337,488
        归属于本行股东的净利润
        150,181
        148,391
        """,
        fiscal_period="2025年度报告",
    )
    chunks = build_chunks(parsed, chunk_size=200, chunk_overlap=20)

    evidence_units = build_evidence_units(parsed_filing=parsed, chunks=chunks)

    kinds = {unit.metadata.chunk_kind for unit in evidence_units}
    assert EvidenceKind.PAGE_TEXT in kinds
    assert EvidenceKind.TABLE_ROW in kinds


def test_table_row_evidence_node_carries_metric_metadata() -> None:
    parsed = _parsed_filing(
        """
        财务比率(%)
        2025年
        2024年
        归属于本行普通股股东的加权平均净资产收益率
        13.44
        14.49
        """,
        fiscal_period="2025年度报告",
    )
    unit = next(unit for unit in build_table_row_evidence(parsed) if unit.metadata.row_label == "净资产收益率")

    node = evidence_to_node(unit, document_id=parsed.document.document_id)

    assert node.metadata["chunk_kind"] == EvidenceKind.TABLE_ROW.value
    assert node.metadata["row_label"] == "净资产收益率"
    assert node.metadata["metric_tags"] == ["roe", "profitability_ratio"]
    assert node.metadata["period_hint"] == "fy2025"


def _parsed_filing(text: str, *, fiscal_period: str) -> ParsedFiling:
    source_path = Path("dummy.pdf")
    return ParsedFiling(
        document=FilingDocument(
            document_id="doc-test",
            company_name="招商银行股份有限公司",
            market=Market.A_SHARE,
            fiscal_period=fiscal_period,
            source_path=source_path,
            parser_kind=ParserKind.PYMUPDF,
            total_pages=1,
        ),
        pages=[
            ParsedPage(
                page_number=1,
                text=_strip_test_text(text),
                markdown=_strip_test_text(text),
            )
        ],
    )


def _strip_test_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines())
