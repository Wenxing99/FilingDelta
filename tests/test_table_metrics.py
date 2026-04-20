from __future__ import annotations

from pathlib import Path

from filingdelta.ingestion.table_metrics import extract_table_headline_metrics
from filingdelta.schemas.filing import (
    FilingDocType,
    FilingDocument,
    FilingSource,
    Market,
    ParsedFiling,
    ParsedPage,
    ParserKind,
)


class _Selection:
    def __init__(self, page_numbers: list[int]) -> None:
        self._page_numbers = page_numbers

    def all_pages(self) -> list[int]:
        return self._page_numbers

    def pages_for(self, _field_name: str) -> list[int]:
        return self._page_numbers


def test_table_metrics_extracts_annual_summary_rows() -> None:
    parsed = _parsed_filing(
        """
        3.1 本集团主要会计数据和财务指标
        （人民币百万元，特别注明除外）
        2025年
        2024年
        本年比上年
        增减(%)
        2023年
        经营业绩
        营业收入
        337,532
        337,488
        0.01
        339,123
        归属于本行股东的净利润
        150,181
        148,391
        1.21
        146,602
        财务比率(%)
        归属于本行股东的平均总资产收益率
        1.19
        1.28
        下降0.09个百分点
        1.39
        归属于本行普通股股东的加权平均净资产收益率
        13.44
        14.49
        下降1.05个百分点
        16.22
        """,
        doc_type=FilingDocType.ANNUAL_REPORT,
    )

    result = extract_table_headline_metrics(
        source=_source(doc_type=FilingDocType.ANNUAL_REPORT),
        parsed_filing=parsed,
        selection=_Selection([1]),
    ).structured

    assert result.revenue.value == 337_532
    assert result.net_profit.value == 150_181
    assert result.roe.value == 13.44
    assert result.unit.value == "人民币百万元,特别注明除外"


def test_table_metrics_prefers_ytd_columns_for_quarterly_reports() -> None:
    parsed = _parsed_filing(
        """
        2 主要财务数据
        2.1 本集团主要会计数据及财务指标
        （人民币百万元，特别注明除外）
        报告期
        2025年
        7-9月
        2025年7-9月
        比上年同期
        增减(%)
        2025年
        1-9月
        2025年1-9月
        比上年同期
        增减(%)
        营业收入
        81,451
        2.11
        251,420
        -0.51
        归属于本行股东的净利润
        38,842
        1.04
        113,772
        0.52
        年化后归属于本行普通股股东的加权平均
        净资产收益率(%)
        14.44
        下降1.24个百分点
        13.96 下降1.42个百分点
        """,
        doc_type=FilingDocType.INTERIM_REPORT,
    )

    result = extract_table_headline_metrics(
        source=_source(
            doc_type=FilingDocType.INTERIM_REPORT,
            fiscal_period="2025年第三季度报告",
        ),
        parsed_filing=parsed,
        selection=_Selection([1]),
    ).structured

    assert result.revenue.value == 251_420
    assert result.net_profit.value == 113_772
    assert result.roe.value == 13.96


def test_table_metrics_does_not_use_roaa_as_roe() -> None:
    parsed = _parsed_filing(
        """
        主要会计数据和财务指标
        财务比率(%)
        2025年
        2024年
        归属于本行股东的平均总资产收益率
        1.19
        1.28
        下降0.09个百分点
        """,
        doc_type=FilingDocType.ANNUAL_REPORT,
    )

    result = extract_table_headline_metrics(
        source=_source(doc_type=FilingDocType.ANNUAL_REPORT),
        parsed_filing=parsed,
        selection=_Selection([1]),
    ).structured

    assert result.roe.value is None


def test_table_metrics_keeps_roe_empty_when_no_explicit_roe_row_exists() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            財務表現摘要
            未經審核
            截至下列日期止三個月
            二零二五年
            十二月三十一日
            二零二四年
            十二月三十一日
            同比變動
            二零二五年
            九月三十日
            環比變動
            （人民幣百萬元，另有指明者除外）
            收入
            194,371
            172,446
            13%
            192,869
            0.8%
            本公司權益持有人應佔盈利
            58,260
            51,324
            14%
            63,133
            -8%
            非國際財務報告準則本公司權益持有人應佔盈利
            64,694
            55,312
            17%
            70,551
            -8%
            """,
            """
            截至十二月三十一日止年度
            二零二五年
            二零二四年
            同比變動
            （人民幣百萬元，另有指明者除外）
            收入
            751,766
            660,257
            14%
            本公司權益持有人應佔盈利
            224,842
            194,073
            16%
            非國際財務報告準則本公司權益持有人應佔盈利
            259,626
            222,703
            17%
            """,
        ],
        doc_type=FilingDocType.ANNUAL_REPORT,
    )

    result = extract_table_headline_metrics(
        source=_source(
            company_name="腾讯控股",
            doc_type=FilingDocType.ANNUAL_REPORT,
            market=Market.H_SHARE,
        ),
        parsed_filing=parsed,
        selection=_Selection([1, 2]),
    ).structured

    assert result.revenue.value == 751_766
    assert result.net_profit.value == 224_842
    assert result.roe.value is None


def test_table_metrics_binds_annual_values_to_target_year_when_years_ascend() -> None:
    parsed = _parsed_filing(
        """
        簡明綜合全面收益表
        截至十二月三十一日止年度
        二零二一年
        人民幣百萬元
        二零二二年
        人民幣百萬元
        二零二三年
        人民幣百萬元
        二零二四年
        人民幣百萬元
        二零二五年
        人民幣百萬元
        收入
        560,118
        554,552
        609,015
        660,257
        751,766
        毛利
        245,944
        238,746
        293,109
        349,246
        422,593
        本公司權益持有人應佔盈利
        224,822
        188,243
        115,216
        194,073
        224,842
        """,
        doc_type=FilingDocType.ANNUAL_REPORT,
    )

    result = extract_table_headline_metrics(
        source=_source(
            company_name="腾讯控股",
            doc_type=FilingDocType.ANNUAL_REPORT,
            market=Market.H_SHARE,
        ),
        parsed_filing=parsed,
        selection=_Selection([1]),
    ).structured

    assert result.revenue.value == 751_766
    assert result.net_profit.value == 224_842
    assert result.roe.value is None


def _source(
    *,
    company_name: str = "招商银行股份有限公司",
    doc_type: FilingDocType,
    market: Market = Market.A_SHARE,
    fiscal_period: str = "2025年度报告",
) -> FilingSource:
    return FilingSource(
        source_path=Path("dummy.pdf"),
        company_name=company_name,
        market=market,
        doc_type=doc_type,
        fiscal_period=fiscal_period,
    )


def _parsed_filing(text: str, *, doc_type: FilingDocType) -> ParsedFiling:
    return _parsed_filing_pages([text], doc_type=doc_type)


def _parsed_filing_pages(texts: list[str], *, doc_type: FilingDocType) -> ParsedFiling:
    source_path = Path("dummy.pdf")
    return ParsedFiling(
        document=FilingDocument(
            document_id="dummy",
            company_name="招商银行股份有限公司",
            market=Market.A_SHARE,
            doc_type=doc_type,
            fiscal_period="2025年度报告",
            source_path=source_path,
            parser_kind=ParserKind.PYMUPDF,
            total_pages=len(texts),
        ),
        pages=[
            ParsedPage(
                page_number=index,
                text=_strip_test_text(text),
                markdown=_strip_test_text(text),
            )
            for index, text in enumerate(texts, start=1)
        ],
    )


def _strip_test_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines())
