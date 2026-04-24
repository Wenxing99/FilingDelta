from __future__ import annotations

from pathlib import Path

from filingdelta.ingestion.chunking import build_chunks
from filingdelta.ingestion.evidence_builder import build_evidence_units
from filingdelta.ingestion.section_evidence import build_section_evidence
from filingdelta.retrieval.indexer import evidence_to_node
from filingdelta.schemas.filing import (
    EvidenceKind,
    FilingDocType,
    FilingDocument,
    Market,
    ParsedFiling,
    ParsedPage,
    ParserKind,
)


def test_build_section_evidence_extracts_numbered_sections_and_types() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            招商银行股份有限公司
            2025年度报告（A股）
            43
            第三章 管理层讨论与分析
            3.8.5 践行AI First，数智化转型全面提速
            本公司锚定守正创新，坚定实施科技兴行战略，通过科技赋能，以“线上化、数据化、智能化、平台化、生态化”为演进方向，
            加速“数智招行”建设；落实“人工智能+”行动，提出“AI First”理念，坚定执行AI“优先”“领先”“率先”，全面拥抱以大模型
            为代表的新一代人工智能革命，做好数字金融大文章。报告期内，本公司信息科技投入129.01亿元，达到本公司营业收入的4.31%。
            3.8.6 巩固堡垒式的全面风险与合规管理体系
            报告期内，本公司坚持风险为本、合规优先，统筹发展与安全，持续巩固堡垒式的风险合规管理体系。
            持续防范化解重点领域风险，积极应对房地产、地方政府隐性债务、零售贷款、表外业务等领域的风险挑战，
            加强风险前瞻排查，动态调整风控策略。全面加强合规管理，深入开展“合规履职年”活动。
            """,
        ],
        company_name="招商银行",
    )

    units = build_section_evidence(parsed)

    titles = [unit.metadata.section_title for unit in units]
    assert "3.8.5 践行AI First，数智化转型全面提速" in titles
    assert "3.8.6 巩固堡垒式的全面风险与合规管理体系" in titles

    ai_unit = next(unit for unit in units if unit.metadata.section_title == "3.8.5 践行AI First，数智化转型全面提速")
    risk_unit = next(unit for unit in units if unit.metadata.section_title == "3.8.6 巩固堡垒式的全面风险与合规管理体系")

    assert ai_unit.metadata.chunk_kind == EvidenceKind.SECTION_TEXT
    assert ai_unit.metadata.section_type == "strategy_outlook"
    assert risk_unit.metadata.section_type == "risk_asset_quality"


def test_build_section_evidence_extracts_inline_topic_heading() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            騰訊控股有限公司
            8
            管理層討論及分析
            收入。截至二零二五年十二月三十一日止年度的收入同比增長14% 至人民幣7,518 億元。下表載列本集團截至
            二零二五年及二零二四年十二月三十一日止年度按分部劃分的收入。增值服務業務截至二零二五年十二月三十一日止年度
            的收入同比增長16% 至人民幣3,693 億元。本土市場遊戲收入為人民幣1,642 億元，同比增長18%。
            """,
        ],
        company_name="腾讯控股",
        market=Market.H_SHARE,
    )

    units = build_section_evidence(parsed)

    assert len(units) == 1
    unit = units[0]
    assert unit.metadata.section_title == "收入"
    assert unit.metadata.section_type == "financial_summary"
    assert "收入。截至二零二五年十二月三十一日止年度的收入同比增長14%" in unit.text


def test_build_evidence_units_includes_page_and_section_text() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            第三章 管理层讨论与分析
            3.8.5 践行AI First，数智化转型全面提速
            本公司锚定守正创新，坚定实施科技兴行战略，通过科技赋能，加速“数智招行”建设。
            报告期内，本公司信息科技投入129.01亿元，达到本公司营业收入的4.31%。
            """,
        ],
        company_name="招商银行",
    )
    chunks = build_chunks(parsed, chunk_size=200, chunk_overlap=20)

    evidence_units = build_evidence_units(parsed_filing=parsed, chunks=chunks)

    kinds = {unit.metadata.chunk_kind for unit in evidence_units}
    assert EvidenceKind.PAGE_TEXT in kinds
    assert EvidenceKind.SECTION_TEXT in kinds

    section_unit = next(unit for unit in evidence_units if unit.metadata.chunk_kind == EvidenceKind.SECTION_TEXT)
    node = evidence_to_node(section_unit, document_id=parsed.document.document_id)

    assert node.metadata["chunk_kind"] == EvidenceKind.SECTION_TEXT.value
    assert node.metadata["section_title"] == "3.8.5 践行AI First，数智化转型全面提速"
    assert node.metadata["section_type"] == "strategy_outlook"


def test_build_section_evidence_supports_traditional_single_number_heading() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            騰訊控股有限公司
            173
            4 財務風險管理
            本集團的業務面臨多種財務風險，包括市場風險、信貸風險及流動性風險。
            我們持續完善風險管理框架，監察利率、匯率及信用暴露，並維持審慎的流動性管理。
            4.1 信貸風險
            我們根據交易對手、資產類型及減值政策管理信貸風險暴露。
            """,
        ],
        company_name="腾讯控股",
        market=Market.H_SHARE,
    )

    units = build_section_evidence(parsed)

    titles = [unit.metadata.section_title for unit in units]
    assert "4 財務風險管理" in titles
    risk_unit = next(unit for unit in units if unit.metadata.section_title == "4 財務風險管理")
    assert risk_unit.metadata.section_type == "risk_asset_quality"


def test_build_section_evidence_splits_multiple_inline_topics_on_one_page() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            騰訊控股有限公司
            10
            管理層討論及分析
            銷售及市場推廣開支。截至二零二五年十二月三十一日止年度的銷售及市場推廣開支同比增長15%，
            反映為支持我們的AI 原生應用程序及遊戲的發展而加大推廣力度，並持續提升品牌推廣效率。
            一般及行政開支。截至二零二五年十二月三十一日止年度的一般及行政開支同比增長21%，主要由於研發開支增加，
            包括與AI 投資相關的僱員成本及折舊費用增加，亦包括若干一次性股份酬金開支。
            利息收入。截至二零二五年十二月三十一日止年度的利息收入同比增長6%，乃由於現金儲備增加，
            並受益於更穩健的資金管理和利率環境。
            財務成本。截至二零二五年十二月三十一日止年度的財務成本增加，主要由於確認滙兌虧損淨額，
            同時反映若干融資安排帶來的成本上升。
            """,
        ],
        company_name="腾讯控股",
        market=Market.H_SHARE,
    )

    units = build_section_evidence(parsed)
    titles = [unit.metadata.section_title for unit in units]

    assert "銷售及市場推廣開支" in titles
    assert "一般及行政開支" in titles
    assert "利息收入" in titles
    assert "財務成本" in titles


def test_build_section_evidence_prefers_plain_subheading_over_generic_wrapper() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            第三章 管理层讨论与分析
            消费信贷类业务风险管控
            本公司围绕国家政策鼓励的升级型消费场景及个人或家庭综合消费场景开展消费信贷类业务，
            坚持聚焦优质客户，加强客户和区域的差异化经营，构建风险收益比更优的资产结构，稳健发展消费贷款业务。
            后续，本公司将持续完善消费信贷类业务的精细化风险管控策略，强化贷前准入和贷后监测。
            """,
        ],
        company_name="招商银行",
    )

    units = build_section_evidence(parsed)
    titles = [unit.metadata.section_title for unit in units]

    assert "消费信贷类业务风险管控" in titles
    assert "第三章 管理层讨论与分析" not in titles


def test_build_section_evidence_supports_split_number_heading() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            騰訊控股有限公司
            107
            企業管治報告
            本公司持續加大技術投入，利用混元基礎模型等AI 核心技術提升產品智能化水平。
            2.
            組織及人才管理風險
            在技術快速迭代、行業競爭加劇以及AI 技術引發顛覆性變革的背景下，
            高素質人才儲備與敏捷高效的組織能力，是公司保持長期競爭力的關鍵。
            """,
        ],
        company_name="腾讯控股",
        market=Market.H_SHARE,
    )

    units = build_section_evidence(parsed)
    titles = [unit.metadata.section_title for unit in units]

    assert "2. 組織及人才管理風險" in titles


def test_build_section_evidence_ignores_frame_wrapper_and_extracts_colon_heading() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            82 企業管治報告
            主席報告
            以下為二零二五年我們主要產品及服務的重點表現：
            視頻號受益於升級的內容推薦算法及更豐富的內容生態，總用戶使用時長同比增長超過20%。
            我們升級了廣告技術的基礎模型，並推出了智能投放產品矩陣騰訊廣告AIM+。
            我們的混元基礎模型在多模態能力方面成為了行業領導者。
            """,
        ],
        company_name="腾讯控股",
        market=Market.H_SHARE,
    )

    units = build_section_evidence(parsed)
    titles = [unit.metadata.section_title for unit in units]

    assert "82 企業管治報告" not in titles
    assert "主席報告" not in titles
    assert "主要產品及服務的重點表現" in titles


def test_build_section_evidence_classifies_operating_metrics_as_product_user_metrics() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            經營資料
            微信及WeChat 的合併月活躍賬戶數為1,418，QQ 的移動終端月活躍賬戶數為508。
            視頻號受益於升級的內容推薦算法及更豐富的內容生態，總用戶使用時長同比增長超過20%。
            收費增值服務訂閱會員數為267。
            """,
        ],
        company_name="腾讯控股",
        market=Market.H_SHARE,
    )

    units = build_section_evidence(parsed)
    metrics_unit = next(unit for unit in units if unit.metadata.section_title == "經營資料")

    assert metrics_unit.metadata.section_type == "product_user_metrics"


def test_build_section_evidence_skips_generic_carry_over_without_local_heading() -> None:
    parsed = _parsed_filing_pages(
        [
            """
            管理層討論及分析
            收入。截至二零二五年十二月三十一日止年度的收入同比增長14%。
            營銷服務業務收入同比增長，主要受益於視頻號及廣告技術提升。
            """,
            """
            本集團持續優化內部流程並提升信息披露質量。
            本頁沒有新的局部小標題，只是 generic wrapper 的自然延續。
            """,
        ],
        company_name="腾讯控股",
        market=Market.H_SHARE,
    )

    units = build_section_evidence(parsed)
    pages = [unit.metadata.page_number for unit in units]

    assert 1 in pages
    assert 2 not in pages


def _parsed_filing_pages(
    pages: list[str],
    *,
    company_name: str,
    market: Market = Market.A_SHARE,
    doc_type: FilingDocType = FilingDocType.ANNUAL_REPORT,
) -> ParsedFiling:
    return ParsedFiling(
        document=FilingDocument(
            document_id="demo_doc",
            company_name=company_name,
            market=market,
            doc_type=doc_type,
            source_path=Path("/tmp/demo.pdf"),
            parser_kind=ParserKind.PYMUPDF,
            total_pages=len(pages),
        ),
        pages=[
            ParsedPage(
                page_number=index,
                text=_clean_page_text(page_text),
                markdown=_clean_page_text(page_text),
            )
            for index, page_text in enumerate(pages, start=1)
        ],
    )


def _clean_page_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines())
