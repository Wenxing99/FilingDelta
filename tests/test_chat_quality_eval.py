from filingdelta.eval.chat_quality import CHAT_QUALITY_CASES, evaluate_chat_quality
from filingdelta.schemas.chat import ChatAnswer, ChatAnswerSection, ChatCitation


def test_chat_quality_passes_customer_deposit_group_level_answer() -> None:
    case = _case("cmb-customer-deposits-quality")
    answer = ChatAnswer(
        document_id="doc-cmb",
        session_id="session-test",
        question=case.question,
        route="document_only",
        answer=(
            "招商银行客户存款余额为98361.30 亿元，较上年末增长8.13%。"
            "活期存款占比为50.79%，较上年末下降。"
        ),
        sections=[
            ChatAnswerSection(
                section_type="document_evidence",
                title="文档证据",
                items=["第30页披露客户存款余额、活期占比和同比变化。"],
            )
        ],
        citations=[
            ChatCitation(citation_id="doc-0", page_number=30, quote="客户存款余额"),
            ChatCitation(citation_id="doc-1", page_number=47, quote="活期占比"),
        ],
    )

    result = evaluate_chat_quality(case=case, answer=answer, wall_ms=100)

    assert result.passed


def test_chat_quality_fails_customer_deposit_answer_without_group_level_terms() -> None:
    case = _case("cmb-customer-deposits-quality")
    answer = ChatAnswer(
        document_id="doc-cmb",
        session_id="session-test",
        question=case.question,
        route="document_only",
        answer="招商银行公司客户存款余额为51,953.62亿元，较上年末增长4.90%。",
        citations=[
            ChatCitation(citation_id="doc-0", page_number=54, quote="公司客户存款"),
            ChatCitation(citation_id="doc-1", page_number=47, quote="客户存款"),
        ],
    )

    result = evaluate_chat_quality(case=case, answer=answer, wall_ms=100)

    assert not result.passed
    assert "required_answer_terms" in {check.check_id for check in result.failed_checks}


def test_chat_quality_fails_empty_parentheses_and_internal_markers() -> None:
    case = _case("cmb-roe-mixed-quality")
    answer = ChatAnswer(
        document_id="doc-cmb",
        session_id="session-test",
        question=case.question,
        route="mixed",
        answer="净资产收益率（DOC_1、WEB_1）为13.44%，用于衡量股东权益回报。（）",
        citations=[
            ChatCitation(citation_id="doc-0", page_number=19, quote="净资产收益率"),
            ChatCitation(
                citation_id="external-0",
                source_type="external",
                url="https://example.test/roe",
                title="ROE",
            ),
        ],
    )

    result = evaluate_chat_quality(case=case, answer=answer, wall_ms=100)

    failed = {check.check_id for check in result.failed_checks}
    assert "empty_citation_markers" in failed
    assert "internal_markers" in failed


def test_chat_quality_fails_duplicate_citations() -> None:
    case = _case("cmb-roe-mixed-quality")
    answer = ChatAnswer(
        document_id="doc-cmb",
        session_id="session-test",
        question=case.question,
        route="mixed",
        answer="净资产收益率为13.44%，说明股东权益回报水平。",
        citations=[
            ChatCitation(citation_id="doc-0", page_number=19, quote="净资产收益率"),
            ChatCitation(citation_id="doc-1", page_number=19, quote="同比下降"),
            ChatCitation(
                citation_id="external-0",
                source_type="external",
                url="https://example.test/roe",
                title="ROE",
            ),
            ChatCitation(
                citation_id="external-1",
                source_type="external",
                url="https://example.test/roe",
                title="ROE duplicate",
            ),
        ],
    )

    result = evaluate_chat_quality(case=case, answer=answer, wall_ms=100)

    failed = {check.check_id for check in result.failed_checks}
    assert "duplicate_document_pages" in failed
    assert "duplicate_external_urls" in failed


def _case(case_id: str):
    return next(case for case in CHAT_QUALITY_CASES if case.case_id == case_id)
