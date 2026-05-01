from __future__ import annotations

from pathlib import Path

import pytest

from filingdelta.agents.chat_router_direct import (
    _build_direct_router_user_prompt,
    _build_direct_router_system_prompt,
    _chat_completions_url,
    infer_direct_router_document_evidence_intent,
    parse_direct_router_response,
)
from filingdelta.schemas.filing import FilingDocType, FilingDocument, Market, ParserKind


def _example_document() -> FilingDocument:
    return FilingDocument(
        document_id="doc-1",
        company_name="家电企业",
        ticker="000000",
        market=Market.A_SHARE,
        doc_type=FilingDocType.ANNUAL_REPORT,
        fiscal_period="2025",
        source_path=Path("sample.pdf"),
        parser_kind=ParserKind.PYMUPDF,
        total_pages=10,
    )


def test_parse_direct_router_response_validates_json_content() -> None:
    decision = parse_direct_router_response(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"route":"document_only",'
                            '"needs_external_background":false,'
                            '"needs_risk_reasoning":false,'
                            '"document_evidence_intent":"business_narrative",'
                            '"rationale":"Disclosure-only question."}'
                        )
                    }
                }
            ]
        }
    )

    assert decision.route == "document_only"
    assert decision.needs_external_background is False
    assert decision.needs_risk_reasoning is False
    assert decision.document_evidence_intent == "business_narrative"


def test_parse_direct_router_response_rejects_missing_content() -> None:
    with pytest.raises(ValueError, match="missing message content"):
        parse_direct_router_response({"choices": []})


def test_parse_direct_router_response_rejects_bad_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_direct_router_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "route=document_only",
                        }
                    }
                ]
            }
        )


def test_chat_completions_url_respects_custom_base_url() -> None:
    assert _chat_completions_url("https://example.test/v1/") == "https://example.test/v1/chat/completions"


def test_json_object_prompt_requests_exact_json_keys() -> None:
    prompt = _build_direct_router_system_prompt("json_object")

    assert "Return only a valid JSON object" in prompt
    assert "needs_external_background" in prompt
    assert "document_evidence_intent" in prompt


def test_direct_router_keeps_inventory_channel_efficiency_as_attribution() -> None:
    question = "家电企业存货或渠道库存是否异常？公司如何描述渠道效率？"

    assert infer_direct_router_document_evidence_intent(question) == "metric_attribution"
    assert infer_direct_router_document_evidence_intent("家电企业存货有什么变化？") is None
    assert infer_direct_router_document_evidence_intent("公司如何描述库存管理策略？") is None
    assert infer_direct_router_document_evidence_intent("公司如何描述供应链库存管理措施？") is None
    assert infer_direct_router_document_evidence_intent("腾讯如何描述 AI 广告能力？") is None

    prompt = _build_direct_router_user_prompt(
        question=question,
        document=_example_document(),
    )

    assert "Local deterministic intent hint" in prompt
    assert "- document_evidence_intent: metric_attribution" in prompt

    inventory_strategy_prompt = _build_direct_router_user_prompt(
        question="公司如何描述库存管理策略？",
        document=_example_document(),
    )
    assert "Local deterministic intent hint" not in inventory_strategy_prompt


def test_direct_router_system_prompt_documents_ha03_boundary() -> None:
    prompt = _build_direct_router_system_prompt("json_schema")

    assert "channel inventory, channel efficiency, and turnover" in prompt
    assert "Inventory management strategy or supply-chain inventory management measure" in prompt
    assert "家电企业存货或渠道库存是否异常？公司如何描述渠道效率？" in prompt
    assert "document_evidence_intent=metric_attribution" in prompt
    assert "腾讯如何描述 AI 广告能力？" in prompt
    assert "document_evidence_intent=business_narrative" in prompt
