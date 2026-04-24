from __future__ import annotations

import pytest

from filingdelta.agents.chat_router_direct import (
    _build_direct_router_system_prompt,
    _chat_completions_url,
    parse_direct_router_response,
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
