from filingdelta.prompts.chat_router import CHAT_ROUTER_PROMPT


def test_chat_router_prompt_keeps_disclosure_risk_questions_document_only() -> None:
    prompt = CHAT_ROUTER_PROMPT.template

    assert "Do not route to mixed just because the question contains risk words." in prompt
    assert "how the filing says a company manages a risk is document_only" in prompt
    assert "招商银行如何管控房地产风险？ -> route=document_only" in prompt
    assert "needs_risk_reasoning=false" in prompt


def test_chat_router_prompt_keeps_external_reasoning_boundary_for_mixed() -> None:
    prompt = CHAT_ROUTER_PROMPT.template

    assert "usual implications" in prompt
    assert "what the filing facts might mean beyond the filing" in prompt
    assert "什么是净资产收益率？结合当前文档里的披露解释它说明什么。 -> route=mixed" in prompt
    assert "资本开支增加通常意味着什么？结合腾讯这份文档回答。 -> route=mixed" in prompt
