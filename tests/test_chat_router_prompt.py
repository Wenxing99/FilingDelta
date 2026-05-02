from filingdelta.prompts.chat_answer import CHAT_ANSWER_PROMPT
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


def test_chat_router_prompt_classifies_document_evidence_intent() -> None:
    prompt = CHAT_ROUTER_PROMPT.template

    assert "document_evidence_intent" in prompt
    assert "metric_value" in prompt
    assert "metric_attribution" in prompt
    assert "business_narrative" in prompt
    assert "Metric words do not automatically mean metric_value" in prompt
    assert "腾讯2025年营销服务收入增长的主要原因是什么？ -> route=document_only" in prompt
    assert "document_evidence_intent=metric_attribution" in prompt


def test_chat_router_prompt_keeps_inventory_channel_efficiency_as_attribution() -> None:
    prompt = CHAT_ROUTER_PROMPT.template

    assert "inventory, channel inventory, channel efficiency, or turnover" in prompt
    assert "家电企业存货或渠道库存是否异常？公司如何描述渠道效率？" in prompt
    assert (
        "家电企业存货或渠道库存是否异常？公司如何描述渠道效率？ -> route=document_only, "
        "needs_external_background=false, needs_risk_reasoning=false, "
        "document_evidence_intent=metric_attribution"
    ) in prompt
    assert (
        "腾讯如何描述 AI 广告能力？ -> route=document_only, "
        "needs_external_background=false, needs_risk_reasoning=false, "
        "document_evidence_intent=business_narrative"
    ) in prompt


def test_chat_answer_prompt_preserves_customer_deposit_ratio_numbers() -> None:
    prompt = CHAT_ANSWER_PROMPT.template

    assert "Numeric completeness" in prompt
    assert "customer deposits" in prompt
    assert "deposit structure" in prompt
    assert "demand-deposit/current-deposit ratio" in prompt
    assert "daily-average balance share" in prompt
    assert "percentage-point change" in prompt
    assert "客户存款" in prompt
    assert "活期存款占比" in prompt
    assert "日均余额占比" in prompt
