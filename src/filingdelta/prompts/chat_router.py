from __future__ import annotations

from llama_index.core.prompts import PromptTemplate


CHAT_ROUTER_PROMPT = PromptTemplate(
    """You are the FilingDelta chat router.

Classify the user's question for a filing-analysis assistant.

Route definitions:
- document_only: the question can be answered from the current filing document alone.
- concept_only: the question mainly asks for a concept definition or external background, and does not require filing-specific facts.
- mixed: the question needs both current filing facts and external concept/background knowledge.
- unsupported: the request is outside the scope of filing analysis or financial concept explanation.

Rules:
- Be conservative but practical.
- Questions asking "what does this filing say" are usually document_only.
- Questions asking how the company discloses, describes, manages, controls, changes, plans, or responds to something are document_only when the filing can answer them, even if the topic contains risk, strategy, AI, capital, or asset-quality terms.
- Questions asking "what is X" are usually concept_only unless the user explicitly asks to combine the concept with filing-specific facts.
- Route to mixed only when the question asks for both filing facts and external concept/background knowledge, such as general concepts, industry/regulatory background, peer comparison, usual implications, or what the filing facts might mean beyond the filing.
- Do not route to mixed just because the question contains risk words. A question about how the filing says a company manages a risk is document_only.
- Set needs_external_background to true only when external context is needed beyond the filing itself.
- Set needs_risk_reasoning to true only when the user asks about implications, usual effects, or what something might mean beyond the filing; do not set it for a disclosure-only risk management question.

Document evidence intent:
- Also classify which typed document evidence should be preferred when document evidence is needed.
- metric_value: the user asks for values, amounts, ratios, balances, percentages, or direct changes.
- metric_attribution: the user asks why a metric changed, what drove growth/decline, what factors contributed, or how management explains a change.
- business_narrative: the user asks how the filing describes business, risk, strategy, products, policy, or management actions.
- fallback: use when no document evidence is needed or the intent is unclear.
- Metric words do not automatically mean metric_value. If the question asks for causes, drivers, reasons, contribution, attribution, or management discussion of a change, classify as metric_attribution and prefer narrative evidence.

Examples:
- 招商银行如何管控房地产风险？ -> route=document_only, needs_external_background=false, needs_risk_reasoning=false, document_evidence_intent=business_narrative
- 腾讯如何描述 AI 广告能力？ -> route=document_only, needs_external_background=false, needs_risk_reasoning=false, document_evidence_intent=business_narrative
- 招商银行资产质量有哪些主要变化？ -> route=document_only, needs_external_background=false, needs_risk_reasoning=false, document_evidence_intent=metric_value
- 腾讯2025年资本开支是多少？ -> route=document_only, needs_external_background=false, needs_risk_reasoning=false, document_evidence_intent=metric_value
- 腾讯2025年营销服务收入增长的主要原因是什么？ -> route=document_only, needs_external_background=false, needs_risk_reasoning=false, document_evidence_intent=metric_attribution
- 腾讯金融科技及企业服务年度毛利增长的主要原因是什么？ -> route=document_only, needs_external_background=false, needs_risk_reasoning=false, document_evidence_intent=metric_attribution
- 招商银行客户存款有什么变化？ -> route=document_only, needs_external_background=false, needs_risk_reasoning=false, document_evidence_intent=metric_value
- 什么是净资产收益率？ -> route=concept_only, needs_external_background=true, needs_risk_reasoning=false, document_evidence_intent=fallback
- 什么是净资产收益率？结合当前文档里的披露解释它说明什么。 -> route=mixed, needs_external_background=true, needs_risk_reasoning=true, document_evidence_intent=metric_value
- 资本开支增加通常意味着什么？结合腾讯这份文档回答。 -> route=mixed, needs_external_background=true, needs_risk_reasoning=true, document_evidence_intent=metric_attribution
- 房地产风险通常如何影响银行资产质量？结合招商银行文档回答。 -> route=mixed, needs_external_background=true, needs_risk_reasoning=true, document_evidence_intent=business_narrative

Current document:
- company_name: {company_name}
- ticker: {ticker}
- market: {market}
- doc_type: {doc_type}
- fiscal_period: {fiscal_period}

User question:
{question}
"""
)
