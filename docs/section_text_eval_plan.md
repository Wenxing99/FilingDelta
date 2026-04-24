# Section Text Eval Plan

## Purpose

This document defines the next step after the `section_text` groundwork landed.

The immediate goal is **not** to switch the right-column chat to `section_text`.
The immediate goal is to answer one narrower question:

> Is the current `section_text` evidence layer good enough to justify controlled activation in document-scoped QA?

This should be answered offline first, before asking the user to do meaningful front-end testing.

## Current State

Already landed:

- `EvidenceKind / EvidenceUnit / EvidenceMetadata`
- `section_text` builder
- typed metadata in indexing / retrieval
- chat path still explicitly locked to `page_text`

Current constraint:

- do **not** let new evidence types silently change the existing document-scoped QA semantics
- do **not** replace `page_text` as the stable fallback / citation anchor
- do **not** mix this phase with `table_row`, query normalization, or `EvidenceQueryRouter`

## Scope

This eval only targets the narrative side of retrieval:

- strategy / outlook
- business review
- AI / digital transformation
- dividend policy wording
- risk / asset quality
- deep-page risk notes

This eval does **not** target:

- headline metrics
- table-row retrieval
- final answer synthesis quality
- external web search
- front-end UX

## Documents

Primary long-document set:

- `data/raw/招商银行2025年度报告.pdf`
- `data/raw/腾讯控股2025年度报告.pdf`

## Eval Questions

The `section_text` eval should use a narrative-focused subset instead of the full baseline metric-heavy set.

### Core Narrative Set

| id | document | query | expected_pages | expected_section_type |
|---|---|---|---|---|
| CMB-07 | 招行年报 | 年报哪些段落提到 AI First / 数智化转型？列出2-3个原文要点。 | `[44, 45]` | `strategy_outlook` |
| CMB-08A | 招行年报 | 招商银行如何管控房地产风险？ | `[47]` | `risk_asset_quality` |
| CMB-08B | 招行年报 | 招商银行如何管控地方政府隐性债务或地方债务风险？ | `[64]` | `risk_asset_quality` |
| CMB-08C | 招行年报 | 招商银行如何管控零售贷款或消费贷款风险？ | `[49, 50]` | `risk_asset_quality` |
| CMB-10 | 招行年报 | 招商银行在2026年前景展望中提出哪些经营重点或应对措施？ | `[68, 69]` | `strategy_outlook` |
| TCEHY-04 | 腾讯年报 | 年报将腾讯全年营销服务收入增长归因于哪些主要因素？ | `[9]` | `business_review` |
| TCEHY-05A | 腾讯年报 | 腾讯金融科技及企业服务年度收入增长的主要原因是什么？ | `[9]` | `business_review` |
| TCEHY-05B | 腾讯年报 | 腾讯金融科技及企业服务年度毛利增长的主要原因是什么？ | `[10]` | `business_review` |
| TCEHY-06A | 腾讯年报 | 腾讯视频号在内容生态或用户时长方面有哪些披露？ | `[6]` | `product_user_metrics` |
| TCEHY-06B | 腾讯年报 | 腾讯视频号对营销服务或广告增长有什么贡献？ | `[9, 10]` | `business_review` |
| TCEHY-08A | 腾讯年报 | 腾讯如何描述 AI 投入？ | `[5, 11]` | `strategy_outlook` |
| TCEHY-08B | 腾讯年报 | 腾讯如何描述混元模型？ | `[5, 6]` | `strategy_outlook` |
| TCEHY-08C | 腾讯年报 | 腾讯如何描述 AI 广告能力？ | `[6, 9, 13]` | `business_review` |
| TCEHY-09A | 腾讯年报 | 腾讯面临哪些市场风险？ | `[173, 174, 175, 176]` | `risk_asset_quality` |
| TCEHY-09B | 腾讯年报 | 腾讯面临哪些信贷风险？ | `[178, 179, 180, 181]` | `risk_asset_quality` |
| TCEHY-09C | 腾讯年报 | 腾讯面临哪些流动性风险？ | `[182, 183]` | `risk_asset_quality` |

### Hold-Out / Optional

These can be checked later, but they are not required for the first `section_text` go/no-go decision:

- `CMB-06`
  - hybrid dividend wording + metric extraction
- `TCEHY-07`
  - board dividend resolution wording plus numeric value
- `TCEHY-02`
  - user metric but often still table-like / summary-like

## Retrieval Modes To Compare

The offline eval should compare three retrieval shapes.

### Mode A: `page_text only`

Purpose:

- current stable baseline

Behavior:

- retrieve only `chunk_kind=page_text`
- keep current `document_id` filter

### Mode B: `section_text only`

Purpose:

- measure whether `section_text` is a useful retrieval object by itself

Behavior:

- retrieve only `chunk_kind=section_text`
- keep current `document_id` filter

### Mode C: `section_text first + page_text fallback`

Purpose:

- simulate the likely activation path without changing production chat yet

Behavior:

1. retrieve `section_text`
2. if results are empty, low-confidence, or obviously mismatched, fall back to `page_text`
3. keep `document_id` filter in both steps

## Stage 1: Evidence Quality Audit

Before comparing retrieval quality, inspect the evidence layer itself.

### Audit questions

1. Are `section_title` values readable?
2. Are there obvious junk titles?
3. Are key narrative sections present?
4. Are H-share / Traditional Chinese risk sections covered?
5. Is duplication excessive?

### Suggested audit output

Per document:

- total `section_text` count
- unique `section_title` count
- repeated-title count
- suspicious-title sample
- strong-title sample
- missing expected headings

### Suspicious-title examples to flag

- naked dates
- naked percentages
- page furniture / addresses
- truncated fragments
- repeated boilerplate chapter wrappers that drown out more specific headings

## Stage 2: Retrieval Comparison

For every question in the Core Narrative Set, record:

- retrieval mode
- top-k titles
- top-k pages
- expected page hit
- expected section-type hit
- whether fallback was needed

## Metrics

### Primary metrics

- `page_hit@k`
- `expected_section_type_hit@k`
- `fallback_needed_rate`
- `top1_title_quality`
- `noise_rate`

### Metric definitions

`page_hit@k`

- at least one retrieved unit comes from an expected physical page

`expected_section_type_hit@k`

- at least one retrieved unit has the expected normalized `section_type`

`fallback_needed_rate`

- percentage of questions where `section_text first` required `page_text` rescue

`top1_title_quality`

- simple qualitative label:
  - `strong`
  - `acceptable`
  - `weak`
  - `junk`

`noise_rate`

- percentage of audited `section_text` units that have obviously low-quality `section_title`

## Stage 3: Manual Spot Check

This should stay small and focused.

### Required spot checks

1. 招行 `AI First / 数智化`
2. 招行 `房地产 / 地方债 / 零售贷款风险`
3. 招行 `2026 展望`
4. 腾讯 `营销服务归因`
5. 腾讯 `金融科技及企业服务`
6. 腾讯 `视频号`
7. 腾讯 `市场风险 / 信贷风险 / 流动性风险`

### What to inspect

- does the retrieved object feel like the right section?
- is the title more helpful than a generic page chunk?
- does citation still land on a sensible page?
- would this be safe to show in chat if the answerer used it?

## Activation Gate

`section_text` should only be activated in chat if all of the following are true:

1. `section_text only` is meaningfully better than `page_text only` on narrative `expected_section_type_hit@k`
2. `section_text first + page_text fallback` does not materially regress `page_hit@k`
3. `noise_rate` is low enough that top results are not regularly polluted by junk titles
4. H-share / Traditional Chinese risk queries are covered well enough to avoid obvious blind spots
5. Manual spot checks say the retrieved object type is genuinely better, not just differently wrong

## Observed Results (2026-04-23)

Artifacts:

- Stage 1 audit: `data/outputs/eval/section_text_audit.json`
- Stage 2 retrieval comparison: `data/outputs/eval/section_text_retrieval_eval.json`

### Stage 1 audit summary

- 招行年报：`section_text=430`，noise rate `0.063`
- 腾讯年报：`section_text=238`，noise rate `0.063`
- 两份长文档的关键 heading groups 都已覆盖
- 但仍存在三类明显问题：
  - generic wrapper titles
  - fragment / truncated titles
  - repeated titles that drown out more specific local headings

### Stage 2 retrieval summary

- `page_text only`
  - `page_hit@6 = 16/16`
  - `expected_section_type_hit@6 = 0/16`
  - `top1_page_hit = 7/16`
- `section_text only`
  - `page_hit@6 = 16/16`
  - `expected_section_type_hit@6 = 14/16`
  - `top1_page_hit = 8/16`
  - `top1_section_type_hit = 10/16`
  - `top1_title_quality = strong 9 / acceptable 2 / weak 5`
- `section_text first + page_text fallback`
  - `page_hit@6 = 16/16`
  - `expected_section_type_hit@6 = 13/16`
  - `fallback_rate = 1/16 = 0.062`
  - `top1_page_hit = 7/16`
  - `top1_section_type_hit = 9/16`

### Key interpretation

- `section_text` is clearly more useful than `page_text` for narrative `section_type` matching.
- But the current builder is still too noisy for direct activation:
  - several top-1 hits are still chapter wrappers such as `第三章 管理层讨论与分析` / `管理層討論及分析`
  - Tencent still produces misleading wrappers such as `82 企業管治報告`
  - the current fallback heuristic is too blunt and can reduce `section_type` quality instead of helping
- So the current evidence says:
  - the direction is correct
  - the current implementation is not yet clean enough to become the default chat retrieval object

### Representative weak cases

- `CMB-08C`
  - top-1 title is still `第三章 管理层讨论与分析`
- `CMB-10`
  - top-1 drifts to a wrapper title on page `47` instead of the outlook pages
- `TCEHY-04`
  - top-1 is `管理層討論及分析`, which is still too generic to be a good chat-facing retrieval object
- `TCEHY-08B` / `TCEHY-08C`
  - top-1 becomes `82 企業管治報告`, which is the wrong local evidence shape for AI questions

## Current Decision

Current status is **Outcome A: Activate (controlled)**.

What changed after builder cleanup:

- generic wrapper demotion / stripping improved
- split-number and plain-heading support improved
- carry-over suppression reduced wrapper noise
- fallback logic became more conservative

Latest offline summary:

- 招行 audit noise rate: `0.052`
- 腾讯 audit noise rate: `0.168`
- `section_text only`: `page_hit@6 = 16/16`, `expected_section_type_hit@6 = 15/16`
- `section_text first + page_text fallback`: `page_hit@6 = 16/16`, `expected_section_type_hit@6 = 15/16`
- top-1 title quality on the narrative set is now `strong 8 / acceptable 8 / weak 0`

Residual issue:

- `TCEHY-08C` is still a taxonomy-overlap case between `business_review` and `product_user_metrics`
- this now looks more like a classification boundary issue than a retrieval-object failure

Current activation policy:

- document QA now allows a **controlled `section_text` activation**
- only narrative / risk / strategy / AI / product-metrics style questions should prefer `section_text`
- metric-heavy questions still default to `page_text`
- `page_text` remains the stable fallback and citation anchor

Immediate next step:

1. do a small smoke test of document QA with the new controlled activation path
2. verify citation quality on a few narrative queries
3. if stable, move to `table_row`

## Smoke Test Notes (2026-04-24)

Smoke setup:

- local demo started successfully with `./start_demo.sh`
- backend health passed at `http://127.0.0.1:8000/health`
- frontend root served successfully at `http://127.0.0.1:5173`
- the browser auto-opened and successfully requested demo documents from the backend

Analysis smoke:

- 招行完整年报 run succeeded
- artifact snapshot:
  - `total_pages = 350`
  - `chunk_count = 660`
  - `summary_sections_count = 5`
  - `verification_issues_count = 0`
  - review status passed

Chat smoke:

1. 招行：`招商银行如何管控房地产风险？`
   - citations landed on page `47` (and in one run also page `64`)
   - answer quality looked correct
   - but route classification was **not stable**:
     - one run returned `document_only`
     - another run returned `mixed_document_external`
   - this means the current router/planner still sometimes over-escalates a pure document question into unnecessary external search

2. 腾讯：`腾讯如何描述 AI 广告能力？`
   - route stayed `document_only`
   - retrieval mode stayed `semantic_with_filters`
   - citations landed on pages `5 / 6 / 11 / 15`
   - answer quality matched the expected narrative evidence shape

3. 腾讯：`腾讯2025年资本开支是多少？`
   - route stayed `document_only`
   - retrieval mode stayed `semantic_with_filters`
   - citation landed on page `18`
   - returned value: `79,198 百万元`

Current interpretation after smoke:

- controlled `section_text` activation is good enough to keep
- citation quality on narrative questions is materially better than before
- the next issue is no longer `section_text` builder quality
- the next issue is **router stability**, especially for document-only risk questions that should not trigger `mixed`

So the next implementation step after this smoke is:

1. tighten document-vs-mixed routing for pure document questions
2. keep `section_text` activation in place
3. only then move to `table_row`

## Decision Outcomes

### Outcome A: Activate

If the gate passes:

- add controlled `section_text` activation in document QA
- likely start with narrative / risk / strategy style questions only
- keep `page_text` fallback

### Outcome B: Revise builder first

If the gate fails because of title quality or missing sections:

- improve builder heuristics
- keep chat on `page_text`

### Outcome C: Delay activation

If `section_text` quality is mixed and the gain is small:

- keep the groundwork
- do not activate yet
- shift attention to `table_row` or query normalization only after this is explicitly judged

## Deliverables

The next concrete deliverables should be:

- a small local audit artifact summarizing `section_text` quality
- a local retrieval comparison artifact for Modes A/B/C
- a short decision note:
  - `activate`
  - `revise builder`
  - `delay`

## User Testing Guidance

At this stage, front-end testing is optional and low priority.

Reason:

- production chat is still intentionally locked to `page_text`
- the most important question is still offline:
  - whether `section_text` is worth activating at all

Meaningful front-end testing should happen **after** a controlled `section_text` activation path exists.
