# Retrieval Eval Plan

## Purpose

This document captures the next implementation step after the chunking research and
golden-query reviews.

The immediate goal is **not** to change chunking yet. The immediate goal is to build
a repeatable baseline retrieval eval for the current implementation:

`local parse -> page-scoped SentenceSplitter chunks -> Qdrant index -> document_id-filtered retrieval`

This gives the project a measurable baseline before any section-aware or table-aware
RAG chunking changes.

## Current Decision

- Keep the current page-scoped `SentenceSplitter(chunk_size=800, chunk_overlap=120)`
  as the baseline.
- First implement a golden-query manifest and baseline retrieval eval script.
- Do not call the answerer or synthesize final answers in the first eval.
- Do not call cloud parse / LlamaParse for this eval.
- Use local PDF physical page numbers: **PyMuPDF 1-based page numbers**.
- Do not mix these with printed page numbers shown in report headers.
- Continue to enforce `document_id = current document` metadata filtering.

## Recommended Next Files

Suggested next implementation files:

- `data/outputs/eval/golden_queries_v1_1.json`
  - local artifact / manifest for the current eval set
  - do not commit unless explicitly requested
- `scripts/run_retrieval_eval.py`
  - repeatable baseline eval runner
- optional later:
  - `data/outputs/eval/retrieval_eval_baseline.json`
  - generated report from the runner

## Eval Runner Scope

The first eval runner should:

1. Load the golden-query manifest.
2. Parse each target document locally.
3. Build current chunks using the existing `build_chunks(...)`.
4. Index chunks with existing `DocumentChunkIndexer`.
5. Retrieve top-k chunks with existing `DocumentChunkRetriever`.
6. Compare retrieved pages with expected pages.
7. Output:
   - `page_hit@k`
   - retrieved pages
   - chunk count
   - retrieval latency
   - top chunk previews
   - miss cases

The first eval runner should **not**:

- call the chat answerer
- call external web search
- call LLM answer synthesis
- introduce reranking
- introduce new chunking logic
- change `table_metrics.py`

## Core Retrieval Set v1.1

These entries are suitable for the first automatic `page_hit@k` baseline.

| id | document | query | expected_pages |
|---|---|---|---|
| CMB-01 | `data/raw/招商银行2025年度报告.pdf` | 招商银行2025年度营业收入、归属于本行股东的净利润分别是多少？ | `[8, 14, 19]` |
| CMB-02 | `data/raw/招商银行2025年度报告.pdf` | 招商银行2025年加权平均净资产收益率 ROAE 是多少？请不要使用 ROAA。 | `[14, 19]` |
| CMB-03 | `data/raw/招商银行2025年度报告.pdf` | 截至2025年末，招商银行不良贷款率和拨备覆盖率分别是多少？ | `[8, 16, 19]` |
| CMB-04 | `data/raw/招商银行2025年度报告.pdf` | 截至2025年末，招商银行核心一级资本充足率和资本充足率分别是多少？ | `[8, 16, 38]` |
| CMB-05 | `data/raw/招商银行2025年度报告.pdf` | 招商银行客户存款总额是多少？较上年末增长多少？ | `[15, 19, 47]` |
| CMB-06 | `data/raw/招商银行2025年度报告.pdf` | 招商银行2025年度普通股现金分红方案是什么？每股派息和分红比例是多少？ | `[4, 100, 101]` |
| CMB-09 | `data/raw/招商银行2025年度报告.pdf` | 招商银行财富管理手续费及佣金收入是多少，同比增长多少？ | `[24]` |
| CMB-10 | `data/raw/招商银行2025年度报告.pdf` | 招商银行在2026年前景展望中提出哪些经营重点或应对措施？ | `[68, 69]` |
| TCEHY-01 | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯2025年收入和本公司权益持有人应占盈利分别是多少？ | `[4, 8]` |
| TCEHY-02 | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯微信及 WeChat 合并月活跃账户数是多少？QQ 移动终端月活是多少？ | `[5]` |
| TCEHY-03 | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯本土市场游戏和国际市场游戏收入、同比增幅分别是多少？ | `[9]` |
| TCEHY-04 | `data/raw/腾讯控股2025年度报告.pdf` | 年报将腾讯全年营销服务收入增长归因于哪些主要因素？ | `[9]` |
| TCEHY-07 | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯董事会建议派发的2025年末期股息是多少？ | `[7, 27]` |
| TCEHY-10 | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯2025年资本开支是多少？ | `[18]` |

## Qualitative / Sidecar Set

These entries are useful for demo, Mixed QA, chat memory, or later qualitative review,
but they should not be part of the first pure `page_hit@k` score.

| id | document | query | expected_pages | note |
|---|---|---|---|---|
| CMB-07 | `data/raw/招商银行2025年度报告.pdf` | 年报哪些段落提到 AI First / 数智化转型？列出2-3个原文要点。 | `[44, 45]` | good section-aware retrieval case |
| CMB-08A | `data/raw/招商银行2025年度报告.pdf` | 招商银行如何管控房地产风险？ | `[47]` | split from wider CMB-08 |
| CMB-08B | `data/raw/招商银行2025年度报告.pdf` | 招商银行如何管控地方政府隐性债务或地方债务风险？ | `[64]` | split from wider CMB-08 |
| CMB-08C | `data/raw/招商银行2025年度报告.pdf` | 招商银行如何管控零售贷款或消费贷款风险？ | `[49, 50]` | split from wider CMB-08 |
| TCEHY-05A | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯金融科技及企业服务年度收入增长的主要原因是什么？ | `[9]` | annual revenue wording |
| TCEHY-05B | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯金融科技及企业服务年度毛利增长的主要原因是什么？ | `[10]` | gross-profit wording |
| TCEHY-06A | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯视频号在内容生态或用户时长方面有哪些披露？ | `[6]` | qualitative |
| TCEHY-06B | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯视频号对营销服务或广告增长有什么贡献？ | `[9, 10]` | qualitative |
| TCEHY-08A | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯如何描述 AI 投入？ | `[5, 11]` | qualitative |
| TCEHY-08B | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯如何描述混元模型？ | `[5, 6]` | qualitative |
| TCEHY-08C | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯如何描述 AI 广告能力？ | `[6, 9, 13]` | qualitative |
| TCEHY-09A | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯面临哪些市场风险？ | `[173, 174, 175, 176]` | deep-page retrieval |
| TCEHY-09B | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯面临哪些信贷风险？ | `[178, 179, 180, 181]` | deep-page retrieval |
| TCEHY-09C | `data/raw/腾讯控股2025年度报告.pdf` | 腾讯面临哪些流动性风险？ | `[182, 183]` | deep-page retrieval |

## Aliases For Tencent Queries

Tencent annual report text is Traditional Chinese. Eval manifests should include
aliases for common simplified/traditional variants:

- `视频号` / `視頻號`
- `营销` / `營銷`
- `云服务` / `雲服務`
- `权益` / `權益`
- `资本开支` / `資本開支`
- `财务风险` / `財務風險`
- `信贷风险` / `信貸風險`
- `流动性风险` / `流動性風險`

## Metrics

First-pass metrics:

- `page_hit@6`
- `retrieved_pages`
- `retrieval_latency_ms`
- `chunk_count`
- `top_chunk_previews`
- `miss_cases`

Optional later metrics:

- `section_hit@6`
- `table_row_hit@6`
- `keyword_fallback_rate`
- `embedding_tokens`
- `index_build_ms`
- `used_document_citations_count`

## Decision Rule After Baseline

Use the baseline report before changing chunking:

- If core set hit rate is already high, do not rush chunking changes.
- If misses cluster around business sections, prioritize `section-aware metadata v1`.
- If misses cluster around table rows, consider `table-row QA chunks`.
- If misses are mostly Traditional/Simplified wording issues, improve aliases or keyword fallback first.
- Keep Headline Metrics v2 in `table_metrics.py`; do not replace it with RAG-generated metrics.
