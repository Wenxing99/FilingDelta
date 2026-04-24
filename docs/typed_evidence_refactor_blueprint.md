# FilingDelta Typed Evidence Retrieval Blueprint

## 1. Why Change

FilingDelta current retrieval path is strong enough to prove the MVP chain, but it is still built on a single retrieval object:

- parse into `ParsedFiling`
- split into page-scoped `FilingChunk`
- embed generic text chunks
- retrieve generic text chunks with `document_id` filter

That baseline is implemented in:

- [src/filingdelta/ingestion/pipeline.py](/Users/wenxing/FilingDelta/src/filingdelta/ingestion/pipeline.py:15)
- [src/filingdelta/ingestion/chunking.py](/Users/wenxing/FilingDelta/src/filingdelta/ingestion/chunking.py:11)
- [src/filingdelta/retrieval/indexer.py](/Users/wenxing/FilingDelta/src/filingdelta/retrieval/indexer.py:22)
- [src/filingdelta/retrieval/retriever.py](/Users/wenxing/FilingDelta/src/filingdelta/retrieval/retriever.py:18)
- [src/filingdelta/services/chat_qa.py](/Users/wenxing/FilingDelta/src/filingdelta/services/chat_qa.py:59)

This works, but our recent baseline eval and miss cases show a structural limit:

- narrative questions want section-level evidence
- metric questions want table-row evidence
- page chunks are still needed, but mainly as fallback and citation anchors

So the next meaningful upgrade is not "tune `SentenceSplitter` harder". It is to move from a single `chunk` retrieval model to a `typed evidence retrieval` model.

## 2. Current Constraints

This refactor should respect the current MVP boundaries in `AGENTS.md`:

- keep document-scoped retrieval with `document_id` filtering
- keep current Qdrant-based local retrieval path
- do not replace `Headline Metrics v2` in `table_metrics.py`
- do not introduce a heavy planner / reranker / hybrid stack first
- do not depend on Windows `.cmd` scripts
- preserve current `page_number + quote` citation contract

## 2.1 Current Status And Priority Override

This blueprint remains the medium-term retrieval architecture, but the immediate MVP priority has changed.

What has already landed:

- shared `EvidenceKind / EvidenceMetadata / EvidenceUnit` schema groundwork
- `page_text` evidence backed by the existing page-scoped chunks
- `section_text` builder, taxonomy, index metadata, and typed retrieval filter support
- controlled `section_text` activation for narrative / risk / strategy / AI / product-metrics questions
- prompt-level router boundary tightening so disclosure-only risk questions stay `document_only`

Why the next step is not `table_row` yet:

- frontend testing showed that chat latency is now the demo-critical bottleneck
- first chat on a long document can still synchronously perform parse / chunk / evidence build / embedding / Qdrant indexing
- `section_text` improves narrative retrieval quality but also increases cold-index cost and answer-context size
- `document_only` still waits on contextualizer, router, answerer, and memory summarizer
- `mixed` additionally waits on planner and external web search

Temporary priority:

`Latency-first stabilization -> table_row -> deterministic EvidenceQueryRouter`

The typed evidence plan is not being abandoned. It is being sequenced behind the latency work needed to keep the demo usable.

## 3. Target Architecture

Target chain:

`Parse -> Structure Build -> Evidence Build -> Typed Index -> Query Normalize -> Evidence Route -> Typed Retrieval -> Answer/Citation`

The key change is the middle layer: instead of producing only one kind of retrieval object, the system should produce three evidence kinds.

### 3.1 Evidence kinds

1. `page_text`
   - current page-scoped chunk baseline
   - retrieval fallback
   - stable citation anchor

2. `section_text`
   - section-aware narrative evidence
   - best for strategy / business / risk / outlook / dividend policy style questions

3. `table_row`
   - row-aware structured evidence
   - best for revenue / net profit / ROE / capital expenditure / NPL / provision coverage / dividend-per-share style questions

## 4. Metadata Contract

Every evidence unit should share one core metadata contract.

Required fields:

- `document_id`
- `page_number`
- `section_title`
- `section_type`
- `chunk_kind`
- `table_id`
- `row_label`
- `metric_tags`
- `period_hint`

Recommended optional fields:

- `page_end`
- `header_path`
- `language`
- `source_parser`
- `bbox_hint`

### 4.1 Field semantics

- `chunk_kind`
  - `page_text | section_text | table_row`
- `section_title`
  - raw heading text when available
- `section_type`
  - normalized taxonomy label aligned with current summary taxonomy
- `table_id`
  - empty for non-table evidence
- `row_label`
  - normalized row label for `table_row`
- `metric_tags`
  - normalized tags such as `revenue`, `net_profit`, `roe`, `dividend`, `capital_expenditure`
- `period_hint`
  - normalized period token, not free prose
  - examples: `fy2025`, `q3_2025_ytd`, `as_of_2025-12-31`

## 5. Schema Plan

### 5.1 Keep existing models

Keep:

- `ParsedFiling`
- `ParsedPage`
- `FilingChunk`
- current citation schemas

Do not break current analysis and chat flows while the typed layer is being introduced.

### 5.2 Add new schema models

Recommended new models in [src/filingdelta/schemas/filing.py](/Users/wenxing/FilingDelta/src/filingdelta/schemas/filing.py:1) or a nearby schema file if it becomes too crowded:

- `EvidenceKind`
- `EvidenceMetadata`
- `EvidenceUnit`
- `EvidenceBuildResult`
- `EvidenceQueryRoute`

Suggested shape:

```python
class EvidenceKind(str, Enum):
    PAGE_TEXT = "page_text"
    SECTION_TEXT = "section_text"
    TABLE_ROW = "table_row"


class EvidenceMetadata(BaseModel):
    document_id: str
    page_number: int
    page_end: int | None = None
    section_title: str | None = None
    section_type: str | None = None
    chunk_kind: EvidenceKind
    table_id: str | None = None
    row_label: str | None = None
    metric_tags: list[str] = Field(default_factory=list)
    period_hint: str | None = None
    source_path: Path
    parser_kind: ParserKind


class EvidenceUnit(BaseModel):
    evidence_id: str
    text: str
    metadata: EvidenceMetadata
```

## 6. Ingestion Refactor Plan

### 6.1 Keep current chunking

Do not remove [build_chunks(...)](/Users/wenxing/FilingDelta/src/filingdelta/ingestion/chunking.py:11). It remains the `page_text` builder.

### 6.2 Add structure-aware builders

Recommended new ingestion modules:

- `src/filingdelta/ingestion/evidence_builder.py`
- `src/filingdelta/ingestion/section_evidence.py`
- `src/filingdelta/ingestion/table_row_evidence.py`
- `src/filingdelta/ingestion/query_normalization.py`

### 6.3 Section evidence v1

Inputs:

- `ParsedFiling`
- parser-specific metadata already present in pages
- heuristics from current summary taxonomy

Build strategy:

- HTML / EX-99.1:
  - prefer heading tags / DOM grouping
- PDF:
  - first try TOC / bookmarks when available
  - then use text block / font / spacing signals from parser metadata
  - then use conservative heading heuristics and taxonomy keywords
- if section boundary is not reliable:
  - do not force `section_text`
  - fall back to `page_text`

V1 target taxonomy:

- `financial_summary`
- `dividend`
- `operating_metrics`
- `strategy_outlook`
- `shareholder`
- `risk_asset_quality`
- `business_review`
- `product_user_metrics`
- `sustainability`
- `governance`
- `other`

### 6.4 Table-row evidence v1

Do not build a universal table engine.

Instead:

- reuse current signals from [src/filingdelta/ingestion/table_metrics.py](/Users/wenxing/FilingDelta/src/filingdelta/ingestion/table_metrics.py:1)
- reuse:
  - candidate pages
  - row label matching
  - period selection
  - unit handling
- optionally supplement with `PyMuPDF Page.find_tables()` on high-value pages
- if row binding is weak:
  - skip the `table_row`
  - fall back to `page_text`

Embedding text for each row should be semantic enough to retrieve well, for example:

`表格: 主要会计数据; 行名: 营业收入; 期间: 2025年度; 数值: 3375.32亿元; 对应列: 本报告期`

## 7. Retrieval Refactor Plan

### 7.1 Keep single collection first

Do not split into multiple Qdrant collections in v1.

Recommended v1:

- keep one collection
- add `chunk_kind` and related payload metadata
- keep current `document_id` filter unchanged

This is the smallest change that still unlocks typed retrieval.

### 7.2 Indexer changes

Extend [src/filingdelta/retrieval/indexer.py](/Users/wenxing/FilingDelta/src/filingdelta/retrieval/indexer.py:22) so it can index both:

- current `FilingChunk`
- new `EvidenceUnit`

Suggested direction:

- keep `chunk_to_node(...)` for compatibility
- add `evidence_to_node(...)`
- if needed, add a shared helper to build node metadata

### 7.3 Typed retriever

Recommended new module:

- `src/filingdelta/retrieval/evidence_retriever.py`

Responsibilities:

- retrieve by `document_id`
- optionally filter by `chunk_kind`
- optionally filter by `metric_tags`, `section_type`, `period_hint`
- return a unified retrieval result shape

### 7.4 Query router

Recommended new module:

- `src/filingdelta/retrieval/evidence_router.py`

V1 should be deterministic and local, not LLM-based.

Route labels:

- `metric`
- `narrative`
- `mixed`
- `fallback`

Routing rules:

- metric-like question:
  - `table_row` first
  - then `page_text`
- narrative-like question:
  - `section_text` first
  - then `page_text`
- mixed:
  - retrieve `table_row + section_text`
  - then merge
- fallback:
  - `page_text`

## 8. Query Normalization

Recommended new module:

- `src/filingdelta/ingestion/query_normalization.py`

This should replace ad hoc alias patching with a small deterministic normalization layer.

V1 steps:

1. Unicode NFKC normalization
2. punctuation / whitespace normalization
3. Traditional-Simplified projection
4. company / product alias normalization
5. metric canonicalization
6. period canonicalization

Important: this layer should support retrieval metadata matching, not only raw query rewriting.

## 9. Service Integration Plan

Main integration point:

- [src/filingdelta/services/chat_qa.py](/Users/wenxing/FilingDelta/src/filingdelta/services/chat_qa.py:59)

Recommended integration order:

1. keep existing top-level chat router:
   - `document_only`
   - `concept_only`
   - `mixed`
2. whenever document evidence is needed:
   - normalize question
   - evidence-route the question
   - run typed retrieval
   - fall back to `page_text` if typed retrieval is weak
3. keep current answer synthesis and citation assembly shape as much as possible

Important boundary:

- `table_metrics.py` still owns left-column headline extraction
- typed retrieval supports right-column QA and citation retrieval
- they may share low-level parsing utilities, but should not collapse into one module

## 10. Telemetry Plan

Current telemetry is useful but too generic for this refactor.

Extend telemetry with:

- `evidence_route`
- `retrieved_page_text_count`
- `retrieved_section_text_count`
- `retrieved_table_row_count`
- `typed_fallback_triggered`
- `section_hit@k`
- `table_row_hit@k`

Keep telemetry fail-open.

## 11. Eval Plan

The next eval pass should not stop at `page_hit@k`.

Add:

- `page_hit@k by route`
- `section_hit@k`
- `table_row_hit@k`
- `typed_fallback_rate`
- `query_normalization_applied`

Important first use:

- use current `golden_queries_v1_1`
- tag each query with expected route:
  - `metric`
  - `narrative`
  - `mixed`

## 12. Implementation Phases

### Phase 0: prep

- add schema placeholders
- add evidence metadata contract
- keep all existing code paths working

### Phase 1: `section_text`

- add `section_text` builder
- add payload metadata support in indexer
- add typed retriever support for `section_text`
- evaluate qualitative / business / risk questions first
- status: implemented and controlled-activated; remaining work is latency-aware evidence compression, not another broad builder rewrite

### Interlude: latency-first stabilization

Before `table_row`, stabilize the already-active chat path:

- add a local chat latency smoke script
- report route, step latency, token usage, retrieval counts, and citation counts per question
- make `Memory Summarizer` non-blocking or lower-frequency
- skip `Conversation Contextualizer` when there is no chat history or no follow-up signal
- prewarm chat index after analysis completion or in a background task
- compress `section_text` evidence before answer synthesis while preserving page and citation anchors

### Phase 2: `table_row`

- build conservative `table_row` evidence on top of current table signals
- index with `chunk_kind=table_row`
- route metric queries to it first
- status: deferred until latency-first stabilization is complete

### Phase 3: typed routing + normalization

- add deterministic `EvidenceQueryRouter`
- add query normalization
- replace current shallow keyword fallback with typed-first fallback

### Phase 4: telemetry + eval expansion

- add route-level telemetry
- extend baseline eval artifacts
- compare before / after with the same golden queries

### Phase 5: optional future work

Only after typed retrieval is stable:

- reranker
- parent-child retrieval
- more advanced table extraction
- more formal section hierarchy

## 13. What Not To Do First

Do not start with:

- semantic chunking as the main solution
- reranker-first
- multiple Qdrant collections
- heavy LLM evidence routing
- full parse-stack replacement
- replacing `table_metrics.py` with chat retrieval

These may become useful later, but they do not solve the current object-type mismatch.

## 14. First Concrete Code Cuts

If implementation starts now, the first changes should be:

1. add evidence schemas
2. add `section_text` builder
3. extend indexer metadata contract
4. add typed retriever with `chunk_kind` filtering
5. wire `ChatQAService` to use typed retrieval for document queries

That is the smallest sequence that changes system shape without blowing up the current MVP.

## 15. Success Criteria

This refactor is working if:

- metric queries increasingly hit `table_row`
- narrative queries increasingly hit `section_text`
- page chunks remain available as stable fallback
- citation quality improves without losing page traceability
- the system stops depending on one-off alias and query-specific retrieval patches

The intended outcome is not just "better retrieval score". The intended outcome is a cleaner product story:

FilingDelta does not treat a filing as one undifferentiated blob of text. It builds a typed evidence layer first, then performs traceable reading, QA, and comparison on top of that layer.
