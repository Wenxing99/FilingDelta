# Dayu Agent 技术报告

## 1. 报告目的

本报告面向 FilingDelta 团队，目标不是复述 `dayu-agent` 的 README，而是回答几个更工程化的问题：

- `dayu-agent` 当前真正的主路线是什么；
- 它的 Agent / Host / 财报处理体系是怎样落地的；
- 它解决了哪些问题，哪些问题还没有解决；
- 哪些做法值得 FilingDelta 借鉴，哪些不适合当前阶段直接照搬。

本报告基于对 `/Users/wenxing/dayu-agent` 的本地代码与文档阅读，重点参考：

- [根 README](/Users/wenxing/dayu-agent/README.md:1)
- [根 AGENTS](/Users/wenxing/dayu-agent/AGENTS.md:19)
- [架构文档](/Users/wenxing/dayu-agent/docs/architect.md:27)
- [dayu 开发总览](/Users/wenxing/dayu-agent/dayu/README.md:76)
- [Engine 手册](/Users/wenxing/dayu-agent/dayu/engine/README.md:1)
- [Fins 手册](/Users/wenxing/dayu-agent/dayu/fins/README.md:1)

---

## 2. 项目概览

### 2.1 项目定位

`dayu-agent` 将自己定义为“买方财报分析 Agent”，当前对外主张覆盖四类工作：

1. 财报数据管线：下载、上传、处理财报；
2. 投研问答：单轮 prompt、多轮 interactive、WeChat 问答；
3. 买方分析报告写作；
4. Markdown 报告渲染为 HTML / PDF / Word。

依据见 [README](/Users/wenxing/dayu-agent/README.md:7)。

### 2.2 当前真正的主路线

从代码和文档看，`dayu-agent` 当前最成熟的不是 GUI，也不是 durable retrieval，而是下面这条链：

`UI -> Service -> Execution Contract -> Host -> scene preparation -> Agent`

在这条执行主链之上，再挂两块领域能力：

- `FinsToolService`：提供财报文档读取工具；
- `FinsRuntime`：提供下载 / 上传 / process 等 direct operation。

这意味着它本质上更像：

> 一个“宿主强约束的 Agent 执行平台 + 证券文档工具层 + 写作流水线”

而不是一个以向量检索为核心的财报阅读器。

依据见：

- [架构原则](/Users/wenxing/dayu-agent/docs/architect.md:16)
- [dayu 总览](/Users/wenxing/dayu-agent/dayu/README.md:92)
- [PromptService](/Users/wenxing/dayu-agent/dayu/services/prompt_service.py:32)
- [Host](/Users/wenxing/dayu-agent/dayu/host/host.py:534)

### 2.3 已明确未完成的部分

`dayu-agent` 自己在文档里已经承认几件事还不成熟：

- GUI 尚未实现；
- Web UI 只有 FastAPI 骨架；
- WeChat 仅文本版；
- 港股 / A 股财报下载尚未实现；
- durable memory / retrieval layer 尚未完成，目前只有 working memory + episode summary。

依据见 [README](/Users/wenxing/dayu-agent/README.md:27) 和 [dayu 总览](/Users/wenxing/dayu-agent/dayu/README.md:58)。

### 2.4 一句话判断

如果 FilingDelta 只吸收一个总判断，那就是：

> Dayu 的核心竞争力不在“RAG 黑盒问答”，而在“强约束执行平台 + 工具化财报能力 + 可评测的离线 artifact”。

---

## 3. 核心技术栈

从 [pyproject.toml](/Users/wenxing/dayu-agent/pyproject.toml:5) 可以看到它的技术栈画像：

- 语言 / 运行时：Python 3.11；
- Web / API：`FastAPI`；
- LLM 接入：OpenAI-compatible runner，自研 `AsyncOpenAIRunner`；
- HTTP / async：`aiohttp`、`httpx`、`requests`；
- 文档处理：`docling`、`docling-core`、`beautifulsoup4`、`lxml`；
- SEC / XBRL：`edgartools`；
- Web fallback：`playwright`、`playwright-stealth`；
- CLI：`click`、`prompt_toolkit`；
- 结构化数据：`pandas`；
- HTML 内容提取：`trafilatura`、`readability-lxml`、`markdownify`、`html2text`。

从入口脚本看，当前对外主要有三个命令：

- `dayu-cli`
- `dayu-wechat`
- `dayu-render`

见 [pyproject.toml](/Users/wenxing/dayu-agent/pyproject.toml:91)。

### 技术栈特征判断

1. 它不是基于 LangGraph / LlamaIndex / CrewAI 这类编排框架。
2. 它不是向量数据库优先的 RAG 系统。
3. 它更像“自建 Agent runtime + domain tools + processor stack”。

对 FilingDelta 的意义是：可以学它的“工具接口与执行边界”，但不应该把它误判成 RAG 架构样板。

---

## 4. 系统架构

### 4.1 四层架构是 Dayu 的硬约束

`dayu-agent` 在文档和仓库规则里都反复强调，稳定架构只有四层：

- `UI`
- `Service`
- `Host`
- `Agent`

见：

- [AGENTS](/Users/wenxing/dayu-agent/AGENTS.md:19)
- [architect.md](/Users/wenxing/dayu-agent/docs/architect.md:29)

这不是抽象口号，而是实际代码边界：

- `Service` 负责受理请求、解释业务语义、决定 scene；
- `Host` 负责 session / run / cancellation / concurrency / resume；
- `Agent` 只消费最终消息和工具，不理解业务语义。

### 4.2 三个 public preparation module

Dayu 明确把以下三个东西定义为 public module，而不是新层：

- `startup preparation`
- `contract preparation`
- `scene preparation`

见 [architect.md](/Users/wenxing/dayu-agent/docs/architect.md:40)。

这套判断很重要，因为它让层次保持稳定，同时允许内部装配复杂化。

### 4.3 Contract-first 执行边界

Service 不直接构造最终 `AgentInput`，而是先收敛成 `ExecutionContract`。核心对象包括：

- `ExecutionContract`
- `AcceptedExecutionSpec`
- `ScenePreparationSpec`
- `ExecutionPermissions`

见：

- [contract_preparation.py](/Users/wenxing/dayu-agent/dayu/services/contract_preparation.py:95)
- [agent_execution.py](/Users/wenxing/dayu-agent/dayu/contracts/agent_execution.py:1)
- [scene_execution_acceptance.py](/Users/wenxing/dayu-agent/dayu/services/scene_execution_acceptance.py:92)

这是 Dayu 很值得注意的工程点：它把“业务受理后的执行决定”从“最终 LLM 输入装配”中间切出了一层稳定契约。

### 4.4 Host 是平台层，不是薄壳

`Host` 在 Dayu 里不是一个转发器，而是聚合根。它负责：

- session 管理；
- run 注册与状态；
- timeout / cancel / concurrency lane；
- pending turn resume；
- scene preparation 调度；
- agent stream 与 direct operation 托管。

见 [host.py](/Users/wenxing/dayu-agent/dayu/host/host.py:172) 和 [executor.py](/Users/wenxing/dayu-agent/dayu/host/executor.py:293)。

这使 Dayu 的平台能力很完整，但也意味着体系明显比 FilingDelta 当前 MVP 更重。

---

## 5. Agent / Workflow 设计

### 5.1 Dayu 不是 DAG workflow，而是 scene-driven agent execution

Dayu 的执行方式不是预定义 DAG，也不是简单的“一个 agent + 一组工具”。它更像：

1. Service 选定 `scene`；
2. `SceneExecutionAcceptancePreparer` 根据 manifest、模型规则、runtime 配置生成 `AcceptedExecutionSpec`；
3. Host 在 scene preparation 阶段组 system prompt、messages、tool registry；
4. `AsyncAgent` 负责运行。

依据见：

- [prompt manifest](/Users/wenxing/dayu-agent/dayu/config/prompts/manifests/prompt.json:1)
- [write manifest](/Users/wenxing/dayu-agent/dayu/config/prompts/manifests/write.json:1)
- [scene acceptance](/Users/wenxing/dayu-agent/dayu/services/scene_execution_acceptance.py:129)
- [scene preparer](/Users/wenxing/dayu-agent/dayu/host/scene_preparer.py:368)

### 5.2 Prompt / Chat / Fins / Write 四条服务路径

Dayu 当前主要有四种服务入口：

- `PromptService`：单轮问答；
- `ChatService`：多轮对话；
- `FinsService`：direct operation；
- `WriteService`：报告写作。

见：

- [prompt_service.py](/Users/wenxing/dayu-agent/dayu/services/prompt_service.py:23)
- [chat_service.py](/Users/wenxing/dayu-agent/dayu/services/chat_service.py:42)
- [fins_service.py](/Users/wenxing/dayu-agent/dayu/services/fins_service.py:25)
- [write_service.py](/Users/wenxing/dayu-agent/dayu/services/write_service.py:31)

其中值得特别注意的是：

- `ChatService` 有 session lock，避免同 session 并发 turn 冲突，[chat_service.py](/Users/wenxing/dayu-agent/dayu/services/chat_service.py:96)；
- `FinsService` 在创建 Host run 之前先做同步 preflight 校验，[fins_service.py](/Users/wenxing/dayu-agent/dayu/services/fins_service.py:47)；
- `WriteService` 不是简单发一个 prompt，而是启动完整的 write pipeline，[write_service.py](/Users/wenxing/dayu-agent/dayu/services/write_service.py:42)。

### 5.3 AsyncAgent 的能力边界

`AsyncAgent` 支持的能力很全：

- 迭代上限；
- fallback mode；
- duplicate tool call 防重复；
- context budget soft / hard limit；
- compaction；
- continuation；
- final answer 聚合；
- content filter / degraded 收口。

见 [async_agent.py](/Users/wenxing/dayu-agent/dayu/engine/async_agent.py:314)、[async_agent.py](/Users/wenxing/dayu-agent/dayu/engine/async_agent.py:555)。

这说明 Dayu 的 Agent runtime 很成熟，但也能看到一个风险：

> 这套 runtime 已经有明显的平台复杂度，主循环开始变大，后续维护成本会持续上升。

### 5.4 Write pipeline 是另一条更重的 workflow

Dayu 的写作链路不是普通问答的延伸，而是单独的多阶段 pipeline，包括：

- 初稿写作；
- 条件占位符补强；
- chapter audit；
- evidence confirm；
- repair；
- chapter 0 overview；
- chapter 10 decision；
- source list；
- resume 与 artifact 落盘。

见：

- [write pipeline](/Users/wenxing/dayu-agent/dayu/services/internal/write_pipeline/pipeline.py:1)
- [chapter coordinator](/Users/wenxing/dayu-agent/dayu/services/internal/write_pipeline/chapter_execution_coordinator.py:1)

对 FilingDelta 来说，这说明 Dayu 的“workflow”主力并不在问答路由，而在写作生产线。

---

## 6. 文档 / 财报处理链路

### 6.1 Fins 有两条稳定路径

Fins 在 Dayu 里有两条稳定路径：

1. `Agent augmentation path`：把财报读工具挂给 Agent；
2. `Direct operation path`：download / upload / process 不经过 Agent。

见 [Fins README](/Users/wenxing/dayu-agent/dayu/fins/README.md:7)。

这点非常清晰，也值得 FilingDelta 参考：不要把“文档处理长事务”和“Agent 问答”混成一条执行线。

### 6.2 处理器是主路线，不是 embeddings

`FinsToolService` 的注释非常关键：

- 不依赖 `processed/*.json`；
- 所有读取通过实时 Processor 能力完成；
- 缓存只保留 Processor 实例。

见 [FinsToolService](/Users/wenxing/dayu-agent/dayu/fins/tools/service.py:120)。

这说明 Dayu 的在线文档访问真源是：

> `source docs -> processor -> tools`

而不是：

> `source docs -> chunk -> embedding -> vector retrieval`

### 6.3 财报 direct operation 链路

`DefaultFinsRuntime` 负责 direct operation：

- `validate_command()` 先做同步 preflight；
- `execute()` 再根据命令走下载、上传、process、单文档处理等分支；
- 同时提供共享的 `FinsToolService` 和 ingestion service factory。

见 [service_runtime.py](/Users/wenxing/dayu-agent/dayu/fins/service_runtime.py:1208)。

当前 direct operation 的工程特点：

- 受理前就拒绝非法请求；
- 支持同步和流式；
- process 链路可接受 `document_ids` 精确重跑；
- Host cancellation 通过窄 `cancel_checker` 继续下传。

### 6.4 process 的核心产物是 tool snapshots

Dayu 的 `process` 不是传统“清洗文本后入索引”。

它的关键产物是单文档工具快照：

- `list_documents`
- `get_document_sections`
- `read_section`
- `list_tables`
- `get_table`
- `get_page_content`
- `get_financial_statement`
- CI 模式下额外加 `search_document`、`query_xbrl_facts`

见 [tool_snapshot_export.py](/Users/wenxing/dayu-agent/dayu/fins/pipelines/tool_snapshot_export.py:650)。

因此 `processed` 产物更像：

> “把一个文档暴露给 LLM 时，工具层到底能看到什么”的离线快照

这和 FilingDelta 当前的 `chunk/index/citation` 中间产物是两种不同哲学。

### 6.5 在线读工具设计

`FinsToolService` 当前提供的核心读工具包括：

- `list_documents`
- `get_document_sections`
- `read_section`
- `search_document`
- `list_tables`
- `get_table`
- `get_page_content`
- `get_financial_statement`
- `query_xbrl_facts`

见 [service.py](/Users/wenxing/dayu-agent/dayu/fins/tools/service.py:168)。

这套工具很像一层“文档智能 API”，而不是一组无约束函数。

### 6.6 检索策略：bounded search，不是向量 RAG

`search_document` 的内部流程很有代表性：

- 查询歧义诊断；
- query intent 分类；
- 搜索计划生成；
- phrase / synonym / token 扩展；
- section semantic bucket 过滤；
- BM25F 排序；
- 证据化结果构建。

见 [search_engine.py](/Users/wenxing/dayu-agent/dayu/fins/tools/search_engine.py:1)。

这一点的工程价值在于：

- 它非常适合 SEC 章节结构稳定、主题强的场景；
- 但它不应被误读成通用 semantic retrieval。

### 6.7 Citation 与可追溯性

Dayu 的所有 Fins 工具输出都带统一 citation，字段包括：

- `source_type`
- `document_id`
- `ticker`
- `form_type`
- `filing_date`
- `accession_no`
- `fiscal_year`
- `fiscal_period`
- `item`
- `heading`

见 [tool_models.py](/Users/wenxing/dayu-agent/dayu/fins/domain/tool_models.py:81) 和 [service.py](/Users/wenxing/dayu-agent/dayu/fins/tools/service.py:1422)。

但需要区分：

- Dayu 的 citation 更偏“文档 / 章节 / 表格 / 报表定位”；
- FilingDelta 当前追求的是“结论级、point 级、quote 级 citation”。

因此 Dayu 的 citation 非常适合作为 provenance 底层，但不够直接拿来做 FilingDelta 中栏的最终交互形态。

---

## 7. Memory、Tool Use、Trace 与评测

### 7.1 Conversation memory：当前是 working memory + episodic summary

Dayu 的 memory manager 当前实现了：

- working memory；
- episodic memory compaction；
- background compaction coordinator；
- transcript 持久化。

但 durable memory store 和 retrieval index 默认都是空实现。

见：

- [conversation memory manager](/Users/wenxing/dayu-agent/dayu/host/conversation_memory.py:1102)
- [working memory policy](/Users/wenxing/dayu-agent/dayu/host/conversation_memory.py:701)
- [episodic compressor](/Users/wenxing/dayu-agent/dayu/host/conversation_memory.py:777)
- [compaction scene prompt](/Users/wenxing/dayu-agent/dayu/config/prompts/scenes/conversation_compaction.md:1)

这说明 Dayu 对 memory 的阶段判断比较克制：先做“会话辅助理解”，还没做“历史知识检索”。

### 7.2 ToolRegistry 是成熟的执行边界

`ToolRegistry` 支持：

- schema 验证；
- allowlist 路径控制；
- `fetch_more`；
- truncate spec；
- execution context 注入；
- response middleware。

见 [tool_registry.py](/Users/wenxing/dayu-agent/dayu/engine/tool_registry.py:57)。

这是 Dayu 一个很值得吸收的方法论：工具不是裸函数，而是带契约、权限和续读策略的执行单元。

### 7.3 Tool Trace V2 很成熟

Dayu 的 `tool_trace_v2` 会记录：

- `iteration_context_snapshot`
- `tool_call`
- `iteration_usage`
- `final_response`

并把 raw payload 放到冷存，再用 `utils/analyze_tool_trace.py` 做离线诊断。

见：

- [tool_trace.py](/Users/wenxing/dayu-agent/dayu/engine/tool_trace.py:7)
- [trace analyzer](/Users/wenxing/dayu-agent/utils/analyze_tool_trace.py:1)

这比普通的“日志 + latency”强很多，因为它能回答：

- 模型为什么连续重复调工具；
- 哪个工具 payload 太大；
- 截断 contract 是否有效；
- 最终 degraded 是怎么发生的。

### 7.4 Eval / benchmark 体系是 Dayu 的强项

Dayu 在质量保障上做了三层东西：

1. 工具快照导出；
2. ground truth baseline 固化；
3. SEC CI 评分。

见：

- [tool_snapshot_export.py](/Users/wenxing/dayu-agent/dayu/fins/pipelines/tool_snapshot_export.py:528)
- [ground_truth_baseline.py](/Users/wenxing/dayu-agent/dayu/fins/ground_truth_baseline.py:1)
- [score_sec_ci.py](/Users/wenxing/dayu-agent/dayu/fins/score_sec_ci.py:1)

这套设计的价值不在于“分数好看”，而在于：

> 它把文档处理能力变成了稳定 artifact，可以回归、可以审查、可以定位退化来源。

---

## 8. 它解决了什么问题

截至当前版本，`dayu-agent` 已经较好解决了以下问题：

1. 把 Agent 执行从 UI 和业务语义中分层拆开；
2. 把 direct operation 与 Agent augmentation 解耦；
3. 为 SEC 财报建立了较完整的 processor + tools 能力；
4. 为写作链路建立了 audit / confirm / repair 闭环；
5. 为执行链路建立了较强的 trace、snapshot 与 CI 体系。

如果从工程成熟度看，Dayu 最扎实的三块是：

- Host / scene / contract 执行体系；
- Fins 文档工具层；
- snapshot / trace / CI。

---

## 9. 它没有解决或还没完全解决的问题

当前仍然明显未完成或不是强项的部分：

1. durable retrieval layer；
2. 长期记忆；
3. 图形化阅读器产品体验；
4. A 股 / H 股下载主链；
5. point 级 quote citation；
6. 面向阅读与 diff 的轻交互 UI。

这也是它与 FilingDelta 当前阶段最大的产品差异。

---

## 10. 值得借鉴的点

按 FilingDelta 当前阶段的重要性排序，我认为最值得借鉴的是：

### 10.1 借鉴“文档工具层”，但做轻量版

FilingDelta 当前已经有 parse / chunk / metadata / citation / retrieval。下一步最适合借鉴 Dayu 的地方，不是重做 Host，而是补一层稳定文档工具接口，例如：

- `get_sections`
- `read_section`
- `list_tables`
- `get_table`
- `get_page_content`

原因：

- 对表格和章节问答更稳；
- 更适合 citation repair；
- 更适合以后做 compare 前的结构化证据读取。

参考实现见 [FinsToolService](/Users/wenxing/dayu-agent/dayu/fins/tools/service.py:168)。

### 10.2 借鉴 section semantic metadata

Dayu 的 section semantic 和 intent-aware search 很适合 FilingDelta 后续 retrieval eval 的 miss case 分析：

- 哪些问题其实不是 embedding 不行，而是章节结构没被利用；
- 哪些 query 需要 section bucket 辅助，而不是更大的 top-k。

参考见 [search_engine.py](/Users/wenxing/dayu-agent/dayu/fins/tools/search_engine.py:236)。

### 10.3 借鉴 artifact-first eval

FilingDelta 现在已经在做 `golden queries` 和 baseline retrieval eval。完全可以进一步学习 Dayu：

- 给关键样本文档导出固定中间产物；
- 用这些产物做结构化回归；
- 把“结果稳定性”从聊天记录迁回文件 artifact。

这是 Dayu 对 FilingDelta 最直接、最务实的启发之一。

### 10.4 借鉴 tool trace 的问题意识

FilingDelta 已经做了 telemetry v1。下一步不一定需要复制 `tool_trace_v2` 全套，但可以学习它记录这些事实：

- 当前轮路由是什么；
- 调了哪些工具 / 检索子模块；
- 哪个阶段触发了 fallback；
- final answer 是否 degraded；
- 重复调用发生在什么位置。

这会比单纯的 latency / token 汇总更能解释 Mixed QA 和 memory 路由。

### 10.5 借鉴“先把质量做成可审查 artifact，再扩功能”

Dayu 在 CI 和 baseline 这块的价值观与 FilingDelta 是相容的：

- 不把“模型感觉还行”当完成；
- 把退化变成可复现事实；
- 用固定样本和结构化导出做回归。

这点非常值得保留。

---

## 11. 不建议直接照搬的点

### 11.1 不建议照搬整套 Host 平台

Dayu 的 Host 层很完整，但对 FilingDelta 当前 `demo-first` 节奏来说太重了。照搬会立刻带来：

- 层次复杂度上升；
- 接口与类型维护成本上升；
- 改动范围扩散到当前并不需要的产品面。

### 11.2 不建议用 Dayu 搜索替代 FilingDelta 现有 RAG 主线

FilingDelta 当前主叙事已经明确是：

- single-document chatbot
- document-scoped retrieval
- `document_id` filter
- citation

Dayu 的 `search_document` 值得吸收，但更适合当“结构化辅助检索工具”，不适合作为主线替换。

### 11.3 不建议直接搬写作流水线

Dayu 的 write pipeline 很强，但它解决的是“买方报告生产”问题。FilingDelta 当前更核心的是：

- 可阅读
- 可追溯
- 可比较
- 可演示

所以现在直接搬 chapter audit / repair / decision pipeline，会明显偏离当前阶段主线。

### 11.4 不建议照搬它的 citation 终态

Dayu 的 citation 更偏 provenance metadata，不够细粒度，不适合作为 FilingDelta 当前中栏 `Citation Detail` 的最终产品形态。

FilingDelta 应继续坚持：

- `page_number`
- `quote`
- `summary point / metric` 绑定

Dayu 的 citation 更适合作为底层证据身份层。

### 11.5 不建议跟着它扩多入口产品面

CLI / WeChat / render / Web skeleton 是 Dayu 的产品扩张路线，不是 FilingDelta 当前最缺的东西。对 FilingDelta 来说，最重要的是把主阅读链路做扎实，而不是复制入口矩阵。

---

## 12. 对 FilingDelta 的具体启发

### 12.1 近期可落地启发

1. 在当前 ingestion / retrieval 之上补一层轻量文档工具接口。
   先不碰主链，只把 `sections / tables / page / statement` 这些确定性读能力抽出来。

2. 给 chunk 和 summary 补 section-aware metadata。
   不替代当前 Qdrant 检索，但让 retrieval eval 能观察“命中的 chunk 属于什么 section / topic”。

3. 把关键样本文档做成 artifact-first 回归集。
   除了 `golden queries`，再为 Headline Metrics、sectioned summary、citation 绑定、Mixed QA 输出固定 artifact。

4. 在 telemetry 基础上增加“执行解释层”。
   记录 router 类型、fallback 触发、citation 使用情况、是否用到外部证据，而不是只有 latency。

### 12.2 中期可落地启发

1. 为结构化财务事实补强“表格 / statement / XBRL”证据源。
   如果 FilingDelta 后续继续加深美股能力，可以吸收 Dayu 的 `financial_statement` / `query_xbrl_facts` 思路。

2. 为 compare 链路补“结构化证据接口”，而不是只靠自由文本 diff。
   这会比单纯让 LLM 比较两份长文更稳。

3. 如果以后 FilingDelta 进入“自动写 memo / 生成分析 note”阶段，再考虑借鉴 Dayu 的 write pipeline 思路。
   当前阶段不建议前移。

### 12.3 建议的吸收方式

最合理的落地方式不是“重构成 Dayu”，而是：

- 保持 FilingDelta 当前 `LlamaIndex + Qdrant + document_id filter` 主线；
- 在其上增加少量 Dayu 风格的“文档工具层”和“artifact eval 层”；
- 把 Dayu 的 section semantics、tool trace 思路、snapshot 回归方法吸收进来；
- 继续坚持 FilingDelta 自己的产品目标：阅读、追溯、差异分析。

---

## 13. 总结

`dayu-agent` 最值得 FilingDelta 学的，不是“再造一个更大的 Agent 平台”，而是三件事：

1. 用清晰契约把执行边界、工具边界和业务边界拆开；
2. 用 processor-aware 的文档工具层增强财报可读性和结构化证据能力；
3. 把质量评估做成固定 artifact，而不是停留在聊天里的主观印象。

如果只用一句话概括本报告的最终结论：

> Dayu 更像一个“财报 Agent 平台 + 文档工具系统”，FilingDelta 最该吸收的是它的工具层和评测方法，而不是照搬它更重的平台形态。
