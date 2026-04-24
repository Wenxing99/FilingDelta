import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import { askDemoChatStream } from "../../lib/api";
import type { ChatCitation, ChatResponse, CitationTarget, DemoDocument } from "../../lib/types";
import { MarkdownInline, MarkdownText } from "./MarkdownText";

type ChatPanelProps = {
  document: DemoDocument | null;
  onSelectCitation: (target: CitationTarget) => void;
};

type ChatThreadMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sections: ChatResponse["sections"];
  citations: ChatCitation[];
  route: ChatResponse["route"] | null;
  retrievalMode: ChatResponse["retrieval_mode"] | null;
  telemetry: ChatResponse["telemetry"];
  streamStatus?: string;
  isPending?: boolean;
  isError?: boolean;
};

type ChatSessionState = {
  sessionId: string;
  messages: ChatThreadMessage[];
};

export function ChatPanel({ document, onSelectCitation }: ChatPanelProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [draft, setDraft] = useState("");
  const [sessions, setSessions] = useState<Record<string, ChatSessionState>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const currentDocumentId = document?.document_id ?? null;
  const activeSession = currentDocumentId ? sessions[currentDocumentId] : null;
  const messages = activeSession?.messages ?? [];

  useEffect(() => {
    if (!currentDocumentId) {
      setDraft("");
      return;
    }
    setSessions((current) => {
      if (current[currentDocumentId]) {
        return current;
      }
      return {
        ...current,
        [currentDocumentId]: {
          sessionId: createSessionId(),
          messages: [],
        },
      };
    });
    setDraft("");
  }, [currentDocumentId]);

  useEffect(() => {
    const bodyNode = bodyRef.current;
    if (!bodyNode) {
      return;
    }
    bodyNode.scrollTop = bodyNode.scrollHeight;
  }, [messages]);

  const helperBadge = useMemo(() => {
    if (!document) {
      return "Select document";
    }
    return "Mixed QA beta";
  }, [document]);

  async function handleSubmit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (!document || !activeSession || isSubmitting) {
      return;
    }

    const question = draft.trim();
    if (!question) {
      return;
    }

    const userMessage: ChatThreadMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: question,
      sections: [],
      citations: [],
      route: null,
      retrievalMode: null,
      telemetry: null,
    };
    const pendingMessageId = `${Date.now()}-assistant`;
    const documentId = document.document_id;
    const sessionId = activeSession.sessionId;

    appendMessages(documentId, [
      userMessage,
      {
        id: pendingMessageId,
        role: "assistant",
        content: "",
        sections: [],
        citations: [],
        route: null,
        retrievalMode: null,
        telemetry: null,
        streamStatus: "正在结合当前文档与对话上下文生成回答...",
        isPending: true,
      },
    ]);
    setDraft("");
    setIsSubmitting(true);

    try {
      await askDemoChatStream(documentId, sessionId, question, {
        onStatus: (event) => {
          updateMessage(documentId, pendingMessageId, (message) => ({
            ...message,
            streamStatus: event.message,
          }));
        },
        onDelta: (event) => {
          updateMessage(documentId, pendingMessageId, (message) => ({
            ...message,
            content: `${message.content}${event.text}`,
            streamStatus: undefined,
          }));
        },
        onCitations: (event) => {
          updateMessage(documentId, pendingMessageId, (message) => ({
            ...message,
            citations: event.citations,
          }));
        },
        onTelemetry: (event) => {
          updateMessage(documentId, pendingMessageId, (message) => ({
            ...message,
            telemetry: event.telemetry,
          }));
        },
        onDone: (event) => {
          const response = event.response;
          replacePendingMessage(documentId, pendingMessageId, {
            id: pendingMessageId,
            role: "assistant",
            content: response.answer,
            sections: response.sections,
            citations: response.citations,
            route: response.route,
            retrievalMode: response.retrieval_mode,
            telemetry: response.telemetry,
          });
          setSessions((current) => {
            const currentSession = current[documentId];
            if (!currentSession) {
              return current;
            }
            return {
              ...current,
              [documentId]: {
                ...currentSession,
                sessionId: response.session_id,
              },
            };
          });
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "问答请求失败。";
      replacePendingMessage(documentId, pendingMessageId, {
        id: pendingMessageId,
        role: "assistant",
        content: message,
        sections: [],
        citations: [],
        route: null,
        retrievalMode: null,
        telemetry: null,
        isError: true,
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSubmit();
    }
  }

  function appendMessages(documentId: string, newMessages: ChatThreadMessage[]) {
    setSessions((current) => {
      const currentSession = current[documentId] ?? {
        sessionId: createSessionId(),
        messages: [],
      };
      return {
        ...current,
        [documentId]: {
          ...currentSession,
          messages: [...currentSession.messages, ...newMessages],
        },
      };
    });
  }

  function updateMessage(
    documentId: string,
    messageId: string,
    updater: (message: ChatThreadMessage) => ChatThreadMessage,
  ) {
    setSessions((current) => {
      const currentSession = current[documentId];
      if (!currentSession) {
        return current;
      }
      return {
        ...current,
        [documentId]: {
          ...currentSession,
          messages: currentSession.messages.map((message) => (message.id === messageId ? updater(message) : message)),
        },
      };
    });
  }

  function replacePendingMessage(documentId: string, pendingId: string, nextMessage: ChatThreadMessage) {
    setSessions((current) => {
      const currentSession = current[documentId];
      if (!currentSession) {
        return current;
      }
      return {
        ...current,
        [documentId]: {
          ...currentSession,
          messages: currentSession.messages.map((message) => (message.id === pendingId ? nextMessage : message)),
        },
      };
    });
  }

  function handleClearConversation() {
    if (!currentDocumentId) {
      return;
    }
    setSessions((current) => ({
      ...current,
      [currentDocumentId]: {
        sessionId: createSessionId(),
        messages: [],
      },
    }));
    setDraft("");
  }

  function renderRetrievalMode(mode: ChatResponse["retrieval_mode"] | null): string | null {
    if (!mode) {
      return null;
    }
    if (mode === "semantic_with_keyword_fallback") {
      return "关键词回退";
    }
    if (mode === "external_web_search") {
      return "外部检索";
    }
    if (mode === "external_search_unavailable") {
      return "外部检索不可用";
    }
    if (mode === "mixed_document_external") {
      return "Mixed QA";
    }
    if (mode === "unsupported") {
      return "超出范围";
    }
    return "语义检索";
  }

  return (
    <section className="chat-shell">
      <div className="chat-shell__header">
        <div>
          <p className="panel-card__kicker">Ask Filing</p>
          <h3>对话问答</h3>
        </div>
        <div className="chat-shell__header-actions">
          <span className="status-chip status-chip--muted">{helperBadge}</span>
          <button
            type="button"
            className="secondary-button secondary-button--compact"
            onClick={handleClearConversation}
            disabled={!document || isSubmitting}
          >
            清空对话
          </button>
        </div>
      </div>

      <div ref={bodyRef} className="chat-shell__body">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <p>系统会先结合当前文档与最近对话，再判断问题类型、检索证据并生成回答。</p>
          </div>
        ) : (
          <div className="chat-thread">
            {messages.map((message) => (
              <article
                key={message.id}
                className={`chat-message chat-message--${message.role} ${message.isError ? "chat-message--error" : ""}`}
              >
                <div className="chat-message__meta">
                  <strong>{message.role === "user" ? "你" : "FilingDelta"}</strong>
                  {renderRetrievalMode(message.retrievalMode) ? (
                    <span className="chat-message__mode">{renderRetrievalMode(message.retrievalMode)}</span>
                  ) : null}
                </div>
                {message.streamStatus ? <p className="chat-message__status">{message.streamStatus}</p> : null}
                {message.content ? (
                  <div className="chat-message__content">
                    <MarkdownText text={message.content} />
                  </div>
                ) : null}

                {message.sections.length > 0 ? (
                  <div className="chat-message__sections">
                    {message.sections.map((section) => (
                      <section key={`${message.id}:${section.section_type}`} className="chat-message__section">
                        <h4>{section.title}</h4>
                        <ul>
                          {section.items.map((item, index) => (
                            <li key={`${message.id}:${section.section_type}:${index}`}>
                              <MarkdownInline text={item} />
                            </li>
                          ))}
                        </ul>
                      </section>
                    ))}
                  </div>
                ) : null}

                {message.citations.length > 0 ? (
                  <div className="chat-message__citations">
                    {message.citations.map((citation, index) =>
                      citation.source_type === "document" ? (
                        <button
                          key={`${message.id}:${citation.citation_id}:${index}`}
                          type="button"
                          className="chat-citation-chip"
                          onClick={() =>
                            onSelectCitation({
                              kind: "chat",
                              id: `chat:${message.id}:${citation.citation_id}`,
                              title: `问答引用 ${index + 1}`,
                              page: citation.page_number,
                              quote: citation.quote,
                            })
                          }
                        >
                          {citation.page_number ? `第 ${citation.page_number} 页` : "文档引用"}
                        </button>
                      ) : (
                        <a
                          key={`${message.id}:${citation.citation_id}:${index}`}
                          className="chat-citation-chip chat-citation-chip--external"
                          href={citation.url || "#"}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {citation.title || "外部来源"}
                        </a>
                      ),
                    )}
                  </div>
                ) : null}

                {message.telemetry ? (
                  <details className="chat-telemetry">
                    <summary>
                      调试指标
                      <span className="chat-telemetry__summary">
                        {formatLatency(message.telemetry.total_latency_ms)} · {message.telemetry.route_type}
                      </span>
                    </summary>
                    <div className="chat-telemetry__grid">
                      <div className="chat-telemetry__group">
                        <h5>总览</h5>
                        <ul>
                          <li>总耗时：{formatLatency(message.telemetry.total_latency_ms)}</li>
                          <li>路由：{message.telemetry.route_type}</li>
                          <li>成功：{message.telemetry.succeeded ? "是" : "否"}</li>
                        </ul>
                      </div>
                      <div className="chat-telemetry__group">
                        <h5>步骤耗时</h5>
                        <ul>
                          {renderStep("首轮建索引", message.telemetry.steps.index_build_ms)}
                          {renderStep("上下文化", message.telemetry.steps.contextualizer_ms)}
                          {renderStep("路由", message.telemetry.steps.router_ms)}
                          {renderStep("规划", message.telemetry.steps.planner_ms)}
                          {renderStep("文档检索", message.telemetry.steps.document_retrieval_ms)}
                          {renderStep("外部检索", message.telemetry.steps.external_search_ms)}
                          {renderStep("回答生成", message.telemetry.steps.answerer_ms)}
                          {renderStep("记忆摘要", message.telemetry.steps.memory_summarizer_ms)}
                        </ul>
                      </div>
                      <div className="chat-telemetry__group">
                        <h5>Token / Usage</h5>
                        <ul>
                          <li>LLM 输入：{message.telemetry.usage.llm_prompt_tokens}</li>
                          <li>LLM 输出：{message.telemetry.usage.llm_completion_tokens}</li>
                          <li>Embedding：{message.telemetry.usage.embedding_tokens}</li>
                          <li>Web search：{message.telemetry.usage.web_search_total_tokens}</li>
                          <li>总 token：{message.telemetry.usage.total_tokens}</li>
                        </ul>
                      </div>
                      <div className="chat-telemetry__group">
                        <h5>检索</h5>
                        <ul>
                          <li>document top-k：{message.telemetry.retrieval.document_top_k}</li>
                          <li>命中文档 chunks：{message.telemetry.retrieval.document_retrieved_chunks}</li>
                          <li>外部来源数：{message.telemetry.retrieval.external_sources_count}</li>
                          <li>使用文档引用：{message.telemetry.retrieval.used_document_citations_count}</li>
                          <li>使用外部引用：{message.telemetry.retrieval.used_external_citations_count}</li>
                        </ul>
                      </div>
                    </div>
                  </details>
                ) : null}
              </article>
            ))}
          </div>
        )}
      </div>

      <form className="chat-shell__composer" onSubmit={(event) => void handleSubmit(event)}>
        <textarea
          className="chat-shell__input"
          rows={4}
          placeholder={document ? "例如：优先股是什么？它和这份报告里的披露意味着什么？" : "先选择一份文档，再开始问答。"}
          value={draft}
          disabled={!document || isSubmitting}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          type="submit"
          className="primary-button chat-shell__submit"
          disabled={!document || isSubmitting || !draft.trim()}
        >
          {isSubmitting ? "生成中..." : "发送"}
        </button>
      </form>
    </section>
  );
}

function createSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `chat-session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatLatency(value: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(0)} ms`;
}

function renderStep(label: string, value: number | null) {
  return (
    <li key={label}>
      {label}：{formatLatency(value)}
    </li>
  );
}
