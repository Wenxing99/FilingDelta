import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import { askDemoChat } from "../../lib/api";
import type { CitationTarget, ChatCitation, ChatResponse, DemoDocument } from "../../lib/types";

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
  isPending?: boolean;
  isError?: boolean;
};

export function ChatPanel({ document, onSelectCitation }: ChatPanelProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<ChatThreadMessage[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    setDraft("");
    setMessages([]);
  }, [document?.document_id]);

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
    if (!document || isSubmitting) {
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
    };
    const pendingMessageId = `${Date.now()}-assistant`;

    setMessages((current) => [
      ...current,
      userMessage,
      {
        id: pendingMessageId,
        role: "assistant",
        content: "正在按问题类型规划证据源，并生成回答...",
        sections: [],
        citations: [],
        route: null,
        retrievalMode: null,
        isPending: true,
      },
    ]);
    setDraft("");
    setIsSubmitting(true);

    try {
      const response = await askDemoChat(document.document_id, question);
      setMessages((current) =>
        current.map((message) =>
          message.id === pendingMessageId
            ? {
                id: pendingMessageId,
                role: "assistant",
                content: response.answer,
                sections: response.sections,
                citations: response.citations,
                route: response.route,
                retrievalMode: response.retrieval_mode,
              }
            : message,
        ),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "问答请求失败。";
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingMessageId
            ? {
                id: pendingMessageId,
                role: "assistant",
                content: message,
                sections: [],
                citations: [],
                route: null,
                retrievalMode: null,
                isError: true,
              }
            : item,
        ),
      );
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
        <span className="status-chip status-chip--muted">{helperBadge}</span>
      </div>

      <div ref={bodyRef} className="chat-shell__body">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <p>系统会先判断问题类型，再组合文档证据与外部来源生成回答。</p>
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
                <p className="chat-message__content">{message.content}</p>

                {message.sections.length > 0 ? (
                  <div className="chat-message__sections">
                    {message.sections.map((section) => (
                      <section key={`${message.id}:${section.section_type}`} className="chat-message__section">
                        <h4>{section.title}</h4>
                        <ul>
                          {section.items.map((item, index) => (
                            <li key={`${message.id}:${section.section_type}:${index}`}>{item}</li>
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
              </article>
            ))}
          </div>
        )}
      </div>

      <form className="chat-shell__composer" onSubmit={(event) => void handleSubmit(event)}>
        <textarea
          className="chat-shell__input"
          rows={4}
          placeholder={document ? "例如：优先股是什么？它和这份报告的披露信息意味着什么？" : "先选择一份文档，再开始问答。"}
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
          {isSubmitting ? "检索中..." : "发送"}
        </button>
      </form>
    </section>
  );
}
