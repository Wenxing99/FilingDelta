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
  citations: ChatCitation[];
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
    return "RAG beta";
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
      citations: [],
      retrievalMode: null,
    };
    const pendingMessageId = `${Date.now()}-assistant`;

    setMessages((current) => [
      ...current,
      userMessage,
      {
        id: pendingMessageId,
        role: "assistant",
        content: "正在检索当前文档并生成回答...",
        citations: [],
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
                citations: response.citations,
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
                citations: [],
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
    if (mode === "semantic_with_filters_and_keyword_fallback") {
      return "语义检索 + 关键词补充";
    }
    if (mode === "semantic_with_keyword_fallback") {
      return "关键词回退";
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
            <p>基于当前文档提问，系统会先检索相关 chunk，再生成带 citation 的回答。</p>
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

                {message.citations.length > 0 ? (
                  <div className="chat-message__citations">
                    {message.citations.map((citation, index) => (
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
                        {citation.page_number ? `第 ${citation.page_number} 页` : "引用"}
                      </button>
                    ))}
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
          placeholder={
            document
              ? "例如：这份报告里对股息是怎么说的？"
              : "先选择一份文档，再开始问答。"
          }
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
