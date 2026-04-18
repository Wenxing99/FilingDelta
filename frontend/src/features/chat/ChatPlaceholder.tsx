export function ChatPlaceholder() {
  return (
    <section className="chat-shell">
      <div className="chat-shell__header">
        <div>
          <p className="panel-card__kicker">Ask Filing</p>
          <h3>对话问答</h3>
        </div>
        <span className="status-chip status-chip--muted">Coming soon</span>
      </div>
      <div className="chat-shell__body">
        <p>这里会接入基于全文上下文的问答能力。</p>
      </div>
      <textarea
        className="chat-shell__input"
        rows={5}
        placeholder="Chat coming soon"
        disabled
      />
    </section>
  );
}
