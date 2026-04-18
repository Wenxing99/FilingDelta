import type { CitationTarget, SummaryItem } from "../../lib/types";

type SummaryListProps = {
  items: SummaryItem[];
  activeTargetId: string | null;
  onSelect: (target: CitationTarget) => void;
};

export function SummaryList({ items, activeTargetId, onSelect }: SummaryListProps) {
  return (
    <section className="panel-card panel-card--stretch">
      <div className="panel-card__header">
        <p className="panel-card__kicker">Key Summaries</p>
        <h3>重点摘要</h3>
      </div>
      <div className="summary-list">
        {items.length === 0 ? (
          <div className="empty-inline">分析完成后，这里会显示结构化摘要。</div>
        ) : (
          items.map((item, index) => {
            const citation = item.citations[0];
            const targetId = `summary:${index + 1}`;
            const isActive = activeTargetId === targetId;
            return (
              <button
                key={targetId}
                type="button"
                className={isActive ? "summary-card summary-card--active" : "summary-card"}
                onClick={() =>
                  onSelect({
                    kind: "summary",
                    id: targetId,
                    title: item.title,
                    page: citation?.page_number ?? null,
                    quote: citation?.quote || "暂无引用片段。",
                  })
                }
              >
                <div className="summary-card__title-row">
                  <strong>{item.title}</strong>
                  <span className="summary-card__page">
                    {citation?.page_number ? `第 ${citation.page_number} 页` : "待核验"}
                  </span>
                </div>
                <p>{item.summary}</p>
              </button>
            );
          })
        )}
      </div>
    </section>
  );
}
