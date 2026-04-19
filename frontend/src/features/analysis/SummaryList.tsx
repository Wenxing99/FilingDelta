import type { CitationTarget, SummaryItem, SummarySection } from "../../lib/types";

type SummaryListProps = {
  overview: SummaryItem | null;
  sections: SummarySection[];
  fallbackItems: SummaryItem[];
  showVerifiedOnly: boolean;
  activeTargetId: string | null;
  onSelect: (target: CitationTarget) => void;
};

function buildSummaryTarget(
  target: {
    id: string;
    title: string;
    citationPage: number | null;
    citationQuote: string | null | undefined;
  },
  onSelect: (target: CitationTarget) => void,
) {
  onSelect({
    kind: "summary",
    id: target.id,
    title: target.title,
    page: target.citationPage,
    quote: target.citationQuote || "暂无引用片段。",
  });
}

export function SummaryList({
  overview,
  sections,
  fallbackItems,
  showVerifiedOnly,
  activeTargetId,
  onSelect,
}: SummaryListProps) {
  const filteredSections = sections
    .map((section) => ({
      ...section,
      points: showVerifiedOnly ? section.points.filter((point) => !point.needs_human_review) : section.points,
    }))
    .filter((section) => section.points.length > 0);

  const filteredOverview = showVerifiedOnly && overview?.needs_human_review ? null : overview;
  const filteredFallbackItems = showVerifiedOnly
    ? fallbackItems.filter((item) => !item.needs_human_review)
    : fallbackItems;
  const hasSectionedSummary = filteredSections.length > 0;

  return (
    <section className="panel-card panel-card--stretch">
      <div className="panel-card__header">
        <p className="panel-card__kicker">Structured Summary</p>
        <h3>结构化摘要</h3>
      </div>

      <div className="summary-list">
        {filteredOverview ? (
          <button
            type="button"
            className={activeTargetId === "summary:overview" ? "summary-card summary-card--active" : "summary-card"}
            onClick={() =>
              buildSummaryTarget(
                {
                  id: "summary:overview",
                  title: "Overview",
                  citationPage: filteredOverview.citations[0]?.page_number ?? null,
                  citationQuote: filteredOverview.citations[0]?.quote,
                },
                onSelect,
              )
            }
          >
            <div className="summary-card__title-row">
              <strong>Overview</strong>
              <span className="summary-card__page">
                {filteredOverview.citations[0]?.page_number
                  ? `第 ${filteredOverview.citations[0].page_number} 页`
                  : "待核验"}
              </span>
            </div>
            <p>{filteredOverview.summary}</p>
          </button>
        ) : null}

        {!hasSectionedSummary && filteredFallbackItems.length === 0 ? (
          <div className="empty-inline">
            {showVerifiedOnly ? "当前仅显示已通过内容，暂无可展示的摘要。" : "分析完成后，这里会显示结构化摘要。"}
          </div>
        ) : hasSectionedSummary ? (
          filteredSections.map((section) => (
            <div key={section.section_id} className="summary-section">
              <div className="summary-section__header">
                <strong>{section.title}</strong>
                <span className="summary-section__count">{section.points.length} 条</span>
              </div>

              <div className="summary-section__points">
                {section.points.map((point) => {
                  const citation = point.citations[0];
                  const targetId = `summary-point:${point.point_id}`;
                  const isActive = activeTargetId === targetId;

                  return (
                    <button
                      key={point.point_id}
                      type="button"
                      className={isActive ? "summary-point summary-point--active" : "summary-point"}
                      onClick={() =>
                        buildSummaryTarget(
                          {
                            id: targetId,
                            title: section.title,
                            citationPage: citation?.page_number ?? null,
                            citationQuote: citation?.quote,
                          },
                          onSelect,
                        )
                      }
                    >
                      <span className="summary-point__bullet" aria-hidden="true" />
                      <span className="summary-point__text">{point.text}</span>
                      <span className="summary-point__page">
                        {citation?.page_number ? `第 ${citation.page_number} 页` : "待核验"}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))
        ) : (
          filteredFallbackItems.map((item, index) => {
            const citation = item.citations[0];
            const targetId = `summary:${index + 1}`;
            const isActive = activeTargetId === targetId;

            return (
              <button
                key={targetId}
                type="button"
                className={isActive ? "summary-card summary-card--active" : "summary-card"}
                onClick={() =>
                  buildSummaryTarget(
                    {
                      id: targetId,
                      title: item.title,
                      citationPage: citation?.page_number ?? null,
                      citationQuote: citation?.quote,
                    },
                    onSelect,
                  )
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
