import type { CitationTarget, HeadlineMetrics } from "../../lib/types";

type HeadlineMetricsPanelProps = {
  metrics: HeadlineMetrics | null;
  showVerifiedOnly: boolean;
  activeTargetId: string | null;
  onSelect: (target: CitationTarget) => void;
};

const METRIC_CONFIG = [
  { key: "revenue", label: "营业收入" },
  { key: "net_profit", label: "归属股东净利润" },
  { key: "unit", label: "单位" },
  { key: "fiscal_period", label: "报告期" },
] as const;

export function HeadlineMetricsPanel({
  metrics,
  showVerifiedOnly,
  activeTargetId,
  onSelect,
}: HeadlineMetricsPanelProps) {
  const visibleMetrics = METRIC_CONFIG.filter(({ key }) => {
    const field = metrics?.[key];
    if (!field) {
      return true;
    }
    if (!showVerifiedOnly) {
      return true;
    }
    if (field.value === null || field.value === "") {
      return false;
    }
    return (field.citations?.length ?? 0) > 0;
  });

  return (
    <section className="panel-card">
      <div className="panel-card__header">
        <p className="panel-card__kicker">Headline Metrics</p>
        <h3>关键数据</h3>
      </div>
      <div className="metrics-grid">
        {visibleMetrics.map(({ key, label }) => {
          const field = metrics?.[key];
          const citation = field?.citations?.[0];
          const targetId = `metric:${key}`;
          const isActive = activeTargetId === targetId;
          return (
            <button
              key={key}
              type="button"
              className={isActive ? "metric-card metric-card--active" : "metric-card"}
              disabled={!field}
              onClick={() =>
                onSelect({
                  kind: "metric",
                  id: targetId,
                  title: label,
                  page: citation?.page_number ?? field?.evidence_page ?? null,
                  quote: citation?.quote || field?.evidence_quote || "暂无引用片段。",
                  value: field?.value ?? null,
                })
              }
            >
              <span className="metric-card__label">{label}</span>
              <strong className="metric-card__value">{formatMetricValue(field?.value ?? null)}</strong>
            </button>
          );
        })}
      </div>
      {showVerifiedOnly && visibleMetrics.length === 0 ? (
        <div className="empty-inline">当前仅显示已核验内容，暂无可展示的关键数据。</div>
      ) : null}
    </section>
  );
}

function formatMetricValue(value: string | number | null): string {
  if (value === null || value === "") {
    return "—";
  }
  if (typeof value === "number") {
    return new Intl.NumberFormat("zh-CN", {
      maximumFractionDigits: 2,
    }).format(value);
  }
  return value;
}
