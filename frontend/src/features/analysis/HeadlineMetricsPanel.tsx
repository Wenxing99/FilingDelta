import type { CitationTarget, DemoDocument, ExtractedFactField, HeadlineMetrics } from "../../lib/types";

type HeadlineMetricsPanelProps = {
  metrics: HeadlineMetrics | null;
  document: DemoDocument | null;
  showVerifiedOnly: boolean;
  activeTargetId: string | null;
  onSelect: (target: CitationTarget) => void;
};

const METRIC_CONFIG = [
  { key: "fiscal_period", label: "报告期" },
  { key: "revenue", label: "营业收入" },
  { key: "net_profit", label: "归属股东净利润" },
  { key: "roe", label: "净资产收益率" },
] as const;

type MetricKey = (typeof METRIC_CONFIG)[number]["key"];
type MetricField = ExtractedFactField | null | undefined;

export function HeadlineMetricsPanel({
  metrics,
  document,
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
              <strong className="metric-card__value">
                {formatMetricValue(key, field, metrics?.unit.value ?? null, document)}
              </strong>
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

function formatMetricValue(
  key: MetricKey,
  field: MetricField,
  unitValue: string | number | null,
  document: DemoDocument | null,
): string {
  const value = field?.value ?? null;
  if (value === null || value === "") {
    return "—";
  }

  if (key === "fiscal_period") {
    return buildFiscalPeriodDisplay(document, value);
  }

  if (key === "roe") {
    return formatPercentValue(value);
  }

  if (typeof value === "number") {
    return formatAmountValue(value, field, unitValue);
  }

  return String(value);
}

function buildFiscalPeriodDisplay(
  document: DemoDocument | null,
  fallbackValue: string | number | null,
): string {
  const rawPeriod = document?.fiscal_period || String(fallbackValue ?? "").trim();
  const normalized = rawPeriod.replace(/\s+/g, "");
  const yearMatch = normalized.match(/(20\d{2})/);
  const year = yearMatch?.[1] ?? null;
  const preciseQuarterLabel = year ? inferQuarterPeriodLabel(rawPeriod, year) : null;
  const preciseInterimLabel = year ? inferInterimPeriodLabel(rawPeriod, year) : null;

  if (document?.doc_type === "annual_report" && year) {
    return buildPeriodLabel(`${year}年度报告`, document.market);
  }
  if (document?.doc_type === "interim_report" && year) {
    if (preciseQuarterLabel) {
      return buildPeriodLabel(preciseQuarterLabel, document.market);
    }
    if (preciseInterimLabel) {
      return buildPeriodLabel(preciseInterimLabel, document.market);
    }
  }
  if (document?.doc_type === "earnings_release" && year) {
    return buildPeriodLabel(`${year}业绩公告`, document.market);
  }

  if (normalized.includes("年度报告")) {
    return normalized;
  }
  if (normalized.includes("半年度报告") || normalized.includes("中期报告")) {
    return normalized;
  }
  if (normalized.includes("季度报告")) {
    return normalized;
  }
  if (preciseQuarterLabel) {
    return buildPeriodLabel(preciseQuarterLabel, document?.market);
  }
  if (preciseInterimLabel) {
    return buildPeriodLabel(preciseInterimLabel, document?.market);
  }
  if (year && document?.doc_type === "annual_report") {
    return buildPeriodLabel(`${year}年度报告`, document.market);
  }

  return String(fallbackValue);
}

function buildPeriodLabel(label: string, market: string | null | undefined): string {
  if (market === "a_share") {
    return `${label}（A股）`;
  }
  if (market === "h_share") {
    return `${label}（H股）`;
  }
  return label;
}

function inferQuarterPeriodLabel(rawPeriod: string, year: string): string | null {
  const normalized = rawPeriod.replace(/\s+/g, "");
  const englishQuarter = normalized.match(/Q([1-4])/i);
  const chineseQuarter = normalized.match(/第([一二三四1-4])季度/);
  const plainQuarter = normalized.match(/([1-4])季度/);

  const quarterToken =
    englishQuarter?.[1] ?? chineseQuarter?.[1] ?? plainQuarter?.[1] ?? null;
  if (!quarterToken) {
    return null;
  }

  return `${year}年${normalizeQuarterToken(quarterToken)}季度报告`;
}

function inferInterimPeriodLabel(rawPeriod: string, year: string): string | null {
  if (/半年度|半年|H1/i.test(rawPeriod)) {
    return `${year}年半年度报告`;
  }
  if (/中期/i.test(rawPeriod)) {
    return `${year}年中期报告`;
  }
  return null;
}

function normalizeQuarterToken(token: string): string {
  if (token === "1" || token === "一") return "第一";
  if (token === "2" || token === "二") return "第二";
  if (token === "3" || token === "三") return "第三";
  if (token === "4" || token === "四") return "第四";
  return token;
}

function formatAmountValue(
  value: number,
  field: MetricField,
  unitValue: string | number | null,
): string {
  const unitText = inferMetricUnitText(field, unitValue);
  const normalizedUnit = unitText.replace(/\s+/g, "").toLowerCase();
  if (!normalizedUnit) {
    return new Intl.NumberFormat("zh-CN", {
      maximumFractionDigits: 2,
    }).format(value);
  }

  const resolvedUnit = resolveAmountUnit(normalizedUnit);
  if (!resolvedUnit) {
    const formattedRawValue = new Intl.NumberFormat("zh-CN", {
      maximumFractionDigits: 2,
    }).format(value);
    return `${formattedRawValue} ${unitText}`.trim();
  }

  const { displayUnit, divisor } = resolvedUnit;
  const displayValue = divisor > 0 ? value / divisor : value;
  const formattedValue = new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: displayValue >= 1000 ? 0 : 2,
    maximumFractionDigits: 2,
  }).format(displayValue);
  return `${formattedValue} ${displayUnit}`.trim();
}

function inferMetricUnitText(
  field: MetricField,
  fallbackUnit: string | number | null,
): string {
  const quote = field?.evidence_quote || field?.citations?.[0]?.quote || "";
  const normalizedQuote = quote.replace(/\s+/g, "").toLowerCase();

  if (
    normalizedQuote.includes("million")
    || normalizedQuote.includes("百万元")
    || normalizedQuote.includes("百萬元")
  ) {
    if (
      normalizedQuote.includes("港元")
      || normalizedQuote.includes("港幣")
      || normalizedQuote.includes("港币")
      || normalizedQuote.includes("hkd")
    ) {
      return "港币百万元";
    }
    if (normalizedQuote.includes("美元") || normalizedQuote.includes("usd")) {
      return "美元百万元";
    }
    if (normalizedQuote.includes("欧元") || normalizedQuote.includes("歐元") || normalizedQuote.includes("eur")) {
      return "欧元百万元";
    }
    if (
      normalizedQuote.includes("人民币")
      || normalizedQuote.includes("人民幣")
      || normalizedQuote.includes("rmb")
      || normalizedQuote.includes("cny")
    ) {
      return "人民币百万元";
    }
    return "百万元";
  }

  if (normalizedQuote.includes("thousand") || normalizedQuote.includes("千元")) {
    if (
      normalizedQuote.includes("港元")
      || normalizedQuote.includes("港幣")
      || normalizedQuote.includes("港币")
      || normalizedQuote.includes("hkd")
    ) {
      return "港币千元";
    }
    if (normalizedQuote.includes("美元") || normalizedQuote.includes("usd")) {
      return "美元千元";
    }
    if (normalizedQuote.includes("欧元") || normalizedQuote.includes("歐元") || normalizedQuote.includes("eur")) {
      return "欧元千元";
    }
    return "千元";
  }

  if (normalizedQuote.includes("亿港元")) {
    return "亿港元";
  }
  if (normalizedQuote.includes("亿美元")) {
    return "亿美元";
  }
  if (normalizedQuote.includes("亿欧元")) {
    return "亿欧元";
  }
  if (normalizedQuote.includes("亿元") || normalizedQuote.includes("億元")) {
    return "亿元";
  }
  if (normalizedQuote.includes("百万元") || normalizedQuote.includes("百萬元")) {
    return "百万元";
  }
  if (normalizedQuote.includes("万元") || normalizedQuote.includes("萬元")) {
    return "万元";
  }
  return typeof fallbackUnit === "string" ? fallbackUnit : "";
}

function formatPercentValue(value: string | number): string {
  const numericValue = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numericValue)) {
    return String(value);
  }
  return `${new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: 2,
  }).format(numericValue)}%`;
}

function inferDisplayUnit(normalizedUnit: string): string {
  if (
    normalizedUnit.includes("港元")
    || normalizedUnit.includes("港幣")
    || normalizedUnit.includes("港币")
    || normalizedUnit.includes("hkd")
  ) {
    return "亿港元";
  }
  if (normalizedUnit.includes("美元") || normalizedUnit.includes("usd")) {
    return "亿美元";
  }
  if (normalizedUnit.includes("欧元") || normalizedUnit.includes("歐元") || normalizedUnit.includes("eur")) {
    return "亿欧元";
  }
  return "亿元";
}

function inferAmountDivisor(normalizedUnit: string): number {
  if (normalizedUnit.includes("亿") || normalizedUnit.includes("億")) {
    return 1;
  }
  if (
    normalizedUnit.includes("百万")
    || normalizedUnit.includes("百萬")
    || normalizedUnit.includes("million")
  ) {
    return 100;
  }
  if (normalizedUnit.includes("万元") || normalizedUnit.includes("萬元")) {
    return 10_000;
  }
  if (normalizedUnit.includes("千元") || normalizedUnit.includes("thousand")) {
    return 100_000;
  }
  if (normalizedUnit) {
    return 100_000_000;
  }
  return 1;
}

function resolveAmountUnit(normalizedUnit: string): { displayUnit: string; divisor: number } | null {
  if (!normalizedUnit) {
    return null;
  }

  return {
    displayUnit: inferDisplayUnit(normalizedUnit),
    divisor: inferAmountDivisor(normalizedUnit),
  };
}
