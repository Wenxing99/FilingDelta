import { StatusBadge } from "../../components/StatusBadge";
import type { DemoDocument } from "../../lib/types";

type DocumentOverviewProps = {
  document: DemoDocument | null;
};

export function DocumentOverview({ document }: DocumentOverviewProps) {
  return (
    <section className="panel-card">
      <div className="panel-card__header panel-card__header--split">
        <div>
          <p className="panel-card__kicker">Document Overview</p>
          <h3>{document?.company_name || "未选择文档"}</h3>
        </div>
        {document ? <StatusBadge label={marketLabel(document.market)} /> : null}
      </div>
      {document ? (
        <dl className="meta-list">
          <div>
            <dt>文档</dt>
            <dd>{document.label}</dd>
          </div>
          <div>
            <dt>Ticker</dt>
            <dd>{document.ticker || "—"}</dd>
          </div>
          <div>
            <dt>类型</dt>
            <dd>{document.doc_type}</dd>
          </div>
          <div>
            <dt>报告期</dt>
            <dd>{document.fiscal_period || "—"}</dd>
          </div>
        </dl>
      ) : (
        <p className="empty-inline">先选择一份样例文档。</p>
      )}
    </section>
  );
}

function marketLabel(market: string): string {
  if (market === "a_share") return "A 股";
  if (market === "h_share") return "H 股";
  if (market === "adr") return "中概 / ADR";
  return market;
}
