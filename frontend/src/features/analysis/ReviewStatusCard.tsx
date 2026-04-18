import { StatusBadge } from "../../components/StatusBadge";
import type { DemoRun } from "../../lib/types";

type ReviewStatusCardProps = {
  run: DemoRun | null;
};

export function ReviewStatusCard({ run }: ReviewStatusCardProps) {
  const tone =
    run?.status === "failed"
      ? "danger"
      : run?.result?.needs_human_review
        ? "warning"
        : run?.result
          ? "success"
          : "neutral";

  const label =
    run?.status === "failed"
      ? "失败"
      : run?.result?.needs_human_review
        ? "需复核"
        : run?.result
          ? "通过"
          : "待分析";

  const issues = run?.result?.verification_issues ?? [];

  return (
    <section className="panel-card">
      <div className="panel-card__header panel-card__header--split">
        <div>
          <p className="panel-card__kicker">Review Status</p>
          <h3>核验状态</h3>
        </div>
        <StatusBadge label={label} tone={tone} />
      </div>
      {run?.status === "failed" ? (
        <p className="review-error">{run.error_message || "本次分析失败。请稍后重试。"}</p>
      ) : issues.length === 0 ? (
        <p className="review-pass">
          当前没有发现需要人工复核的问题，摘要与关键数字都已通过现有规则校验。
        </p>
      ) : (
        <ul className="review-issues">
          {issues.map((issue) => (
            <li key={`${issue.scope}-${issue.item_key}`}>
              <strong>{issue.item_key}</strong>
              <p>{issue.message}</p>
              {issue.evidence_page ? (
                <span>
                  第 {issue.evidence_page} 页
                  {issue.evidence_quote ? ` · ${issue.evidence_quote}` : ""}
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
