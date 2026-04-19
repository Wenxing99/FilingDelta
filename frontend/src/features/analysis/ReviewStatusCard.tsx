import { useState } from "react";

import { StatusBadge } from "../../components/StatusBadge";
import type { DemoRun } from "../../lib/types";

type ReviewStatusCardProps = {
  run: DemoRun | null;
  showVerifiedOnly: boolean;
  activeIssueActionKey: string | null;
  activeFeedbackActionKey: string | null;
  onToggleVerifiedOnly: () => void;
  onApproveIssue: (itemKey: string) => void;
  onRerunIssue: (itemKey: string) => void;
  onFeedback: (category: "citation" | "numeric" | "summary") => void;
};

export function ReviewStatusCard({
  run,
  showVerifiedOnly,
  activeIssueActionKey,
  activeFeedbackActionKey,
  onToggleVerifiedOnly,
  onApproveIssue,
  onRerunIssue,
  onFeedback,
}: ReviewStatusCardProps) {
  const [showPendingItems, setShowPendingItems] = useState(false);
  const review = run?.result?.review;
  const pendingIssues = (run?.result?.verification_issues ?? []).filter((issue) => issue.severity === "review");
  const isAnyActionBusy = activeIssueActionKey !== null || activeFeedbackActionKey !== null;

  const tone =
    run?.status === "failed"
      ? "danger"
      : review?.status === "needs_confirmation"
        ? "warning"
      : run?.result
        ? "success"
        : "neutral";

  const label =
    run?.status === "failed"
      ? "失败"
      : review?.status === "needs_confirmation"
        ? "待确认"
        : run?.result
          ? "通过"
          : "待分析";

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
      ) : (
        <>
          {run?.result ? (
            <div className="review-summary-grid">
              <div className="review-summary-stat">
                <span className="review-summary-stat__label">已通过</span>
                <strong>{review?.verified_count ?? 0} 条</strong>
              </div>
              <div className="review-summary-stat">
                <span className="review-summary-stat__label">待确认</span>
                <strong>{review?.pending_confirmation_count ?? 0} 条</strong>
              </div>
            </div>
          ) : null}

          {pendingIssues.length === 0 ? (
            <p className="review-pass">当前没有发现需要确认的问题，摘要与关键数字都已通过现有规则校验。</p>
          ) : (
            <div className="review-actions">
              <button type="button" className="secondary-button" onClick={onToggleVerifiedOnly}>
                {showVerifiedOnly ? "显示全部内容" : "仅显示已通过内容"}
              </button>
              <button
                type="button"
                className="secondary-button secondary-button--ghost"
                onClick={() => setShowPendingItems((current) => !current)}
              >
                {showPendingItems ? "收起待确认项" : `查看待确认项（${pendingIssues.length}）`}
              </button>
            </div>
          )}

          {showPendingItems && pendingIssues.length > 0 ? (
            <ul className="review-issues">
              {pendingIssues.map((issue) => {
                const rerunKey = `rerun:${issue.item_key}`;
                const approveKey = `approve:${issue.item_key}`;

                return (
                  <li key={`${issue.scope}-${issue.item_key}`}>
                    <div className="review-issue__header">
                      <strong>{issue.item_label}</strong>
                      <span className="review-issue__reason">{issue.user_visible_reason}</span>
                    </div>
                    {issue.evidence_page ? (
                      <span>
                        第 {issue.evidence_page} 页
                        {issue.evidence_quote ? ` · ${issue.evidence_quote}` : ""}
                      </span>
                    ) : (
                      <span>当前暂无可直接定位的证据片段。</span>
                    )}
                    <div className="review-issue__actions">
                      <button
                        type="button"
                        className="secondary-button secondary-button--ghost"
                        disabled={isAnyActionBusy}
                        onClick={() => onRerunIssue(issue.item_key)}
                      >
                        {activeIssueActionKey === rerunKey ? "重新生成中..." : "重新生成这一条"}
                      </button>
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={isAnyActionBusy}
                        onClick={() => onApproveIssue(issue.item_key)}
                      >
                        {activeIssueActionKey === approveKey ? "确认中..." : "标记通过"}
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          ) : null}

          {run?.result ? (
            <div className="review-feedback">
              <div className="review-feedback__header">
                <strong>整体不满意？</strong>
                <span>按问题类型重新处理最相关的链路。</span>
              </div>
              <div className="review-feedback__actions">
                <button
                  type="button"
                  className="secondary-button secondary-button--ghost"
                  disabled={isAnyActionBusy}
                  onClick={() => onFeedback("citation")}
                >
                  {activeFeedbackActionKey === "feedback:citation" ? "处理中..." : "对引用回溯不满意"}
                </button>
                <button
                  type="button"
                  className="secondary-button secondary-button--ghost"
                  disabled={isAnyActionBusy}
                  onClick={() => onFeedback("numeric")}
                >
                  {activeFeedbackActionKey === "feedback:numeric" ? "处理中..." : "对数据准确度不满意"}
                </button>
                <button
                  type="button"
                  className="secondary-button secondary-button--ghost"
                  disabled={isAnyActionBusy}
                  onClick={() => onFeedback("summary")}
                >
                  {activeFeedbackActionKey === "feedback:summary" ? "处理中..." : "对摘要信息不满意"}
                </button>
              </div>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}
