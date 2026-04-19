import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, MouseEvent as ReactMouseEvent } from "react";

import { AnalysisProgressCard } from "../features/analysis/AnalysisProgressCard";
import { ChatPlaceholder } from "../features/chat/ChatPlaceholder";
import { HeadlineMetricsPanel } from "../features/analysis/HeadlineMetricsPanel";
import { ReviewStatusCard } from "../features/analysis/ReviewStatusCard";
import { SummaryList } from "../features/analysis/SummaryList";
import { DocumentOverview } from "../features/documents/DocumentOverview";
import { DocumentSelector } from "../features/documents/DocumentSelector";
import { CitationDetailBar } from "../features/viewer/CitationDetailBar";
import { DocumentViewer } from "../features/viewer/DocumentViewer";
import {
  approveDemoRunIssue,
  createDemoRun,
  getDemoRun,
  listDemoDocuments,
  rerunDemoRunFeedback,
  rerunDemoRunIssue,
} from "../lib/api";
import type { CitationTarget, DemoDocument, DemoRun, WorkflowResult } from "../lib/types";

export default function App() {
  const workspaceRef = useRef<HTMLElement | null>(null);
  const resizeStateRef = useRef<{
    side: "left" | "right";
    startX: number;
    startAnalysisWidth: number;
    startChatWidth: number;
  } | null>(null);
  const [documents, setDocuments] = useState<DemoDocument[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [run, setRun] = useState<DemoRun | null>(null);
  const [activeCitationTarget, setActiveCitationTarget] = useState<CitationTarget | null>(null);
  const [isStartingRun, setIsStartingRun] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const [showVerifiedOnly, setShowVerifiedOnly] = useState(false);
  const [activeIssueActionKey, setActiveIssueActionKey] = useState<string | null>(null);
  const [activeFeedbackActionKey, setActiveFeedbackActionKey] = useState<string | null>(null);
  const [analysisColumnWidth, setAnalysisColumnWidth] = useState(340);
  const [chatColumnWidth, setChatColumnWidth] = useState(300);
  const [isResizingColumns, setIsResizingColumns] = useState(false);
  const [workspaceWidth, setWorkspaceWidth] = useState(0);

  const selectedDocument = useMemo(
    () => documents.find((item) => item.document_id === selectedDocumentId) ?? null,
    [documents, selectedDocumentId],
  );

  const result = run?.result ?? null;
  const isRunning = run?.status === "queued" || run?.status === "running";
  const shouldShowAnalysisProgress = Boolean(run && (run.status === "queued" || run.status === "running"));
  const workspaceStyle = useMemo(
    () =>
      ({
        "--analysis-width": `${analysisColumnWidth}px`,
        "--chat-width": `${chatColumnWidth}px`,
      }) as CSSProperties,
    [analysisColumnWidth, chatColumnWidth],
  );

  useEffect(() => {
    let ignore = false;

    async function bootstrap() {
      try {
        const loadedDocuments = await listDemoDocuments();
        if (ignore) {
          return;
        }
        setDocuments(loadedDocuments);
        if (loadedDocuments.length > 0) {
          setSelectedDocumentId((current) => current || loadedDocuments[0].document_id);
        }
      } catch (error) {
        if (ignore) {
          return;
        }
        setPageError(error instanceof Error ? error.message : "加载样例文档失败。");
      }
    }

    void bootstrap();
    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (!run || run.status === "succeeded" || run.status === "failed") {
      return undefined;
    }

    const timer = window.setTimeout(async () => {
      try {
        const nextRun = await getDemoRun(run.run_id);
        setRun(nextRun);
      } catch (error) {
        setPageError(error instanceof Error ? error.message : "刷新分析状态失败。");
      }
    }, 1000);

    return () => window.clearTimeout(timer);
  }, [run]);

  useEffect(() => {
    if (!result) {
      return;
    }
    setActiveCitationTarget((current) => {
      if (!current) {
        return buildInitialCitationTarget(result, showVerifiedOnly);
      }
      return refreshCitationTarget(current, result, showVerifiedOnly);
    });
  }, [result, showVerifiedOnly]);

  useEffect(() => {
    const workspaceNode = workspaceRef.current;
    if (!workspaceNode) {
      return undefined;
    }

    const syncWorkspaceWidth = () => {
      setWorkspaceWidth(workspaceNode.clientWidth);
    };

    syncWorkspaceWidth();

    const observer = new ResizeObserver(() => {
      syncWorkspaceWidth();
    });

    observer.observe(workspaceNode);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (workspaceWidth === 0) {
      return;
    }

    const sideSpaceBudget = workspaceWidth - CENTER_COLUMN_MIN_WIDTH - RESIZER_LAYOUT_OVERHEAD;
    if (sideSpaceBudget <= LEFT_COLUMN_MIN_WIDTH + RIGHT_COLUMN_MIN_WIDTH) {
      return;
    }

    const nextAnalysisWidth = clamp(
      analysisColumnWidth,
      LEFT_COLUMN_MIN_WIDTH,
      Math.min(LEFT_COLUMN_MAX_WIDTH, sideSpaceBudget - RIGHT_COLUMN_MIN_WIDTH),
    );
    const nextChatWidth = clamp(
      chatColumnWidth,
      RIGHT_COLUMN_MIN_WIDTH,
      Math.min(RIGHT_COLUMN_MAX_WIDTH, sideSpaceBudget - nextAnalysisWidth),
    );
    const finalAnalysisWidth = clamp(
      nextAnalysisWidth,
      LEFT_COLUMN_MIN_WIDTH,
      Math.min(LEFT_COLUMN_MAX_WIDTH, sideSpaceBudget - nextChatWidth),
    );

    if (finalAnalysisWidth !== analysisColumnWidth) {
      setAnalysisColumnWidth(finalAnalysisWidth);
    }
    if (nextChatWidth !== chatColumnWidth) {
      setChatColumnWidth(nextChatWidth);
    }
  }, [analysisColumnWidth, chatColumnWidth, workspaceWidth]);

  useEffect(() => {
    if (!isResizingColumns) {
      return undefined;
    }

    const handlePointerMove = (event: MouseEvent) => {
      const resizeState = resizeStateRef.current;
      const workspaceNode = workspaceRef.current;
      if (!resizeState || !workspaceNode) {
        return;
      }

      const totalWidth = workspaceNode.clientWidth;
      const maxLeftWidth = Math.max(
        LEFT_COLUMN_MIN_WIDTH,
        totalWidth - resizeState.startChatWidth - CENTER_COLUMN_MIN_WIDTH - RESIZER_LAYOUT_OVERHEAD,
      );
      const maxRightWidth = Math.max(
        RIGHT_COLUMN_MIN_WIDTH,
        totalWidth - resizeState.startAnalysisWidth - CENTER_COLUMN_MIN_WIDTH - RESIZER_LAYOUT_OVERHEAD,
      );

      if (resizeState.side === "left") {
        const nextWidth = resizeState.startAnalysisWidth + (event.clientX - resizeState.startX);
        setAnalysisColumnWidth(clamp(nextWidth, LEFT_COLUMN_MIN_WIDTH, Math.min(LEFT_COLUMN_MAX_WIDTH, maxLeftWidth)));
        return;
      }

      const nextWidth = resizeState.startChatWidth - (event.clientX - resizeState.startX);
      setChatColumnWidth(clamp(nextWidth, RIGHT_COLUMN_MIN_WIDTH, Math.min(RIGHT_COLUMN_MAX_WIDTH, maxRightWidth)));
    };

    const handlePointerUp = () => {
      resizeStateRef.current = null;
      setIsResizingColumns(false);
    };

    window.addEventListener("mousemove", handlePointerMove);
    window.addEventListener("mouseup", handlePointerUp);
    document.body.style.cursor = "col-resize";

    return () => {
      window.removeEventListener("mousemove", handlePointerMove);
      window.removeEventListener("mouseup", handlePointerUp);
      document.body.style.cursor = "";
    };
  }, [isResizingColumns]);

  function handleResizeStart(side: "left" | "right", event: ReactMouseEvent<HTMLDivElement>) {
    event.preventDefault();
    resizeStateRef.current = {
      side,
      startX: event.clientX,
      startAnalysisWidth: analysisColumnWidth,
      startChatWidth: chatColumnWidth,
    };
    setIsResizingColumns(true);
  }

  async function handleStartRun() {
    if (!selectedDocumentId) {
      return;
    }

    try {
      setPageError(null);
      setIsStartingRun(true);
      setActiveCitationTarget(null);
      setShowVerifiedOnly(false);
      const nextRun = await createDemoRun(selectedDocumentId);
      setRun(nextRun);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "启动分析失败。");
    } finally {
      setIsStartingRun(false);
    }
  }

  async function handleApproveIssue(itemKey: string) {
    if (!run) {
      return;
    }

    try {
      setPageError(null);
      setActiveIssueActionKey(`approve:${itemKey}`);
      const nextRun = await approveDemoRunIssue(run.run_id, itemKey);
      setRun(nextRun);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "手动确认待确认项失败。");
    } finally {
      setActiveIssueActionKey(null);
    }
  }

  async function handleRerunIssue(itemKey: string) {
    if (!run) {
      return;
    }

    try {
      setPageError(null);
      setActiveIssueActionKey(`rerun:${itemKey}`);
      const nextRun = await rerunDemoRunIssue(run.run_id, itemKey);
      setRun(nextRun);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "重新处理待确认项失败。");
    } finally {
      setActiveIssueActionKey(null);
    }
  }

  async function handleFeedback(category: "citation" | "numeric" | "summary") {
    if (!run) {
      return;
    }

    try {
      setPageError(null);
      setActiveFeedbackActionKey(`feedback:${category}`);
      const nextRun = await rerunDemoRunFeedback(run.run_id, category);
      setRun(nextRun);
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "按问题类型重新处理失败。");
    } finally {
      setActiveFeedbackActionKey(null);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar topbar--compact">
        <div className="topbar__identity">
          <p className="topbar__kicker">FilingDelta</p>
          <h1>可追溯信披阅读工作台</h1>
        </div>
        <div className="topbar__controls">
          <DocumentSelector
            documents={documents}
            selectedDocumentId={selectedDocumentId}
            disabled={isRunning || isStartingRun}
            onChange={(documentId) => {
              setSelectedDocumentId(documentId);
              setRun(null);
              setActiveCitationTarget(null);
              setShowVerifiedOnly(false);
            }}
          />
          <button
            type="button"
            className="primary-button"
            disabled={!selectedDocumentId || isRunning || isStartingRun}
            onClick={() => void handleStartRun()}
          >
            {isRunning || isStartingRun ? "分析中..." : "开始分析"}
          </button>
        </div>
      </header>

      {pageError ? <div className="page-error">{pageError}</div> : null}

      <main
        ref={workspaceRef}
        className={`workspace ${isResizingColumns ? "workspace--resizing" : ""}`}
        style={workspaceStyle}
      >
        <aside className="analysis-column">
          {shouldShowAnalysisProgress && run ? <AnalysisProgressCard run={run} /> : null}
          <DocumentOverview document={selectedDocument} />
          <HeadlineMetricsPanel
            metrics={result?.headline_metrics ?? null}
            showVerifiedOnly={showVerifiedOnly}
            activeTargetId={activeCitationTarget?.id ?? null}
            onSelect={setActiveCitationTarget}
          />
          <SummaryList
            overview={result?.overview ?? null}
            sections={result?.summary_sections ?? []}
            fallbackItems={result?.summary_items ?? []}
            showVerifiedOnly={showVerifiedOnly}
            activeTargetId={activeCitationTarget?.id ?? null}
            onSelect={setActiveCitationTarget}
          />
          <ReviewStatusCard
            run={run}
            showVerifiedOnly={showVerifiedOnly}
            activeIssueActionKey={activeIssueActionKey}
            activeFeedbackActionKey={activeFeedbackActionKey}
            onToggleVerifiedOnly={() => setShowVerifiedOnly((current) => !current)}
            onApproveIssue={(itemKey) => void handleApproveIssue(itemKey)}
            onRerunIssue={(itemKey) => void handleRerunIssue(itemKey)}
            onFeedback={(category) => void handleFeedback(category)}
          />
        </aside>

        <div
          role="separator"
          aria-label="调整左侧分析栏宽度"
          aria-orientation="vertical"
          className="column-resizer"
          onMouseDown={(event) => handleResizeStart("left", event)}
        />

        <section className="viewer-column">
          <div className="viewer-column__header viewer-column__header--compact">
            <div>
              <p className="panel-card__kicker">Document Viewer</p>
              <h2>{selectedDocument?.label || "原始文档"}</h2>
            </div>
            <span className="viewer-column__hint">点左侧摘要或数字查看对应证据。</span>
          </div>
          <div className="viewer-stage">
            <DocumentViewer document={selectedDocument} citationTarget={activeCitationTarget} />
          </div>
          <CitationDetailBar target={activeCitationTarget} />
        </section>

        <div
          role="separator"
          aria-label="调整右侧问答栏宽度"
          aria-orientation="vertical"
          className="column-resizer"
          onMouseDown={(event) => handleResizeStart("right", event)}
        />

        <aside className="chat-column">
          <ChatPlaceholder />
        </aside>
      </main>
    </div>
  );
}

function buildInitialCitationTarget(result: WorkflowResult, showVerifiedOnly = false): CitationTarget | null {
  if (result.overview?.citations[0] && (!showVerifiedOnly || !result.overview.needs_human_review)) {
    return {
      kind: "summary",
      id: "summary:overview",
      title: "Overview",
      page: result.overview.citations[0].page_number,
      quote: result.overview.citations[0].quote,
    };
  }

  for (const section of result.summary_sections) {
    for (const point of section.points) {
      if (showVerifiedOnly && point.needs_human_review) {
        continue;
      }
      const citation = point.citations[0];
      if (!citation) {
        continue;
      }
      return {
        kind: "summary",
        id: `summary-point:${point.point_id}`,
        title: section.title,
        page: citation.page_number,
        quote: citation.quote,
      };
    }
  }

  const firstSummaryIndex = result.summary_items.findIndex((item) => item.citations.length > 0);
  if (firstSummaryIndex >= 0) {
    const firstSummary = result.summary_items[firstSummaryIndex];
    return {
      kind: "summary",
      id: `summary:${firstSummaryIndex + 1}`,
      title: firstSummary.title,
      page: firstSummary.citations[0].page_number,
      quote: firstSummary.citations[0].quote,
    };
  }

  const metricEntries = [
    ["revenue", "营业收入", result.headline_metrics.revenue],
    ["net_profit", "归属于股东净利润", result.headline_metrics.net_profit],
    ["unit", "单位", result.headline_metrics.unit],
    ["fiscal_period", "报告期", result.headline_metrics.fiscal_period],
  ] as const;

  for (const [key, label, field] of metricEntries) {
    if (showVerifiedOnly && (field.value === null || !field.citations[0])) {
      continue;
    }
    const citation = field.citations[0];
    if (!citation) {
      continue;
    }
    return {
      kind: "metric",
      id: `metric:${key}`,
      title: label,
      page: citation.page_number,
      quote: citation.quote,
      value: field.value,
    };
  }

  return null;
}

function isCitationTargetVisible(target: CitationTarget, result: WorkflowResult): boolean {
  if (target.kind === "metric") {
    const metricKey = target.id.replace("metric:", "") as
      | "revenue"
      | "net_profit"
      | "unit"
      | "fiscal_period";
    const field = result.headline_metrics[metricKey];
    return Boolean(field && field.value !== null && field.citations[0]);
  }

  if (target.id === "summary:overview") {
    return Boolean(result.overview && !result.overview.needs_human_review);
  }

  if (target.id.startsWith("summary-point:")) {
    const pointId = target.id.replace("summary-point:", "");
    return result.summary_sections.some((section) =>
      section.points.some((point) => point.point_id === pointId && !point.needs_human_review),
    );
  }

  if (target.id.startsWith("summary:")) {
    const index = Number(target.id.replace("summary:", "")) - 1;
    const item = result.summary_items[index];
    return Boolean(item && !item.needs_human_review);
  }

  return true;
}

function refreshCitationTarget(
  target: CitationTarget,
  result: WorkflowResult,
  showVerifiedOnly: boolean,
): CitationTarget | null {
  if (showVerifiedOnly && !isCitationTargetVisible(target, result)) {
    return buildInitialCitationTarget(result, true);
  }

  if (target.kind === "metric") {
    const metricKey = target.id.replace("metric:", "") as
      | "revenue"
      | "net_profit"
      | "unit"
      | "fiscal_period";
    const field = result.headline_metrics[metricKey];
    if (!field || field.value === null) {
      return buildInitialCitationTarget(result, showVerifiedOnly);
    }
    const citation = field.citations[0];
    return {
      kind: "metric",
      id: target.id,
      title: target.title,
      page: citation?.page_number ?? field.evidence_page ?? null,
      quote: citation?.quote || field.evidence_quote || "暂无引用片段。",
      value: field.value,
    };
  }

  if (target.id === "summary:overview") {
    if (!result.overview) {
      return buildInitialCitationTarget(result, showVerifiedOnly);
    }
    const citation = result.overview.citations[0];
    return {
      kind: "summary",
      id: target.id,
      title: "Overview",
      page: citation?.page_number ?? null,
      quote: citation?.quote || "暂无引用片段。",
    };
  }

  if (target.id.startsWith("summary-point:")) {
    const pointId = target.id.replace("summary-point:", "");
    for (const section of result.summary_sections) {
      const point = section.points.find((item) => item.point_id === pointId);
      if (!point) {
        continue;
      }
      const citation = point.citations[0];
      return {
        kind: "summary",
        id: target.id,
        title: section.title,
        page: citation?.page_number ?? null,
        quote: citation?.quote || "暂无引用片段。",
      };
    }
    return buildInitialCitationTarget(result, showVerifiedOnly);
  }

  if (target.id.startsWith("summary:")) {
    const index = Number(target.id.replace("summary:", "")) - 1;
    const item = result.summary_items[index];
    if (!item) {
      return buildInitialCitationTarget(result, showVerifiedOnly);
    }
    const citation = item.citations[0];
    return {
      kind: "summary",
      id: target.id,
      title: item.title,
      page: citation?.page_number ?? null,
      quote: citation?.quote || "暂无引用片段。",
    };
  }

  return target;
}

const LEFT_COLUMN_MIN_WIDTH = 280;
const LEFT_COLUMN_MAX_WIDTH = 520;
const RIGHT_COLUMN_MIN_WIDTH = 240;
const RIGHT_COLUMN_MAX_WIDTH = 420;
const CENTER_COLUMN_MIN_WIDTH = 520;
const RESIZER_LAYOUT_OVERHEAD = 72;

function clamp(value: number, lower: number, upper: number) {
  return Math.min(Math.max(value, lower), upper);
}
