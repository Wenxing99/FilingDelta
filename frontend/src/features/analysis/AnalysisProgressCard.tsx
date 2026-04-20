import { ProgressStages } from "../../components/ProgressStages";
import type { DemoRun } from "../../lib/types";

type AnalysisProgressCardProps = {
  run: DemoRun;
};

export function AnalysisProgressCard({ run }: AnalysisProgressCardProps) {
  const isRunning = run.status === "queued" || run.status === "running";
  const title =
    run.status === "failed" ? "分析失败" : isRunning ? "分析进行中" : "分析指标";
  const stageTimings = run.telemetry.stage_timings;
  const artifacts = run.telemetry.artifacts;
  const hasTelemetry = Boolean(
    stageTimings.total_ms ||
      artifacts.total_pages ||
      artifacts.chunk_count ||
      artifacts.summary_sections_count ||
      artifacts.summary_points_count ||
      artifacts.verification_issues_count !== null,
  );

  return (
    <section className="panel-card panel-card--compact">
      <div className="panel-card__header panel-card__header--split">
        <div>
          <p className="panel-card__kicker">Analysis</p>
          <h3>{title}</h3>
        </div>
        <span className="status-chip">{run.stage_label}</span>
      </div>
      <ProgressStages activeIndex={run.stage_index} message={run.progress_message} compact />
      {hasTelemetry ? (
        <div className="analysis-telemetry">
          <div className="analysis-telemetry__grid">
            <div className="analysis-telemetry__group">
              <h5>耗时</h5>
              <ul>
                <li>总耗时：{formatDuration(stageTimings.total_ms)}</li>
                <li>解析文档：{formatDuration(stageTimings.orchestrate_ms)}</li>
                <li>提取重点：{formatDuration(stageTimings.reader_ms)}</li>
                <li>抽取关键数据：{formatDuration(stageTimings.fact_extractor_ms)}</li>
                <li>核验引用：{formatDuration(stageTimings.verifier_ms)}</li>
              </ul>
            </div>
            <div className="analysis-telemetry__group">
              <h5>产物</h5>
              <ul>
                <li>页数：{formatCount(artifacts.total_pages)}</li>
                <li>Chunks：{formatCount(artifacts.chunk_count)}</li>
                <li>Sections：{formatCount(artifacts.summary_sections_count)}</li>
                <li>Points：{formatCount(artifacts.summary_points_count)}</li>
                <li>待确认：{formatCount(artifacts.verification_issues_count)}</li>
              </ul>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

function formatDuration(value: number | null): string {
  if (value === null || value <= 0) {
    return "—";
  }

  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }

  return `${(value / 1000).toFixed(value >= 10_000 ? 0 : 1)} s`;
}

function formatCount(value: number | null): string {
  return value === null ? "—" : String(value);
}
