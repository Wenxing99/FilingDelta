import { ProgressStages } from "../../components/ProgressStages";
import type { DemoRun } from "../../lib/types";

type AnalysisProgressCardProps = {
  run: DemoRun;
};

export function AnalysisProgressCard({ run }: AnalysisProgressCardProps) {
  return (
    <section className="panel-card panel-card--compact">
      <div className="panel-card__header panel-card__header--split">
        <div>
          <p className="panel-card__kicker">Analysis</p>
          <h3>分析进行中</h3>
        </div>
        <span className="status-chip">{run.stage_label}</span>
      </div>
      <ProgressStages
        activeIndex={run.stage_index}
        message={run.progress_message}
        compact
      />
    </section>
  );
}
