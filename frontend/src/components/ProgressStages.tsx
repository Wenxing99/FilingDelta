const STAGES = ["解析文档", "提取重点", "抽取关键数据", "核验引用"];

type ProgressStagesProps = {
  activeIndex: number;
  message: string;
  compact?: boolean;
};

export function ProgressStages({ activeIndex, message, compact = false }: ProgressStagesProps) {
  const shellClassName = compact ? "progress-shell progress-shell--compact" : "progress-shell";
  const rowClassName = compact ? "progress-row progress-row--compact" : "progress-row";

  return (
    <div className={shellClassName}>
      <div className={rowClassName} aria-label="分析进度">
        {STAGES.map((stage, index) => {
          const stepIndex = index + 1;
          const className =
            activeIndex >= stepIndex ? "progress-step progress-step--active" : "progress-step";

          return (
            <div key={stage} className={className}>
              <span className="progress-step__index">{stepIndex}</span>
              <span className="progress-step__label">{stage}</span>
            </div>
          );
        })}
      </div>
      <p className="progress-message">{message}</p>
    </div>
  );
}
