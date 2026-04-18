import type { CitationTarget } from "../../lib/types";

type CitationDetailBarProps = {
  target: CitationTarget | null;
};

export function CitationDetailBar({ target }: CitationDetailBarProps) {
  if (!target) {
    return (
      <div className="citation-bar citation-bar--empty">
        <span>选择左侧摘要或数字，这里会显示当前项的页码和原文证据。</span>
      </div>
    );
  }

  return (
    <div className="citation-bar">
      <strong className="citation-bar__title">{target.title}</strong>
      <span className="citation-bar__page">{target.page ? `第 ${target.page} 页` : "页码待补充"}</span>
      <p className="citation-bar__quote">{target.quote || "暂无引用片段。"}</p>
    </div>
  );
}
