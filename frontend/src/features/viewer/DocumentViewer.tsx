import { apiBaseUrl } from "../../lib/api";
import type { CitationTarget, DemoDocument } from "../../lib/types";
import { PdfDocumentViewer } from "./PdfDocumentViewer";

type DocumentViewerProps = {
  document: DemoDocument | null;
  citationTarget: CitationTarget | null;
};

export function DocumentViewer({ document, citationTarget }: DocumentViewerProps) {
  if (!document) {
    return (
      <div className="viewer-empty">
        <p>中间区域会显示原始文档。</p>
        <p>先选择一份文档并运行分析，随后点击左侧摘要或数字查看对应证据。</p>
      </div>
    );
  }

  const sourceUrl = `${apiBaseUrl()}${document.source_url}`;

  if (document.source_kind === "pdf") {
    return <PdfDocumentViewer document={document} citationTarget={citationTarget} />;
  }

  return (
    <div className="viewer-frame-shell">
      <iframe
        key={sourceUrl}
        className="viewer-frame"
        src={sourceUrl}
        title={document.label}
      />
    </div>
  );
}
