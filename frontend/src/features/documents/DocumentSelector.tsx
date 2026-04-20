import { useRef } from "react";

import type { DemoDocument } from "../../lib/types";

type DocumentSelectorProps = {
  documents: DemoDocument[];
  selectedDocumentId: string;
  disabled?: boolean;
  isImporting?: boolean;
  onChange: (documentId: string) => void;
  onImport: (file: File) => void;
};

export function DocumentSelector({
  documents,
  selectedDocumentId,
  disabled = false,
  isImporting = false,
  onChange,
  onImport,
}: DocumentSelectorProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div className="document-selector">
      <label className="field-stack">
        <span className="field-label">文档选择</span>
        <div className="document-selector__controls">
          <select
            className="field-control document-selector__select"
            value={selectedDocumentId}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
          >
            <option value="">请选择文档</option>
            {documents.map((document) => (
              <option key={document.document_id} value={document.document_id}>
                {document.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="secondary-button"
            disabled={disabled || isImporting}
            onClick={() => fileInputRef.current?.click()}
          >
            {isImporting ? "导入中..." : "导入文件"}
          </button>
        </div>
        <span className="field-hint">支持 PDF 和单文件 HTML 导入</span>
      </label>
      <input
        ref={fileInputRef}
        className="sr-only"
        type="file"
        accept=".pdf,.htm,.html"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (!file) {
            return;
          }
          onImport(file);
          event.currentTarget.value = "";
        }}
      />
    </div>
  );
}
