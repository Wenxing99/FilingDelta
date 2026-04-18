import type { DemoDocument } from "../../lib/types";

type DocumentSelectorProps = {
  documents: DemoDocument[];
  selectedDocumentId: string;
  disabled?: boolean;
  onChange: (documentId: string) => void;
};

export function DocumentSelector({
  documents,
  selectedDocumentId,
  disabled = false,
  onChange,
}: DocumentSelectorProps) {
  return (
    <label className="field-stack">
      <span className="field-label">样例文档</span>
      <select
        className="field-control"
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
    </label>
  );
}
