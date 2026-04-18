export type DemoDocument = {
  document_id: string;
  label: string;
  company_name: string;
  ticker: string | null;
  market: string;
  doc_type: string;
  fiscal_period: string | null;
  language: string;
  source_kind: "pdf" | "html" | "other";
  source_url: string;
};

export type Citation = {
  document_id: string;
  source_path: string;
  page_number: number | null;
  quote: string;
};

export type SummaryItem = {
  title: string;
  summary: string;
  citations: Citation[];
  needs_human_review: boolean;
};

export type ExtractedFactField = {
  value: string | number | null;
  reasoning: string | null;
  confidence: number | null;
  evidence_page: number | null;
  evidence_quote: string | null;
  citations: Citation[];
};

export type HeadlineMetrics = {
  document_id: string;
  source_path: string;
  company_name: ExtractedFactField;
  fiscal_period: ExtractedFactField;
  unit: ExtractedFactField;
  revenue: ExtractedFactField;
  net_profit: ExtractedFactField;
};

export type VerificationIssue = {
  scope: "summary" | "facts";
  item_key: string;
  message: string;
  severity: "warning" | "review";
  evidence_page: number | null;
  evidence_quote: string | null;
};

export type WorkflowResult = {
  document_id: string;
  source_path: string;
  parser_kind: string;
  total_pages: number;
  chunk_count: number;
  summary_items: SummaryItem[];
  headline_metrics: HeadlineMetrics;
  verification_issues: VerificationIssue[];
  needs_human_review: boolean;
};

export type DemoRun = {
  run_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  stage: "queued" | "orchestrate" | "reader" | "fact_extractor" | "verifier" | "done" | "failed";
  stage_label: string;
  stage_index: number;
  stage_count: number;
  progress_message: string;
  document_id: string;
  created_at: string;
  updated_at: string;
  error_message: string | null;
  result: WorkflowResult | null;
};

export type CitationTarget =
  | {
      kind: "summary";
      id: string;
      title: string;
      page: number | null;
      quote: string;
    }
  | {
      kind: "metric";
      id: string;
      title: string;
      page: number | null;
      quote: string;
      value: string | number | null;
    };
