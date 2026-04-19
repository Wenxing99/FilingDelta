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

export type SummaryPoint = {
  point_id: string;
  text: string;
  citations: Citation[];
  verification_status: "verified" | "review";
  needs_human_review: boolean;
};

export type SummarySection = {
  section_id: string;
  title: string;
  points: SummaryPoint[];
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
  item_label: string;
  message: string;
  severity: "warning" | "review";
  review_reason: "citation_pending" | "numeric_pending" | "summary_incomplete";
  user_visible_reason: string;
  evidence_page: number | null;
  evidence_quote: string | null;
};

export type ReviewStatusSummary = {
  status: "passed" | "needs_confirmation" | "failed";
  verified_count: number;
  pending_confirmation_count: number;
  failed_count: number;
};

export type WorkflowResult = {
  document_id: string;
  source_path: string;
  parser_kind: string;
  total_pages: number;
  chunk_count: number;
  overview: SummaryItem | null;
  summary_sections: SummarySection[];
  summary_items: SummaryItem[];
  headline_metrics: HeadlineMetrics;
  verification_issues: VerificationIssue[];
  needs_human_review: boolean;
  review: ReviewStatusSummary;
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
      kind: "chat";
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

export type ChatCitation = {
  citation_id: string;
  source_type: "document" | "external";
  page_number: number | null;
  quote: string;
  url: string | null;
  title: string | null;
  snippet: string | null;
};

export type ChatAnswerSection = {
  section_type: "document_evidence" | "external_evidence" | "analysis_and_limits";
  title: string;
  items: string[];
};

export type ChatResponse = {
  document_id: string;
  question: string;
  answer: string;
  route: "document_only" | "concept_only" | "mixed" | "unsupported";
  sections: ChatAnswerSection[];
  citations: ChatCitation[];
  retrieval_mode:
    | "semantic_with_filters"
    | "semantic_with_keyword_fallback"
    | "external_web_search"
    | "external_search_unavailable"
    | "mixed_document_external"
    | "unsupported";
};
