import type { ChatResponse, DemoDocument, DemoRun } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const fallback = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      throw new Error(payload.detail || fallback);
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(fallback);
    }
  }

  return (await response.json()) as T;
}

export function apiBaseUrl(): string {
  return API_BASE_URL;
}

export async function listDemoDocuments(): Promise<DemoDocument[]> {
  const payload = await requestJson<{ documents: DemoDocument[] }>("/api/demo/documents");
  return payload.documents;
}

export async function createDemoRun(documentId: string): Promise<DemoRun> {
  const payload = await requestJson<{ run: DemoRun }>("/api/demo/runs", {
    method: "POST",
    body: JSON.stringify({ document_id: documentId }),
  });
  return payload.run;
}

export async function getDemoRun(runId: string): Promise<DemoRun> {
  const payload = await requestJson<{ run: DemoRun }>(`/api/demo/runs/${runId}`);
  return payload.run;
}

export async function approveDemoRunIssue(runId: string, itemKey: string): Promise<DemoRun> {
  const payload = await requestJson<{ run: DemoRun }>(`/api/demo/runs/${runId}/issues/approve`, {
    method: "POST",
    body: JSON.stringify({ item_key: itemKey }),
  });
  return payload.run;
}

export async function rerunDemoRunIssue(runId: string, itemKey: string): Promise<DemoRun> {
  const payload = await requestJson<{ run: DemoRun }>(`/api/demo/runs/${runId}/issues/rerun`, {
    method: "POST",
    body: JSON.stringify({ item_key: itemKey }),
  });
  return payload.run;
}

export async function rerunDemoRunFeedback(
  runId: string,
  feedbackCategory: "citation" | "numeric" | "summary",
): Promise<DemoRun> {
  const payload = await requestJson<{ run: DemoRun }>(`/api/demo/runs/${runId}/feedback`, {
    method: "POST",
    body: JSON.stringify({ feedback_category: feedbackCategory }),
  });
  return payload.run;
}

export async function askDemoChat(documentId: string, sessionId: string, question: string): Promise<ChatResponse> {
  const payload = await requestJson<{ response: ChatResponse }>("/api/demo/chat", {
    method: "POST",
    body: JSON.stringify({
      document_id: documentId,
      session_id: sessionId,
      question,
    }),
  });
  return payload.response;
}
