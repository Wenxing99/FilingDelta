import type { ChatResponse, ChatStreamEvent, ChatStreamHandlers, DemoDocument, DemoRun } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const usesFormData = typeof FormData !== "undefined" && init?.body instanceof FormData;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(usesFormData ? {} : { "Content-Type": "application/json" }),
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

export async function importDemoDocument(file: File): Promise<DemoDocument> {
  const formData = new FormData();
  formData.append("file", file);
  return requestJson<DemoDocument>("/api/demo/documents/import", {
    method: "POST",
    body: formData,
  });
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

export async function askDemoChatStream(
  documentId: string,
  sessionId: string,
  question: string,
  handlers: ChatStreamHandlers,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/demo/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      document_id: documentId,
      session_id: sessionId,
      question,
    }),
  });

  if (!response.ok) {
    throw await buildRequestError(response);
  }
  if (!response.body) {
    return askDemoChat(documentId, sessionId, question);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: ChatResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const rawLine = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (rawLine) {
        const event = parseChatStreamEvent(rawLine);
        if (event.type === "error") {
          throw new Error(event.message || "问答请求失败。");
        }
        if (event.type === "status") {
          handlers.onStatus?.(event);
        } else if (event.type === "delta") {
          handlers.onDelta?.(event);
        } else if (event.type === "citations") {
          handlers.onCitations?.(event);
        } else if (event.type === "telemetry") {
          handlers.onTelemetry?.(event);
        } else if (event.type === "done") {
          finalResponse = event.response;
          handlers.onDone?.(event);
        }
      }
      newlineIndex = buffer.indexOf("\n");
    }

    if (done) {
      break;
    }
  }

  const remaining = buffer.trim();
  if (remaining) {
    const event = parseChatStreamEvent(remaining);
    if (event.type === "done") {
      finalResponse = event.response;
      handlers.onDone?.(event);
    } else if (event.type === "error") {
      throw new Error(event.message || "问答请求失败。");
    }
  }

  if (!finalResponse) {
    throw new Error("问答请求未返回完整结果。");
  }
  return finalResponse;
}

async function buildRequestError(response: Response): Promise<Error> {
  const fallback = `${response.status} ${response.statusText}`;
  try {
    const payload = (await response.json()) as { detail?: string };
    return new Error(payload.detail || fallback);
  } catch {
    return new Error(fallback);
  }
}

function parseChatStreamEvent(line: string): ChatStreamEvent {
  return JSON.parse(line) as ChatStreamEvent;
}
