import {
  ChatRequest,
  ChatResponse,
  ResumeRequest,
  SSEStepEvent,
  UploadResponse,
} from "@/types/api";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function uploadDocument(
  userId: string,
  file: File
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    headers: {
      "X-User-ID": userId,
    },
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `Upload failed (${response.status})`);
  }
  return response.json();
}

export async function resumeThread(
  userId: string,
  payload: ResumeRequest
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/resume`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-User-ID": userId,
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `Resume failed (${response.status})`);
  }
  return response.json();
}

export async function streamChatMessage(
  userId: string,
  payload: ChatRequest,
  callbacks: {
    onStep?: (step: SSEStepEvent) => void;
    onComplete: (finalResponse: ChatResponse) => void;
    onError: (errorMessage: string) => void;
  }
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "X-User-ID": userId,
    },
    body: JSON.stringify({ ...payload, stream: true }),
  });

  if (!response.ok || !response.body) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `Chat failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const rawEvents = buffer.split("\n\n");
    buffer = rawEvents.pop() || "";

    for (const rawBlock of rawEvents) {
      if (!rawBlock.trim()) continue;
      let eventType = "message";
      let dataLines: string[] = [];

      for (const line of rawBlock.split("\n")) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      }

      const rawData = dataLines.join("\n");
      if (!rawData) continue;

      if (eventType === "step" && callbacks.onStep) {
        callbacks.onStep(JSON.parse(rawData) as SSEStepEvent);
      } else if (eventType === "complete") {
        callbacks.onComplete(JSON.parse(rawData) as ChatResponse);
      } else if (eventType === "error") {
        const parsedErr = JSON.parse(rawData) as { detail?: string };
        callbacks.onError(parsedErr.detail || "Streaming error occurred.");
      }
    }
  }
}