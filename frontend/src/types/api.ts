export type ThreadStatus = "completed" | "awaiting_approval";

export interface InterruptPreview {
  question: string;
  top_score: number;
  chunks_preview: string[];
}

export interface ChatRequest {
  thread_id: string;
  query: string;
  stream?: boolean;
}

export interface Citation {
  index: number;
  doc_id: string;
  filename: string;
  chunk_index: number;
  content: string;
  score: number;
}

export interface ChatResponse {
  thread_id: string;
  status: ThreadStatus;
  answer: string | null;
  rewritten_query: string | null;
  top_retrieval_score: number;
  citations?: Citation[];
  interrupt_payload: InterruptPreview | null;
}

export interface ResumeRequest {
  thread_id: string;
  approved: boolean;
  feedback?: string | null;
}

export interface UploadResponse {
  doc_id: string;
  filename: string;
  chunks_indexed: number;
}

export interface SSEStepEvent {
  node: string;
  update: Record<string, unknown>;
}