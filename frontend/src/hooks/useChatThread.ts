"use client";

import { useCallback, useState, useEffect } from "react";
import { resumeThread, streamChatMessage } from "@/api/client";
import {
  ChatResponse,
  Citation,
  InterruptPreview,
  ThreadStatus,
} from "@/types/api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  rewrittenQuery?: string | null;
  topScore?: number;
  citations?: Citation[];
}

export function useChatThread(initialUserId = "akash_dev") {
  const [userId, setUserId] = useState<string>(initialUserId);
  const [threadId, setThreadId] = useState<string>("initializing...");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ThreadStatus>("completed");
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [interruptPayload, setInterruptPayload] =
    useState<InterruptPreview | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Generate the initial thread ID strictly on the client to avoid SSR hydration mismatch
  useEffect(() => {
    setThreadId(`thread_${Math.random().toString(36).slice(2, 9)}`);
  }, []);

  const applyFinalResponse = useCallback((res: ChatResponse) => {
    setStatus(res.status);
    setInterruptPayload(res.interrupt_payload);

    if (res.status === "completed" && res.answer) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: res.answer as string,
          rewrittenQuery: res.rewritten_query,
          topScore: res.top_retrieval_score,
          citations: res.citations ?? [],
        },
      ]);
    }
  }, []);

  const sendMessage = useCallback(
    async (query: string) => {
      const trimmed = query.trim();
      if (!trimmed || isStreaming || status === "awaiting_approval") return;

      setError(null);
      setIsStreaming(true);
      setActiveNode("contextualize_query");

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "user",
          content: trimmed,
        },
      ]);

      try {
        await streamChatMessage(
          userId,
          { thread_id: threadId, query: trimmed, stream: true },
          {
            onStep: (step) => {
              setActiveNode(step.node);
            },
            onComplete: (finalRes) => {
              applyFinalResponse(finalRes);
            },
            onError: (errMsg) => {
              setError(errMsg);
            },
          }
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to send message");
      } finally {
        setIsStreaming(false);
        setActiveNode(null);
      }
    },
    [userId, threadId, isStreaming, status, applyFinalResponse]
  );

  const handleResume = useCallback(
    async (approved: boolean, feedback?: string) => {
      if (status !== "awaiting_approval" || isStreaming) return;

      setError(null);
      setIsStreaming(true);
      setActiveNode("human_review -> resume");

      try {
        const res = await resumeThread(userId, {
          thread_id: threadId,
          approved,
          feedback: feedback?.trim() || null,
        });
        applyFinalResponse(res);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to resume thread"
        );
      } finally {
        setIsStreaming(false);
        setActiveNode(null);
      }
    },
    [userId, threadId, status, isStreaming, applyFinalResponse]
  );

  const startNewThread = useCallback((overrideUserId?: string) => {
    if (overrideUserId !== undefined) {
      setUserId(overrideUserId);
    }
    setThreadId(`thread_${Math.random().toString(36).slice(2, 9)}`);
    setMessages([]);
    setStatus("completed");
    setInterruptPayload(null);
    setError(null);
    setActiveNode(null);
  }, []);

  return {
    userId,
    threadId,
    messages,
    status,
    activeNode,
    isStreaming,
    interruptPayload,
    error,
    sendMessage,
    handleResume,
    startNewThread,
  };
}
