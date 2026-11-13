import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatMessage } from "@/hooks/useChatThread";
import { Citation } from "@/types/api";
import {
  BookOpen,
  ChevronDown,
  ChevronUp,
  FileText,
  Sparkles,
} from "lucide-react";

function SourceCard({ cite }: { cite: Citation }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-xl border border-white/5 bg-slate-950/70 p-3 text-xs text-slate-300">
      <div className="mb-2 flex items-center justify-between gap-3 border-b border-white/5 pb-1.5">
        <div className="flex min-w-0 items-center gap-2 font-medium text-indigo-300">
          <span className="flex h-5 min-w-5 shrink-0 items-center justify-center rounded border border-indigo-500/30 bg-indigo-500/20 px-1 font-mono text-[10px]">
            {cite.index}
          </span>
          <FileText className="h-3.5 w-3.5 shrink-0 text-slate-400" />
          <span className="truncate">{cite.filename}</span>
          <span className="shrink-0 font-mono text-[10px] text-slate-500">
            chunk #{cite.chunk_index}
          </span>
        </div>
        <span className="shrink-0 font-mono text-[10px] text-emerald-400">
          {(cite.score * 100).toFixed(1)}%
        </span>
      </div>
      <p
        className={`whitespace-pre-wrap text-[11px] leading-relaxed text-slate-400 ${
          expanded ? "" : "line-clamp-3"
        }`}
      >
        {cite.content}
      </p>
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="mt-1.5 cursor-pointer text-[10px] font-medium text-indigo-300 hover:text-indigo-200"
      >
        {expanded ? "Show less" : "Show more"}
      </button>
    </div>
  );
}

export function MessageBubble({
  message,
  isLoading = false,
}: {
  message?: ChatMessage;
  isLoading?: boolean;
}) {
  const [showSources, setShowSources] = useState(false);

  if (isLoading) {
    return (
      <div className="flex flex-col items-start w-full animate-pulse mt-2">
        <div className="flex items-center gap-3 rounded-2xl border border-indigo-500/20 bg-indigo-500/5 px-5 py-3 text-sm text-indigo-300 shadow-sm">
          <Sparkles className="h-4 w-4 animate-spin" />
          <span className="tracking-wide">AI is thinking...</span>
        </div>
      </div>
    );
  }

  if (!message) return null;

  const isUser = message.role === "user";
  const citations = message.citations ?? [];

  return (
    <div className={`flex flex-col w-full ${isUser ? "items-end" : "items-start"} mt-4`}>
      <div
        className={`max-w-[85%] sm:max-w-[75%] rounded-3xl px-6 py-4 text-sm leading-relaxed shadow-sm ${
          isUser
            ? "bg-gradient-to-br from-indigo-500 to-indigo-600 text-white rounded-tr-sm"
            : "border border-white/10 bg-slate-900/60 backdrop-blur-md text-slate-100 rounded-tl-sm"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-pre:bg-slate-950 prose-pre:border prose-pre:border-slate-800">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {!isUser &&
          (message.rewrittenQuery ||
            message.topScore !== undefined ||
            citations.length > 0) && (
            <div className="mt-4 border-t border-white/5 pt-3 text-[11px] text-slate-400">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-3">
                  {message.rewrittenQuery && (
                    <span className="flex items-center gap-1.5">
                      <span className="opacity-60">Rewritten:</span>
                      <em className="text-slate-300 not-italic bg-white/5 px-2 py-0.5 rounded-md">
                        {message.rewrittenQuery}
                      </em>
                    </span>
                  )}
                  {typeof message.topScore === "number" && (
                    <span className="rounded-md bg-emerald-500/10 px-2 py-0.5 font-mono font-medium text-emerald-400 border border-emerald-500/20">
                      Score: {(message.topScore * 100).toFixed(1)}%
                    </span>
                  )}
                </div>

                {citations.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setShowSources((prev) => !prev)}
                    aria-expanded={showSources}
                    className="flex cursor-pointer items-center gap-1.5 rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-2.5 py-1 text-xs font-medium text-indigo-300 transition hover:bg-indigo-500/20"
                  >
                    <BookOpen className="h-3.5 w-3.5" />
                    <span>
                      {citations.length}{" "}
                      {citations.length === 1 ? "Source" : "Sources"}
                    </span>
                    {showSources ? (
                      <ChevronUp className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5" />
                    )}
                  </button>
                )}
              </div>

              {showSources && citations.length > 0 && (
                <div className="mt-3 space-y-2 border-t border-white/5 pt-3">
                  {citations.map((cite) => (
                    <SourceCard key={cite.index} cite={cite} />
                  ))}
                </div>
              )}
            </div>
          )}
      </div>
    </div>
  );
}
