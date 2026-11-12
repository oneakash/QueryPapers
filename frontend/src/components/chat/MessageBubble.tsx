import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChatMessage } from "@/hooks/useChatThread";
import { Sparkles } from "lucide-react";

export function MessageBubble({
  message,
  isLoading = false,
}: {
  message?: ChatMessage;
  isLoading?: boolean;
}) {
  if (isLoading) {
    return (
      <div className="flex flex-col items-start w-full animate-pulse mt-2">
        <div className="flex items-center gap-3 rounded-2xl border border-indigo-500/20 bg-indigo-500/5 px-5 py-3 text-sm text-indigo-300 shadow-sm">
          <Sparkles className="h-4 w-4 animate-spin-slow" />
          <span className="tracking-wide">AI is thinking...</span>
        </div>
      </div>
    );
  }

  if (!message) return null;

  const isUser = message.role === "user";

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

        {!isUser && (message.rewrittenQuery || message.topScore !== undefined) && (
          <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-white/5 pt-3 text-[11px] text-slate-400">
            {message.rewrittenQuery && (
              <span className="flex items-center gap-1.5">
                <span className="opacity-60">Rewritten:</span> 
                <em className="text-slate-300 not-italic bg-white/5 px-2 py-0.5 rounded-md">{message.rewrittenQuery}</em>
              </span>
            )}
            {typeof message.topScore === "number" && (
              <span className="rounded-md bg-emerald-500/10 px-2 py-0.5 font-mono font-medium text-emerald-400 border border-emerald-500/20">
                Score: {(message.topScore * 100).toFixed(1)}%
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
