"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Upload,
  Send,
  ShieldCheck,
  RefreshCw,
  FileText,
  Activity,
  Lock,
  UserCheck,
  Database,
} from "lucide-react";
import { uploadDocument } from "@/api/client";
import { HitlApprovalCard } from "@/components/chat/HitlApprovalCard";
import { MessageBubble } from "@/components/chat/MessageBubble";
import { useChatThread } from "@/hooks/useChatThread";
import { UploadResponse } from "@/types/api";

export default function HomePage() {
  const {
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
  } = useChatThread("akash_dev");

  const [tenantInput, setTenantInput] = useState(userId);
  const [queryInput, setQueryInput] = useState("");
  const [uploadedDocs, setUploadedDocs] = useState<UploadResponse[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change or streaming starts
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming, interruptPayload]);

  const handleTenantSwitch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!tenantInput.trim()) return;
    startNewThread(tenantInput.trim());
    setUploadedDocs([]);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);
    try {
      const res = await uploadDocument(userId, file);
      setUploadedDocs((prev) => [res, ...prev]);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
      e.target.value = "";
    }
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryInput.trim()) return;
    const currentQuery = queryInput;
    setQueryInput("");
    await sendMessage(currentQuery);
  };

  const isInputLocked = isStreaming || status === "awaiting_approval";

  return (
    <div className="flex h-screen w-full bg-[#020617] text-slate-100 font-sans overflow-hidden selection:bg-indigo-500/30">
      {/* BACKGROUND GLOW EFFECTS */}
      <div className="pointer-events-none absolute -left-40 top-0 h-96 w-96 rounded-full bg-indigo-900/20 blur-[120px]" />
      <div className="pointer-events-none absolute -right-40 bottom-0 h-96 w-96 rounded-full bg-emerald-900/10 blur-[120px]" />

      {/* LEFT SIDEBAR: Frosted Glass */}
      <aside className="z-10 flex w-80 flex-col border-r border-white/5 bg-slate-900/30 backdrop-blur-xl p-6 shadow-2xl">
        <div className="flex items-center gap-3 border-b border-white/10 pb-6">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 shadow-lg shadow-emerald-500/20">
            <ShieldCheck className="h-5 w-5 text-slate-950" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-emerald-100 to-cyan-100 bg-clip-text text-transparent">
              HITL Console
            </h1>
            <p className="text-[11px] font-medium text-slate-400">
              LangGraph • Qdrant • Gemini
            </p>
          </div>
        </div>

        {/* Tenant Switcher */}
        <form onSubmit={handleTenantSwitch} className="mt-6 space-y-2.5">
          <label className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">
            <UserCheck className="h-3.5 w-3.5" />
            Active Tenant
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={tenantInput}
              onChange={(e) => setTenantInput(e.target.value)}
              className="w-full rounded-lg border border-white/10 bg-black/20 px-3 py-2 font-mono text-xs text-slate-200 transition focus:border-indigo-500/50 focus:bg-black/40 focus:outline-none focus:ring-1 focus:ring-indigo-500/50"
            />
            <button
              type="submit"
              className="rounded-lg bg-indigo-500/10 border border-indigo-500/20 px-3 py-2 text-xs font-semibold text-indigo-300 transition hover:bg-indigo-500/20 cursor-pointer"
            >
              Switch
            </button>
          </div>
          <p className="text-[10px] text-slate-500 flex justify-between">
            <span>Thread ID:</span>
            <span className="font-mono text-slate-400">{threadId}</span>
          </p>
        </form>

        {/* Document Upload */}
        <div className="mt-8 space-y-2.5">
          <label className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">
            <Database className="h-3.5 w-3.5" />
            Knowledge Base
          </label>
          <label className="group flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/10 bg-black/20 p-5 text-center transition-all hover:border-emerald-500/40 hover:bg-emerald-500/5 hover:shadow-lg hover:shadow-emerald-500/10">
            <div className="rounded-full bg-white/5 p-2 transition-transform group-hover:scale-110 group-hover:bg-emerald-500/20">
              <Upload className="h-5 w-5 text-emerald-400" />
            </div>
            <span className="mt-3 text-xs font-medium text-slate-300">
              {isUploading ? "Indexing into Qdrant..." : "Upload .PDF or .TXT"}
            </span>
            <span className="mt-1 text-[10px] text-slate-500">
              Scoped strictly to {userId}
            </span>
            <input
              type="file"
              accept=".pdf,.txt,application/pdf,text/plain"
              onChange={handleFileUpload}
              disabled={isUploading}
              className="hidden"
            />
          </label>
          {uploadError && (
            <p className="rounded-lg bg-rose-500/10 p-2.5 text-xs text-rose-300 border border-rose-500/20">
              {uploadError}
            </p>
          )}
        </div>

        {/* Indexed Documents List */}
        <div className="mt-4 flex-1 overflow-y-auto scrollbar-hide space-y-2">
          {uploadedDocs.map((doc) => (
            <div
              key={doc.doc_id}
              className="group flex items-center justify-between rounded-lg border border-white/5 bg-white/5 px-3 py-2.5 text-xs transition hover:bg-white/10"
            >
              <div className="flex items-center gap-2.5 truncate">
                <FileText className="h-4 w-4 shrink-0 text-slate-400 group-hover:text-emerald-400 transition-colors" />
                <span className="truncate font-medium text-slate-200">
                  {doc.filename}
                </span>
              </div>
              <span className="ml-2 shrink-0 rounded-md bg-white/10 px-2 py-0.5 font-mono text-[10px] text-slate-300">
                {doc.chunks_indexed}
              </span>
            </div>
          ))}
        </div>

        <button
          type="button"
          onClick={() => startNewThread()}
          className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-white/5 border border-white/10 py-2.5 text-xs font-semibold text-slate-200 transition hover:bg-white/10 cursor-pointer shadow-sm"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          New Conversation
        </button>
      </aside>

      {/* MAIN CHAT AREA */}
      <main className="relative flex flex-1 flex-col">
        {/* Top Status Bar */}
        <header className="absolute top-0 z-10 flex w-full items-center justify-between border-b border-white/5 bg-slate-950/50 backdrop-blur-md px-8 py-4">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-2 text-xs font-medium text-slate-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              Connected as <strong className="font-mono text-emerald-400">{userId}</strong>
            </span>
          </div>

          {activeNode && (
            <div className="flex items-center gap-2 rounded-full bg-indigo-500/10 border border-indigo-500/30 px-3 py-1.5 text-xs text-indigo-300 shadow-sm shadow-indigo-500/10">
              <Activity className="h-3.5 w-3.5 animate-pulse text-indigo-400" />
              <span>
                Node Executing: <strong className="font-mono font-medium">{activeNode}</strong>
              </span>
            </div>
          )}
        </header>

        {/* Messages Stream */}
        <div className="flex-1 overflow-y-auto scrollbar-hide px-8 pb-36 pt-24 space-y-6">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center text-center text-slate-500 mt-10">
              <div className="rounded-2xl bg-white/5 p-4 mb-4 border border-white/5 shadow-xl">
                <ShieldCheck className="h-10 w-10 text-slate-400" />
              </div>
              <h2 className="text-lg font-semibold text-slate-200">Ready for Interaction</h2>
              <p className="mt-2 max-w-sm text-sm text-slate-400 leading-relaxed">
                Upload a document and ask a question. Low-confidence retrievals will automatically trigger the HITL review protocol.
              </p>
            </div>
          )}

          {messages.map((m) => (
            <MessageBubble key={m.id} message={m} />
          ))}

          {/* Render thinking skeleton if waiting for graph output */}
          {isStreaming && activeNode && messages.length > 0 && messages[messages.length - 1].role === "user" && (
            <MessageBubble isLoading={true} />
          )}

          {/* HITL Interrupt Card */}
          {status === "awaiting_approval" && interruptPayload && (
            <HitlApprovalCard
              payload={interruptPayload}
              isSubmitting={isStreaming}
              onDecision={handleResume}
            />
          )}

          {error && (
            <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 p-4 text-sm text-rose-200 backdrop-blur-sm">
              <strong className="font-semibold block mb-1">Error Executing Graph:</strong>
              {error}
            </div>
          )}
          
          {/* Scroll Anchor */}
          <div ref={messagesEndRef} className="h-4" />
        </div>

        {/* Floating Chat Input */}
        <div className="absolute bottom-0 w-full bg-gradient-to-t from-[#020617] via-[#020617]/90 to-transparent pb-8 pt-10 px-8">
          <form
            onSubmit={handleSend}
            className="mx-auto max-w-4xl"
          >
            <div className="relative flex items-center rounded-2xl border border-white/10 bg-slate-900/60 backdrop-blur-xl shadow-2xl shadow-black/50 transition-all focus-within:border-indigo-500/50 focus-within:bg-slate-900/90 focus-within:ring-4 focus-within:ring-indigo-500/10 p-1.5">
              <input
                type="text"
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                disabled={isInputLocked}
                placeholder={
                  status === "awaiting_approval"
                    ? "Thread locked. Address the HITL review card above."
                    : "Message your knowledge graph..."
                }
                className="w-full bg-transparent px-4 py-3.5 text-sm text-slate-100 placeholder-slate-500 outline-none disabled:opacity-50"
              />
              
              {status === "awaiting_approval" ? (
                <div className="mr-3 p-2 bg-amber-500/10 rounded-xl">
                  <Lock className="h-5 w-5 text-amber-500" />
                </div>
              ) : (
                <button
                  type="submit"
                  disabled={isInputLocked || !queryInput.trim()}
                  className="mr-1 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500 text-white transition-all hover:bg-indigo-400 hover:shadow-lg hover:shadow-indigo-500/25 disabled:opacity-30 disabled:hover:shadow-none disabled:hover:bg-indigo-500 cursor-pointer"
                >
                  <Send className="h-4 w-4 ml-0.5" />
                </button>
              )}
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
