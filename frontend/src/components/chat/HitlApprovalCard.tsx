"use client";

import React, { useState } from "react";
import { AlertTriangle, CheckCircle2, XCircle, Sparkles } from "lucide-react";
import { InterruptPreview } from "@/types/api";

interface HitlApprovalCardProps {
  payload: InterruptPreview;
  isSubmitting: boolean;
  onDecision: (approved: boolean, feedback?: string) => Promise<void>;
}

export function HitlApprovalCard({
  payload,
  isSubmitting,
  onDecision,
}: HitlApprovalCardProps) {
  const [feedback, setFeedback] = useState("");

  return (
    <div className="my-4 rounded-xl border-2 border-amber-500/40 bg-amber-950/20 p-5 text-amber-100 shadow-lg">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <AlertTriangle className="h-5 w-5 shrink-0 text-amber-400" />
          <div>
            <h3 className="text-sm font-semibold tracking-wide text-amber-300 uppercase">
              Human-in-the-Loop Review Required
            </h3>
            <p className="text-xs text-amber-200/80">
              Retrieval confidence fell below threshold. Inspect the chunks
              before allowing LLM generation.
            </p>
          </div>
        </div>
        <span className="rounded-full bg-amber-500/20 px-3 py-1 font-mono text-xs font-medium text-amber-300 border border-amber-500/30">
          Top Score: {(payload.top_score * 100).toFixed(1)}%
        </span>
      </div>

      <div className="mt-4 rounded-lg bg-slate-950/60 p-3 border border-slate-800">
        <p className="text-xs font-medium text-slate-400">Rewritten Query:</p>
        <p className="mt-0.5 text-sm font-medium text-slate-100">
          {payload.question}
        </p>
      </div>

      <div className="mt-3 space-y-2">
        <p className="text-xs font-medium text-slate-400">
          Retrieved Chunks Preview ({payload.chunks_preview.length}):
        </p>
        <div className="max-h-44 space-y-2 overflow-y-auto pr-1">
          {payload.chunks_preview.map((chunk, idx) => (
            <div
              key={idx}
              className="rounded border border-slate-800 bg-slate-900/80 p-2.5 font-mono text-xs text-slate-300 leading-relaxed"
            >
              <span className="mr-2 text-amber-400 font-bold">#{idx + 1}</span>
              {chunk}...
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4">
        <label className="flex items-center gap-1.5 text-xs font-medium text-amber-200/90">
          <Sparkles className="h-3.5 w-3.5 text-amber-400" />
          Optional LTM Rule / Reviewer Feedback (persisted across threads):
        </label>
        <input
          type="text"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          disabled={isSubmitting}
          placeholder="e.g., Always format the output as a Markdown table."
          className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-amber-400 focus:outline-none"
        />
      </div>

      <div className="mt-4 flex items-center justify-end gap-3">
        <button
          type="button"
          disabled={isSubmitting}
          onClick={() => onDecision(false, feedback)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-rose-500/40 bg-rose-950/40 px-4 py-2 text-xs font-semibold text-rose-200 hover:bg-rose-900/50 disabled:opacity-50 cursor-pointer"
        >
          <XCircle className="h-4 w-4 text-rose-400" />
          Reject Context
        </button>
        <button
          type="button"
          disabled={isSubmitting}
          onClick={() => onDecision(true, feedback)}
          className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50 cursor-pointer"
        >
          <CheckCircle2 className="h-4 w-4" />
          {isSubmitting ? "Resuming Graph..." : "Approve & Generate"}
        </button>
      </div>
    </div>
  );
}