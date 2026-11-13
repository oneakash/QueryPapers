import React from "react";
import { CircleCheck, Info, TriangleAlert, X } from "lucide-react";

export type ToastKind = "success" | "info" | "error";

export interface ToastMessage {
  id: number;
  kind: ToastKind;
  title: string;
  description?: string;
}

const STYLES: Record<ToastKind, { box: string; icon: string }> = {
  success: {
    box: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    icon: "text-emerald-400",
  },
  info: {
    box: "border-indigo-500/30 bg-indigo-500/10 text-indigo-100",
    icon: "text-indigo-300",
  },
  error: {
    box: "border-rose-500/30 bg-rose-500/10 text-rose-100",
    icon: "text-rose-300",
  },
};

function ToastIcon({ kind }: { kind: ToastKind }) {
  const className = `mt-0.5 h-4 w-4 shrink-0 ${STYLES[kind].icon}`;
  if (kind === "success") return <CircleCheck className={className} />;
  if (kind === "error") return <TriangleAlert className={className} />;
  return <Info className={className} />;
}

export function Toast({
  toast,
  onDismiss,
}: {
  toast: ToastMessage | null;
  onDismiss: () => void;
}) {
  if (!toast) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed right-6 top-6 z-50 w-full max-w-sm"
    >
      <div
        key={toast.id}
        className={`pointer-events-auto flex items-start gap-3 rounded-xl border px-4 py-3 text-sm shadow-2xl backdrop-blur-xl ${STYLES[toast.kind].box}`}
      >
        <ToastIcon kind={toast.kind} />
        <div className="min-w-0 flex-1">
          <p className="font-semibold">{toast.title}</p>
          {toast.description && (
            <p className="mt-0.5 break-words text-xs opacity-80">
              {toast.description}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss notification"
          className="shrink-0 cursor-pointer rounded p-0.5 opacity-60 transition hover:opacity-100"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}