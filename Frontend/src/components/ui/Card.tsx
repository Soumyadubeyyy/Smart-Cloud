import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-lg border border-line bg-white/60 backdrop-blur-sm",
        className
      )}
      {...props}
    />
  );
}

const CATEGORY_COLORS: Record<string, string> = {
  Work: "bg-teal-soft text-teal",
  Finance: "bg-clay-soft text-clay",
  Legal: "bg-teal-soft text-teal",
  Career: "bg-clay-soft text-clay",
};

export function CategoryTag({ category }: { category: string | null }) {
  const label = category ?? "Uncategorized";
  const style = CATEGORY_COLORS[label] ?? "bg-paper-dim text-ink-soft";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        style
      )}
    >
      {label}
    </span>
  );
}
