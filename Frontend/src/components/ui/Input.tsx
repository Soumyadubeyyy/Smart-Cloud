import { type InputHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-10 w-full rounded-md border border-line bg-paper px-3 text-sm text-ink placeholder:text-ink-soft/60 outline-none focus:border-teal focus:ring-1 focus:ring-teal transition-colors",
        className
      )}
      {...props}
    />
  )
);
Input.displayName = "Input";
