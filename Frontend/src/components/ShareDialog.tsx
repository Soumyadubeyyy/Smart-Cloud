import { useState } from "react";
import { X, Copy, Check } from "lucide-react";
import { createShareLink } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";

export function ShareDialog({
  fileId,
  fileName,
  onClose,
}: {
  fileId: number;
  fileName: string;
  onClose: () => void;
}) {
  const [password, setPassword] = useState("");
  const [expiresInDays, setExpiresInDays] = useState("7");
  const [link, setLink] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleCreate() {
    setLoading(true);
    setError(null);
    try {
      const url = await createShareLink(fileId, {
        password: password || undefined,
        expires_in_days: expiresInDays ? Number(expiresInDays) : undefined,
      });
      setLink(url);
    } catch {
      setError("Couldn't create the share link. Try again.");
    } finally {
      setLoading(false);
    }
  }

  function copyLink() {
    if (!link) return;
    navigator.clipboard.writeText(link);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/40 px-4">
      <div className="w-full max-w-sm rounded-lg bg-paper border border-line p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-display text-lg text-ink truncate pr-4">
            Share “{fileName}”
          </h3>
          <button onClick={onClose} className="text-ink-soft hover:text-ink">
            <X size={18} />
          </button>
        </div>

        {!link ? (
          <div className="space-y-3">
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-soft">
                Password (optional, min 8 chars)
              </label>
              <Input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Leave blank for no password"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-soft">
                Expires in (days)
              </label>
              <Input
                type="number"
                min={1}
                value={expiresInDays}
                onChange={(e) => setExpiresInDays(e.target.value)}
              />
            </div>
            {error && <p className="text-sm text-clay">{error}</p>}
            <Button className="w-full" onClick={handleCreate} disabled={loading}>
              {loading ? "Creating…" : "Create link"}
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Input readOnly value={link} className="font-mono text-xs" />
              <Button variant="outline" size="sm" onClick={copyLink}>
                {copied ? <Check size={14} /> : <Copy size={14} />}
              </Button>
            </div>
            <p className="text-xs text-ink-soft">
              Anyone with this link can access the file
              {password && " with the password you set"}.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
