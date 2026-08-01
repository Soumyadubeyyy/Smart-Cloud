import { useCallback, useEffect, useRef, useState } from "react";
import {
  FileText,
  Search,
  UploadCloud,
  Download,
  Share2,
  Trash2,
  X,
  Loader2,
} from "lucide-react";
import {
  listFiles,
  uploadFile,
  searchFiles,
  getDownloadUrl,
  deleteFile,
  type FileItem,
} from "@/lib/api";
import { Card, CategoryTag } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { ShareDialog } from "@/components/ShareDialog";
import { formatBytes, timeAgo } from "@/lib/utils";

export default function Files() {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [shareTarget, setShareTarget] = useState<FileItem | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setFiles(await listFiles());
    } catch {
      setError("Couldn't load your files. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) {
      refresh();
      return;
    }
    setSearching(true);
    try {
      setFiles(await searchFiles(query));
    } catch {
      setError("Search failed. Try again.");
    } finally {
      setSearching(false);
    }
  }

  async function handleUpload(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      for (const file of Array.from(fileList)) {
        await uploadFile(file);
      }
      await refresh();
    } catch (err) {
      const message =
        (err as { response?: { data?: { detail?: unknown } } })?.response?.data
          ?.detail ?? "Upload failed.";
      setError(typeof message === "string" ? message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function handleDownload(file: FileItem) {
    try {
      const url = await getDownloadUrl(file.id);
      window.open(url, "_blank");
    } catch {
      setError("Couldn't generate a download link.");
    }
  }

  async function handleDelete(file: FileItem) {
    if (!confirm(`Delete "${file.original_filename}"? This can't be undone.`)) return;
    try {
      await deleteFile(file.id);
      setFiles((prev) => prev.filter((f) => f.id !== file.id));
    } catch {
      setError("Couldn't delete the file.");
    }
  }

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="font-display text-3xl text-ink mb-1">Files</h1>
          <p className="text-sm text-ink-soft">
            Upload, search by meaning, and manage your documents.
          </p>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => handleUpload(e.target.files)}
        />
        <Button onClick={() => fileInputRef.current?.click()} disabled={uploading}>
          {uploading ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <UploadCloud size={16} />
          )}
          {uploading ? "Uploading…" : "Upload"}
        </Button>
      </div>

      <form onSubmit={handleSearch} className="mb-6 flex gap-2">
        <div className="relative flex-1">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-soft"
          />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by meaning — e.g. “how much did I spend on groceries”"
            className="pl-9"
          />
        </div>
        {query && (
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              setQuery("");
              refresh();
            }}
          >
            <X size={16} />
          </Button>
        )}
        <Button type="submit" variant="outline" disabled={searching}>
          {searching ? "Searching…" : "Search"}
        </Button>
      </form>

      {error && (
        <p className="mb-4 text-sm text-clay bg-clay-soft rounded-md px-3 py-2">
          {error}
        </p>
      )}

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleUpload(e.dataTransfer.files);
        }}
        className={`rounded-lg border-2 border-dashed transition-colors ${
          dragOver ? "border-teal bg-teal-soft" : "border-line"
        }`}
      >
        {loading ? (
          <p className="p-8 text-center text-sm text-ink-soft">Loading files…</p>
        ) : files.length === 0 ? (
          <div className="p-12 text-center">
            <UploadCloud size={28} className="mx-auto mb-3 text-ink-soft" />
            <p className="text-sm text-ink-soft">
              Drop files here, or click Upload above.
            </p>
          </div>
        ) : (
          <Card className="divide-y divide-line border-0">
            {files.map((file) => (
              <div key={file.id} className="flex items-center gap-3 p-4 group">
                <FileText size={18} className="text-ink-soft shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className="text-sm text-ink truncate">
                    {file.original_filename}
                  </div>
                  {file.summary && (
                    <div className="text-xs text-ink-soft truncate max-w-md">
                      {file.summary}
                    </div>
                  )}
                  <div className="text-xs text-ink-soft font-mono mt-0.5">
                    {formatBytes(file.file_size_bytes)} · {timeAgo(file.upload_date)}
                  </div>
                </div>
                <CategoryTag category={file.category} />
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => handleDownload(file)}
                    title="Download"
                    className="p-2 rounded-md text-ink-soft hover:bg-paper-dim hover:text-ink"
                  >
                    <Download size={15} />
                  </button>
                  <button
                    onClick={() => setShareTarget(file)}
                    title="Share"
                    className="p-2 rounded-md text-ink-soft hover:bg-paper-dim hover:text-ink"
                  >
                    <Share2 size={15} />
                  </button>
                  <button
                    onClick={() => handleDelete(file)}
                    title="Delete"
                    className="p-2 rounded-md text-ink-soft hover:bg-clay-soft hover:text-clay"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            ))}
          </Card>
        )}
      </div>

      {shareTarget && (
        <ShareDialog
          fileId={shareTarget.id}
          fileName={shareTarget.original_filename}
          onClose={() => setShareTarget(null)}
        />
      )}
    </div>
  );
}
