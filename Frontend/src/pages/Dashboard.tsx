import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { FileText } from "lucide-react";
import { getDashboard, type DashboardData } from "@/lib/api";
import { Card, CategoryTag } from "@/components/ui/Card";
import { formatBytes, timeAgo } from "@/lib/utils";

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDashboard()
      .then((res) => {
        if ("total_files" in res) setData(res);
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="font-display text-3xl text-ink mb-1">Dashboard</h1>
      <p className="text-sm text-ink-soft mb-8">
        A quick look at what's in your library.
      </p>

      {loading && <p className="text-sm text-ink-soft">Loading…</p>}

      {!loading && data && (
        <>
          <div className="grid grid-cols-3 gap-4 mb-8">
            <Card className="p-5">
              <div className="text-xs font-medium text-ink-soft uppercase tracking-wide mb-1">
                Total files
              </div>
              <div className="font-display text-3xl text-ink">
                {data.total_files}
              </div>
            </Card>
            <Card className="p-5">
              <div className="text-xs font-medium text-ink-soft uppercase tracking-wide mb-1">
                Storage used
              </div>
              <div className="font-display text-3xl text-ink">
                {data.total_storage_mb} MB
              </div>
            </Card>
            <Card className="p-5">
              <div className="text-xs font-medium text-ink-soft uppercase tracking-wide mb-1">
                Categories
              </div>
              <div className="font-display text-3xl text-ink">
                {data.category_counts.length}
              </div>
            </Card>
          </div>

          {data.category_counts.length > 0 && (
            <div className="mb-8">
              <h2 className="text-sm font-medium text-ink-soft mb-3">
                By category
              </h2>
              <div className="flex flex-wrap gap-2">
                {data.category_counts.map((c) => (
                  <div
                    key={c.category}
                    className="flex items-center gap-2 rounded-full border border-line px-3 py-1.5"
                  >
                    <CategoryTag category={c.category} />
                    <span className="text-sm font-mono text-ink-soft">
                      {c.count}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-medium text-ink-soft">
                Recently uploaded
              </h2>
              <Link to="/files" className="text-sm text-teal font-medium">
                View all
              </Link>
            </div>

            {data.recent_files.length === 0 ? (
              <Card className="p-8 text-center">
                <p className="text-sm text-ink-soft">
                  Nothing uploaded yet. Head to Files to add your first document.
                </p>
              </Card>
            ) : (
              <Card className="divide-y divide-line">
                {data.recent_files.map((f) => (
                  <div key={f.id} className="flex items-center gap-3 p-4">
                    <FileText size={18} className="text-ink-soft shrink-0" />
                    <div className="min-w-0 flex-1">
                      <div className="text-sm text-ink truncate">
                        {f.original_filename}
                      </div>
                      <div className="text-xs text-ink-soft font-mono">
                        {formatBytes(f.file_size_bytes)} · {timeAgo(f.upload_date)}
                      </div>
                    </div>
                    <CategoryTag category={f.category} />
                  </div>
                ))}
              </Card>
            )}
          </div>
        </>
      )}
    </div>
  );
}
