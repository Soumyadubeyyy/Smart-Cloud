import { NavLink, useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import { LayoutDashboard, FolderOpen, LogOut, Cloud } from "lucide-react";
import { logout } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/files", label: "Files", icon: FolderOpen },
];

export function Shell({ children }: { children: ReactNode }) {
  const { clearAuth } = useAuth();
  const navigate = useNavigate();

  async function handleLogout() {
    await logout();
    clearAuth();
    navigate("/login");
  }

  return (
    <div className="min-h-screen flex">
      <aside className="w-56 shrink-0 border-r border-line px-4 py-6 flex flex-col">
        <div className="flex items-center gap-2 px-2 mb-8">
          <Cloud size={20} className="text-teal" strokeWidth={2.5} />
          <span className="font-display text-lg text-ink">Smart Cloud</span>
        </div>

        <nav className="flex flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-ink text-paper"
                    : "text-ink-soft hover:bg-paper-dim"
                )
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        <button
          onClick={handleLogout}
          className="mt-auto flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium text-ink-soft hover:bg-paper-dim transition-colors"
        >
          <LogOut size={16} />
          Sign out
        </button>
      </aside>

      <main className="flex-1 px-8 py-8 max-w-5xl">{children}</main>
    </div>
  );
}
