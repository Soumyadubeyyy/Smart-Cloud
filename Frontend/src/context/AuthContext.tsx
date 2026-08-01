import { createContext, useContext, useState, type ReactNode } from "react";
import { getToken, setToken as persistToken } from "@/lib/api";

interface AuthContextValue {
  isAuthenticated: boolean;
  setAuthenticated: (token: string) => void;
  clearAuth: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(!!getToken());

  function setAuthenticated(token: string) {
    persistToken(token);
    setIsAuthenticated(true);
  }

  function clearAuth() {
    persistToken(null);
    setIsAuthenticated(false);
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, setAuthenticated, clearAuth }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
