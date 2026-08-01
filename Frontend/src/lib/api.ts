import axios from "axios";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const api = axios.create({ baseURL: API_BASE_URL });

const TOKEN_KEY = "smart_cloud_token";

export function setToken(token: string | null) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ---- Types mirroring the backend's Pydantic response models ----

export interface FileItem {
  id: number;
  original_filename: string;
  category: string | null;
  summary: string | null;
  upload_date: string;
  file_size_bytes: number;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export interface DashboardData {
  total_files: number;
  total_storage_mb: number;
  recent_files: FileItem[];
  category_counts: CategoryCount[];
  data_version: string;
}

// ---- Auth ----

export async function signup(email: string, password: string) {
  const { data } = await api.post("/auth/signup", { email, password });
  return data;
}

export async function login(email: string, password: string) {
  const { data } = await api.post("/auth/login", { email, password });
  // Supabase's sign_in_with_password response shape
  const token: string | undefined = data?.session?.access_token;
  if (!token) throw new Error("Login succeeded but no session token was returned.");
  setToken(token);
  return data;
}

export async function logout() {
  try {
    await api.post("/auth/signout");
  } finally {
    setToken(null);
  }
}

// ---- Files ----

export async function listFiles(): Promise<FileItem[]> {
  const { data } = await api.get("/files");
  return data;
}

export async function uploadFile(file: File, force = false) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/upload-and-analyze/", form, {
    params: { force },
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getDownloadUrl(fileId: number): Promise<string> {
  const { data } = await api.get(`/files/${fileId}/download`);
  return data.download_url;
}

export async function deleteFile(fileId: number) {
  const { data } = await api.delete(`/files/${fileId}`);
  return data;
}

export async function searchFiles(query: string): Promise<FileItem[]> {
  const { data } = await api.get("/search", { params: { query } });
  return data;
}

export async function createShareLink(
  fileId: number,
  options: { password?: string; expires_in_days?: number }
): Promise<string> {
  const { data } = await api.post(`/files/${fileId}/share`, options);
  return data.share_url;
}

// ---- Dashboard ----

export async function getDashboard(
  currentVersion?: string
): Promise<DashboardData | { status: "unchanged" }> {
  const { data } = await api.get("/dashboard", {
    params: currentVersion ? { current_version: currentVersion } : {},
  });
  return data;
}
