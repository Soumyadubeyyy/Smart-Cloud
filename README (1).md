# ☁️ Smart Cloud

**Smart Cloud** is an intelligent document management platform that solves the biggest problem with cloud storage: **organization**. Instead of forcing you to manually sort files into folders, Smart Cloud uses **Generative AI** to automatically categorize, tag, and summarize every document you upload — and replaces keyword search with **semantic search**, so you can find files based on what they *mean*, not just what they're named.

## 🚀 Live Demo

- **App:** [https://smart-cloud.vercel.app](https://smart-cloud.vercel.app)
- **Backend API docs:** [https://smart-cloud-backend.onrender.com/docs](https://smart-cloud-backend.onrender.com/docs)

---

## 🌟 Why Smart Cloud?

### 🏷️ Zero-Touch Auto-Categorization
As soon as you upload a file, **Gemini AI** reads its content and assigns it to the correct category (e.g. *Legal, Finance, Medical, Career*) with a generated summary — no manual folder-sorting required.

### 🧠 Context-Aware Smart Search
Smart Cloud uses **vector embeddings (Voyage AI)** and **pgvector** to understand search *intent*. Search "how much did I spend on groceries?" and it can surface `receipt_december.png` because it understands the document's content, not just its filename.

### 📊 Interactive Dashboard
Visual analytics of storage usage, file categories, and recent activity at a glance.

### 🔗 Secure Sharing
Generate time-limited, password-protected public links to share individual files.

---

## ✨ Key Features

- **🔐 Authentication** — Sign up, log in, and log out via Supabase Auth
- **📂 File Management** — Upload, list, download, and delete files with real-time storage tracking
- **🤖 AI Summarization** — Automatic document summaries via Google Gemini
- **🔍 Semantic Search** — Meaning-based file search via Voyage AI embeddings + pgvector
- **📊 Dashboard** — Storage usage, category breakdown, and recent uploads
- **🔗 Secure Sharing** — Expiring, optionally password-protected share links
- **📱 Responsive UI** — Dark-mode-ready interface built with React, Tailwind CSS, and Shadcn/UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend framework** | React 19 (Vite 7) |
| **Language** | TypeScript |
| **Styling** | Tailwind CSS v4, Shadcn/UI, Radix UI primitives |
| **Frontend state/API** | Axios, React Router v7, Context API |
| **Icons** | Lucide React |
| **Backend framework** | FastAPI (Python) |
| **Server** | Uvicorn |
| **Database** | Supabase (PostgreSQL + pgvector) |
| **File storage** | Supabase Storage (S3-compatible, via boto3) |
| **AI — content generation** | Google Gemini (`google-genai`) |
| **AI — embeddings** | Voyage AI (`voyageai`) |
| **Auth** | Supabase Auth |
| **Frontend hosting** | Vercel |
| **Backend hosting** | Render |

---

## ⚙️ Architecture

Smart Cloud follows a decoupled client/server architecture:

1. **Frontend** — a React SPA that talks to the backend over REST.
2. **Backend** — a FastAPI service that handles auth, file uploads, AI processing, and search.
3. **Database** — PostgreSQL (via Supabase) stores file metadata and vector embeddings; Supabase Storage holds the actual file bytes.
4. **AI services** — the backend calls Gemini for summarization/categorization and Voyage AI for embeddings at upload and search time.

```
Browser ──▶ React (Vercel) ──▶ FastAPI (Render) ──┬──▶ Supabase Postgres + pgvector (metadata, embeddings)
                                                    ├──▶ Supabase Storage (file bytes)
                                                    ├──▶ Google Gemini (summarize / categorize)
                                                    └──▶ Voyage AI (embeddings)
```

---

## 📁 Project Structure

```
Smart-Cloud/
├── frontend/                  # React + Vite + TypeScript SPA
│   ├── src/
│   │   ├── components/        # Shell, ShareDialog, ui/ (Button, Input, Card)
│   │   ├── pages/              # Login, Signup, Dashboard, Files
│   │   ├── context/            # AuthContext
│   │   └── lib/                 # api.ts (Axios client), utils.ts
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig*.json
│   └── .env.example
├── backend/                   # FastAPI service
│   ├── main.py                 # App entrypoint & routes
│   ├── auth.py
│   ├── database.py
│   ├── models.py
│   ├── hashing.py
│   ├── embedding.py             # Voyage AI embeddings
│   ├── similarity.py            # pgvector similarity search
│   ├── supabase_client.py
│   ├── supabase_storage.py
│   └── requirements.txt
├── render.yaml                # Render deploy blueprint (backend)
└── README.md
```

---

## 🔌 API Reference

Base URL: `https://smart-cloud-backend.onrender.com` (or `http://localhost:8000` locally)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/signup` | Create a new account |
| `POST` | `/auth/login` | Log in, returns a Supabase session token |
| `POST` | `/auth/signout` | Log out |
| `POST` | `/upload-and-analyze/` | Upload a file; triggers AI categorization, summarization & embedding |
| `GET` | `/files` | List all files for the current user |
| `GET` | `/files/{file_id}/download` | Get a signed download URL for a file |
| `DELETE` | `/files/{file_id}` | Delete a file |
| `POST` | `/files/{file_id}/share` | Create a time-limited, optionally password-protected share link |
| `GET` | `/share/{token}` | Resolve a public share link |
| `GET` | `/search?query=` | Semantic search across the user's files |
| `GET` | `/dashboard` | Aggregated dashboard stats (storage, categories, recent files) |
| `POST` | `/generate-hash/` | Utility endpoint for password hashing |
| `GET` | `/health` | Health check |

Interactive docs (Swagger UI) are auto-generated by FastAPI at `/docs`.

---

## 🧰 Getting Started Locally

### Prerequisites
- Node.js 18+
- Python 3.11+
- A [Supabase](https://supabase.com) project (Postgres + pgvector enabled, Storage bucket created)
- A [Google Gemini](https://ai.google.dev/) API key
- A [Voyage AI](https://www.voyageai.com/) API key

### 1. Clone the repo

```bash
git clone https://github.com/Soumyadubeyyy/Smart-Cloud.git
cd Smart-Cloud
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
DATABASE_URL=your_supabase_postgres_connection_string
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_S3_ENDPOINT=your_supabase_s3_endpoint
SUPABASE_S3_REGION=your_supabase_s3_region
SUPABASE_ACCESS_KEY_ID=your_supabase_storage_access_key
SUPABASE_SECRET_ACCESS_KEY=your_supabase_storage_secret_key
SUPABASE_BUCKET_NAME=your_bucket_name
GEMINI_API_KEY=your_gemini_api_key
VOYAGE_API_KEY=your_voyage_api_key
```

Run the API:

```bash
uvicorn main:app --reload --port 8000
```

The backend is now live at `http://localhost:8000` (docs at `/docs`).

### 3. Frontend setup

```bash
cd ../frontend
npm install
```

Create a `.env` file in `frontend/` (see `.env.example`):

```env
VITE_API_URL=http://localhost:8000
```

Run the dev server:

```bash
npm run dev
```

The app is now live at `http://localhost:5173`.

---

## 🚢 Deployment

- **Backend (Render)** — deployed from `backend/` using the `render.yaml` blueprint. Build: `pip install -r requirements.txt`. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`. All env vars listed above must be set in the Render dashboard.
- **Frontend (Vercel)** — deployed from `frontend/` as the project root. Build: `npm run build` (`tsc -b && vite build`). Set `VITE_API_URL` as an environment variable pointing to the live backend URL (or route through `/api` via `vercel.json` rewrites to serve both from a single domain).

---

## 📄 License

No license file is currently included in this repository. All rights reserved by the author unless a license is added.