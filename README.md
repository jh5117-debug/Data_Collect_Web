# Vigil Recorder

Vigil Recorder is a full-stack MVP for collecting clean voice trigger samples for the Vigil wake-word system. It stores raw browser recordings, writes metadata, and provides a simple `/admin` view for exception-based review and export. Online collection is intentionally lightweight: WAV conversion, Qwen ASR review, semantic validation, and final dataset generation are done offline after export.

The product name and wake word is Vigil. The word `visual` appears only as a hard negative prompt.

## Stack

- Frontend: Vite, React, TypeScript, MediaRecorder
- Backend: FastAPI, SQLAlchemy, SQLite locally or Supabase Postgres in production
- Audio: raw browser upload collection online; offline WAV conversion and ASR review
- Storage: local filesystem under `backend/storage/` locally or Supabase Storage in production

## Requirements

- Python 3.11+
- Node.js 20+

Production deployments should use HTTPS so browser microphone permission and uploads are secure.

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The backend creates tables and loads prompts on startup. Local audio and exports are written under `backend/storage/`, which is ignored by Git.

Useful environment variables:

```bash
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=./storage
DATABASE_URL=sqlite:///./storage/vigil_recorder.sqlite3
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

S3 or MinIO variables are placeholders. Production raw collection uses the Supabase Storage backend.

For production raw collection with Supabase:

```bash
STORAGE_BACKEND=supabase
DATABASE_URL=postgresql+psycopg://postgres.PROJECT_REF:YOUR_DB_PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres
SUPABASE_URL=https://PROJECT_REF.supabase.co
SUPABASE_SECRET_KEY=your-server-side-secret-or-service-role-key
SUPABASE_STORAGE_BUCKET=vigil-audio
CORS_ORIGINS=https://your-frontend-domain.example
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` for the participant flow and `http://localhost:5173/admin` for the coordinator view.

## Docker

```bash
docker compose up --build
```

This starts the backend on `http://localhost:8000` and frontend on `http://localhost:5173`.

## Tests

```bash
cd backend
pytest
```

Tests cover prompt group validation, raw upload metadata, export packaging, and deletion behavior.

## Data Flow

1. Participant logs in with an email verification code.
2. Participant completes consent and simple metadata.
3. Participant records a local microphone test. This calibration check is not uploaded.
4. Participant records examples in four prompt groups, plays clips back locally, and accepts or redoes them.
5. Accepted recordings remain in the browser until the final session submission.
6. On final submit, the backend creates `participant_id` and `session_id`, then uploads all accepted recordings.
7. Backend saves each raw upload exactly as received.
8. Backend stores prompt group, transcript, positive/negative labels, session, and account metadata.
9. Backend only performs lightweight collection checks such as empty upload and transcript rules.
10. Admin summary shows account count, sessions, accepted clips, flagged clips, and rejected clips.
11. Export creates a downloadable ZIP with prompts, metadata, raw audio, by-prompt raw copies, and raw manifests.
12. Offline processing converts raw audio to WAV, runs Qwen ASR review, performs manual review, and creates final train/eval manifests.

## API Overview

- `POST /api/participants`
- `POST /api/auth/request-code`
- `POST /api/auth/verify-code`
- `GET /api/auth/accounts/{email}/sessions`
- `POST /api/sessions`
- `GET /api/prompts`
- `POST /api/clips`
- `POST /api/sessions/{session_id}/submit`
- `GET /api/admin/summary`
- `GET /api/admin/flagged`
- `POST /api/admin/export`

## Storage Layout

```text
backend/storage/
  raw_audio/{participant_id}/{session_id}/{clip_id}.webm
  calibration/{participant_id}/{session_id}/{clip_id}.webm
  exports/vigil_dataset_export_{timestamp}.zip
```

The microphone test is currently local-only and is not uploaded.

## Email Login

Set SMTP variables in `backend/.env.example` style to send real login codes. For 163 mail, use the SMTP authorization code as `SMTP_PASSWORD`, not the mailbox login password. If SMTP is not configured, the backend returns a `dev_code` for local testing.

## Current QC Scope

The online QC is collection-level, not semantic. It validates transcript rules and empty uploads. It does not transcribe speech, verify pronunciation, convert WAV, or run a trained wake-word model. Do those steps offline after export.
