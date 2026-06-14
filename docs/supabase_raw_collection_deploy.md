# Supabase Raw Collection Deployment

This deployment mode collects raw browser audio online and does all expensive data processing offline.

Online:
- Participant browser uploads raw `webm` / `m4a` / `wav` files.
- FastAPI stores raw audio in Supabase Storage.
- FastAPI stores metadata in Supabase Postgres.
- Admin can review counts, play raw clips, delete bad clips, and export raw data.

Offline:
- Download the export ZIP.
- Convert raw audio to 16 kHz mono WAV.
- Run Qwen ASR.
- Review failed clips manually.
- Generate final ASR and keyword-spotting train/eval manifests.

## 1. Supabase Setup

Open your Supabase project dashboard.

Create a private Storage bucket:

```text
Storage -> New bucket
Name: vigil-audio
Public bucket: Off
```

Get your database connection string:

```text
Project dashboard -> Connect -> Connection string
```

For long-running Python deployments such as Render, prefer the Supabase session pooler connection string and convert the scheme for SQLAlchemy:

```text
postgresql+psycopg://postgres.PROJECT_REF:YOUR_DB_PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require
```

The transaction pooler usually uses port `6543` and is more appropriate for short-lived/serverless workloads. The backend has compatibility handling for it, but the session pooler on port `5432` is the recommended Render setting.

Get your server-side key:

```text
Project Settings -> API Keys
```

Use a secret/server-side key only in the backend deployment environment. Never put it in frontend code.

## 2. Backend Deployment

Use Render, Railway, Fly.io, or another Python hosting service. The current backend is FastAPI, so Supabase alone does not host it.

Backend root directory:

```text
backend
```

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Backend environment variables:

```bash
STORAGE_BACKEND=supabase
DATABASE_URL=postgresql+psycopg://postgres.PROJECT_REF:YOUR_DB_PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require
SUPABASE_URL=https://PROJECT_REF.supabase.co
SUPABASE_SECRET_KEY=your-server-side-secret-or-service-role-key
SUPABASE_STORAGE_BUCKET=vigil-audio
CORS_ORIGINS=https://YOUR_FRONTEND_DOMAIN

SMTP_HOST=smtp.163.com
SMTP_PORT=465
SMTP_USE_SSL=true
SMTP_USERNAME=your-email@example.com
SMTP_FROM_EMAIL=your-email@example.com
SMTP_PASSWORD=your-smtp-authorization-code
```

After backend deploys, test:

```text
https://YOUR_BACKEND_DOMAIN/api/health
```

Expected response:

```json
{"status":"ok"}
```

## 3. Frontend Deployment

Use Vercel or Netlify.

Frontend root directory:

```text
frontend
```

Build command:

```bash
npm run build
```

Output directory:

```text
dist
```

Frontend environment variable:

```bash
VITE_API_BASE_URL=https://YOUR_BACKEND_DOMAIN
```

After frontend deploys, return to the backend environment and set:

```bash
CORS_ORIGINS=https://YOUR_FRONTEND_DOMAIN
```

Redeploy the backend after changing CORS.

## 4. End-to-End Test

Open the frontend URL in a normal browser, not localhost.

Test this path:

1. Login with email code.
2. Start a new session.
3. Complete consent and participant details.
4. Record one clip in Prompt 1.
5. Record one clip in Prompt 2 using `Hi VIGIL.`.
6. Record one clip in Prompt 4 using `visual`.
7. Submit the session.
8. Open `/admin`.
9. Confirm positive and negative counts.
10. Play one clip.
11. Export the dataset.

In Supabase, check:

```text
Table Editor -> clips
Storage -> vigil-audio -> raw_audio/
```

## 5. Offline Processing

Download the admin export ZIP.

The export contains:

```text
metadata/
raw_audio/
audio_raw/
by_prompt_group/
qwen_asr/
keyword_spotting/
```

The `qwen_asr` and `keyword_spotting` manifests point to raw audio paths in collection mode. For final training, run an offline script to:

1. Convert raw audio to 16 kHz mono WAV.
2. Update manifest audio paths to the converted WAV files.
3. Run Qwen ASR review.
4. Remove or re-record failed clips.
5. Generate final train/eval JSONL files.

## Security Rules

- Do not commit `.env`.
- Do not commit audio files.
- Do not commit export ZIP files.
- Do not put `SUPABASE_SECRET_KEY` or service role keys in frontend variables.
- Frontend should only know `VITE_API_BASE_URL`.
- Backend should be HTTPS in production.
