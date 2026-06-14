# Vigil Recorder Handoff

Last updated: 2026-06-14
Repository: `jh5117-debug/Data_Collect_Web`
Current branch: `main`

## 1. Current Production URLs

Frontend:

```text
https://data-collect-web.vercel.app
```

Participant app:

```text
https://data-collect-web.vercel.app
```

Admin app:

```text
https://data-collect-web.vercel.app/admin
```

Backend health check:

```text
https://data-collect-web.onrender.com/api/health
```

Expected health response:

```json
{"status":"ok","app":"Vigil Recorder"}
```

Admin summary API:

```text
https://data-collect-web.onrender.com/api/admin/summary
```

## 2. Service Responsibilities

### Vercel

Vercel hosts the static React/Vite frontend.

It serves:

- participant UI
- admin UI
- browser recording page
- local playback UI
- upload requests to the backend

Required frontend environment variable:

```text
VITE_API_BASE_URL=https://data-collect-web.onrender.com
```

### Render

Render hosts the FastAPI backend.

It handles:

- email-code login
- participant sessions
- prompt validation
- raw audio uploads
- Supabase Storage writes
- Supabase Postgres metadata writes
- admin summaries
- playback/download routes
- deletion routes
- export routes

### Supabase

Supabase stores persistent data.

Supabase Postgres stores metadata:

- accounts
- login tokens
- participants
- sessions
- clips
- transcript fields
- prompt labels
- storage paths

Supabase Storage stores raw audio files.

Current bucket:

```text
vigil-audio
```

The bucket should stay private.

## 3. Connection Map

```text
Browser
  -> Vercel frontend
  -> Render backend API
  -> Supabase Postgres
  -> Supabase Storage
```

Detailed flow:

```text
Participant records audio in browser
  -> frontend sends raw audio, prompt_group, transcript to Render
  -> Render validates prompt_group and transcript
  -> Render uploads raw audio to Supabase Storage
  -> Render writes clip metadata to Supabase Postgres
  -> admin reads summary/session/clip data through Render
```

The browser does not receive Supabase service keys.

## 4. Required Environment Variables

### Render Backend

Use real values in Render only. Do not commit these values to Git.

```text
STORAGE_BACKEND=supabase
DATABASE_URL=postgresql+psycopg://postgres.PROJECT_REF:YOUR_DB_PASSWORD@aws-REGION.pooler.supabase.com:6543/postgres
SUPABASE_URL=https://PROJECT_REF.supabase.co
SUPABASE_SECRET_KEY=YOUR_SERVER_SIDE_SECRET
SUPABASE_STORAGE_BUCKET=vigil-audio
CORS_ORIGINS=http://localhost:5173,https://data-collect-web.onrender.com,https://data-collect-web.vercel.app

SMTP_HOST=smtp.163.com
SMTP_PORT=465
SMTP_USE_SSL=true
SMTP_USERNAME=YOUR_EMAIL
SMTP_FROM_EMAIL=YOUR_EMAIL
SMTP_PASSWORD=YOUR_SMTP_AUTHORIZATION_CODE
```

### Vercel Frontend

```text
VITE_API_BASE_URL=https://data-collect-web.onrender.com
```

## 5. Deployment Notes

### Backend on Render

The repo includes a root `Dockerfile` for Render.

Current expected behavior:

- Render builds the Docker image from the repository root.
- The Dockerfile copies `backend/`.
- The container runs `uvicorn`.
- Render provides `$PORT`.

Useful verification URLs:

```text
https://data-collect-web.onrender.com/api/health
https://data-collect-web.onrender.com/api/admin/summary
```

If CORS is changed, update `CORS_ORIGINS` in Render and click:

```text
Save, rebuild, and deploy
```

### Frontend on Vercel

Vercel project should use:

```text
Root Directory: frontend
Install Command: npm ci
Build Command: npm run build
Output Directory: dist
```

The root `vercel.json` is configured for the `frontend` root directory.

If Vercel is instead configured with repository root `./`, the build commands must be changed accordingly. The current recommended setup is to keep root directory as `frontend`.

## 6. Local Development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Local backend URL:

```text
http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Local frontend URLs:

```text
http://localhost:5173
http://localhost:5173/admin
```

For local development, use:

```text
VITE_API_BASE_URL=http://localhost:8000
```

## 7. End-to-End Smoke Test

After every deployment, test:

1. Open `https://data-collect-web.vercel.app`.
2. Start the participant flow.
3. Log in with an email code.
4. Start a new session.
5. Complete consent and participant details.
6. Run the local microphone check.
7. Record one Prompt 1 clip.
8. Record one Prompt 2 clip using `Hi VIGIL.`.
9. Record one Prompt 3 clip using `VIGIL, go back.`.
10. Record one Prompt 4 negative clip using `visual`.
11. Submit the session.
12. Open `/admin`.
13. Confirm positive and negative counts.
14. Play one clip.
15. Delete a test clip if needed.
16. Check Supabase Storage for new raw audio objects.
17. Check Supabase Postgres metadata through admin summary or Supabase Table Editor.

## 8. Current Prompt Groups

```text
P1_vigil_only
P2_phrase_plus_vigil
P3_vigil_plus_phrase
P4_negative
```

Backend derives:

- `prompt_title`
- `normalized_transcript`
- `contains_vigil`
- `wake_intent`
- `is_negative`

P4 rejects the exact word `Vigil`, but accepts words like `visual`, `digital`, `individual`, `visible`, and `vigilant`.

## 9. Data Storage

Current production storage is raw-only:

- raw browser audio goes to Supabase Storage
- metadata goes to Supabase Postgres
- online collection does not run Qwen ASR
- online collection does not generate final training WAV files

Future Supabase-style storage path:

```text
{account_id}/{session_id}/{prompt_group}/{clip_id}.webm
```

Exact path may differ by backend implementation, but metadata must always store `storage_path`.

## 10. Export and Offline Processing

Admin export should produce raw audio and metadata for offline processing.

Offline processing should:

1. Download export ZIP.
2. Convert raw audio to 16 kHz mono WAV.
3. Run Qwen ASR.
4. Compare ASR output against intended transcript.
5. Manually review failed or suspicious clips.
6. Remove or re-record bad clips.
7. Generate final Qwen ASR manifests.
8. Generate final keyword spotting manifests.

Suggested offline output:

```text
processed_wav/
qwen_asr/train.jsonl
qwen_asr/eval.jsonl
keyword_spotting/kws_train.jsonl
keyword_spotting/kws_eval.jsonl
```

## 11. Security Rules

- Do not commit `.env`.
- Do not commit audio files.
- Do not commit export ZIP files.
- Do not put Supabase service keys in frontend variables.
- Do not expose SMTP authorization codes in Git.
- Keep Supabase Storage bucket private.
- Keep admin URLs unshared until admin authentication is strengthened.
- Rotate any key that was accidentally pasted into chat, logs, screenshots, or Git.

## 12. Troubleshooting

### Frontend opens but API requests fail

Likely cause: CORS mismatch.

Fix:

- Add the Vercel URL to Render `CORS_ORIGINS`.
- Redeploy Render.

Current expected value:

```text
http://localhost:5173,https://data-collect-web.onrender.com,https://data-collect-web.vercel.app
```

### Vercel deployment fails at install

Likely cause: root directory mismatch.

Expected Vercel settings:

```text
Root Directory: frontend
Install Command: npm ci
Build Command: npm run build
Output Directory: dist
```

### Render says Dockerfile not found

Render must use the root Dockerfile. The repository now includes one at:

```text
Dockerfile
```

### Login code email does not send

Check Render environment variables:

```text
SMTP_HOST
SMTP_PORT
SMTP_USE_SSL
SMTP_USERNAME
SMTP_FROM_EMAIL
SMTP_PASSWORD
```

For 163 mail, `SMTP_PASSWORD` should be the SMTP authorization code, not the normal mailbox password.

### Supabase upload fails

Check:

- `STORAGE_BACKEND=supabase`
- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `SUPABASE_STORAGE_BUCKET=vigil-audio`
- bucket exists
- bucket is private
- backend logs in Render

### Prompt 4 accepts the word Vigil

This is a bug. P4 must reject exact word `Vigil` using word boundary matching.

Allowed:

```text
vigilant
visual
digital
individual
visible
```

Rejected:

```text
Vigil
Vigil next
Hi Vigil
```

## 13. Cost Notes

Current MVP costs come from:

- Vercel frontend hosting
- Render backend runtime
- Supabase database
- Supabase storage
- Supabase bandwidth
- email sending

Future high-cost areas:

- audio storage
- audio export downloads
- GPU ASR processing
- backups
- long-term retention

If the lab prefers one cloud vendor, AWS can replace the current stack:

```text
Frontend: Amplify Hosting or S3 + CloudFront
Backend: ECS/Fargate, App Runner, Lambda, or Elastic Beanstalk
Database: RDS PostgreSQL or Aurora PostgreSQL
Audio files: S3
Email: SES
Auth: Cognito or custom email-code login
```

This may be more suitable for a long-term lab deployment, but it requires more operational setup than the current MVP stack.

## 14. Immediate Next Steps

Recommended before broad participant collection:

1. Add admin authentication.
2. Run a full end-to-end smoke test with a non-local browser.
3. Confirm Supabase Storage receives raw audio.
4. Confirm admin export works on production data.
5. Decide data retention policy.
6. Decide whether AWS migration is needed for the lab.
7. Write offline conversion and Qwen ASR review scripts.
8. Define final train/eval split rules.
9. Add cost alerts in Supabase, Render, and Vercel.

