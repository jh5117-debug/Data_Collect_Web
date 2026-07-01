# Vigil Recorder

Vigil Recorder is a full-stack MVP for collecting clean voice trigger samples for the Vigil wake-word system. It stores raw browser recordings, writes metadata, and provides a simple `/admin` view for exception-based review and export. Online collection is intentionally lightweight: WAV conversion, Qwen ASR review, semantic validation, and final dataset generation are done offline after export.

The product name and wake word is Vigil. The word `visual` appears only as a hard negative prompt.

## Stack

- Frontend: Vite, React, TypeScript, MediaRecorder
- Backend: FastAPI, SQLAlchemy, SQLite locally or Supabase Postgres in production
- Audio: raw browser upload collection online; offline WAV conversion and ASR review
- Storage: local filesystem under `backend/storage/` locally or Supabase Storage in production

## Project Documents

- [Product Requirements Document](docs/PRD.md)
- [Production Handoff](docs/HANDOFF.md)
- [Supabase raw collection deployment notes](docs/supabase_raw_collection_deploy.md)
- [VIGIL trigger integration guide](docs/VIGIL_TRIGGER_INTEGRATION.md)
- [VIGIL current status](docs/VIGIL_CURRENT_STATUS.md)
- [VIGIL model architecture](docs/VIGIL_MODEL_ARCHITECTURE.md)
- [VIGIL experiment results](docs/VIGIL_EXPERIMENT_RESULTS.md)
- [VIGIL browser demo guide](docs/VIGIL_BROWSER_DEMO.md)
- [VIGIL data collection protocol](docs/VIGIL_DATA_COLLECTION_PROTOCOL.md)

## VIGIL Research And Demo

The repository now contains the latest VIGIL two-stage trigger work in addition to the recorder app. The current system uses a continuous frozen Qwen3-ASR branch for transcript/report text and a parallel VIGIL branch with Stage 1 openWakeWord candidate detection plus Stage 2 frozen-Qwen-feature verification.

Current headline results:

- Optimized two-stage trigger: recall `0.9409`, FPR `0.0050`, precision `0.9957`, F1 `0.9675`.
- Corrected frozen-Qwen LibriSpeech ASR: combined WER `2.7516%`.
- Few-shot best method: 5-shot Stage 2 cosine prototype, F1 `0.97059`.

Finetune smoke:

```bash
cd /home/hj/Data_Collect_Web
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH bash finetune/scripts/run_official_smoke_local_3090.sh
```

Local HAL browser assistant demo:

```bash
cd /home/hj/Data_Collect_Web
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
bash finetune/demo_live_assistant/scripts/run_demo.sh 6 \
  /home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full
```

Main reports live under `finetune/reports/`, `finetune/benchmarks/asr/reports/`, `finetune/experiments/latest_data_optimization/reports/`, `finetune/experiments/fewshot_ablation/reports/`, and `finetune/demo_live_assistant/reports/`.

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
DATABASE_URL=postgresql+psycopg://postgres.PROJECT_REF:YOUR_DB_PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require
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
