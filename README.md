# Vigil Recorder

Vigil Recorder is a full-stack MVP for collecting clean voice trigger samples for the Vigil wake-word system. It stores raw browser recordings, converts accepted uploads to 16 kHz mono WAV, attempts energy-based segmentation for repeated prompts, writes SQLite metadata, and provides a simple `/admin` view for exception-based review and export.

The product name and wake word is Vigil. The word `visual` appears only as a hard negative prompt.

## Stack

- Frontend: Vite, React, TypeScript, MediaRecorder
- Backend: FastAPI, SQLAlchemy, SQLite
- Audio: FFmpeg conversion to 16 kHz mono WAV, NumPy-based QC and segmentation
- Storage: local filesystem under `backend/storage/`

## Requirements

- Python 3.11+
- Node.js 20+
- FFmpeg available on `PATH`

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

S3 or MinIO variables are present as placeholders in `backend/.env.example`; the MVP implements local storage only.

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

The clip conversion success test is skipped automatically if FFmpeg is not installed. Conversion failure handling is still tested.

## Data Flow

1. Participant logs in with an email verification code.
2. Participant completes consent and simple metadata.
3. Participant records a local microphone test. This calibration check is not uploaded.
4. Participant records each prompt at least twice, plays clips back locally, and accepts or redoes them.
5. Accepted recordings remain in the browser until the final session submission.
6. On final submit, the backend creates `participant_id` and `session_id`, then uploads all accepted recordings.
7. Backend saves each raw upload exactly as received.
8. Backend converts each upload to 16 kHz mono WAV with FFmpeg.
9. Backend runs technical automatic QC.
10. For repeated prompts, backend attempts silence-based segmentation and stores derived segment WAVs.
11. Admin summary shows account count, sessions, accepted clips, flagged clips, rejected clips, and generated segments.
12. Export creates a downloadable ZIP with prompts, metadata, raw audio, processed WAVs, segment WAVs, and by-prompt copies.

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
  processed_wav/{participant_id}/{session_id}/{clip_id}.wav
  segments/{participant_id}/{session_id}/{parent_clip_id}_seg001.wav
  calibration/{participant_id}/{session_id}/{clip_id}.webm
  exports/vigil_dataset_export_{timestamp}.zip
```

The microphone test is currently local-only and is not uploaded.

## Email Login

Set SMTP variables in `backend/.env.example` style to send real login codes. For 163 mail, use the SMTP authorization code as `SMTP_PASSWORD`, not the mailbox login password. If SMTP is not configured, the backend returns a `dev_code` for local testing.

## Current QC Scope

The current automatic QC is technical, not semantic. It checks empty audio, duration, FFmpeg conversion, low volume, clipping, and segmentation count mismatch. It does not currently transcribe speech or verify that the participant said the expected phrase.
