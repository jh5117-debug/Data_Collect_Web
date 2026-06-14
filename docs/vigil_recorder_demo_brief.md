# Vigil Recorder Data Collection Website Demo Brief

Note: This file is the earlier demo brief. For the current production architecture,
prompt-group design, and deployment handoff, use [PRD.md](PRD.md) and
[HANDOFF.md](HANDOFF.md) as the canonical documents.

## 1. Project Goal

Vigil Recorder is a web-based data collection tool for gathering clean voice trigger samples for the Vigil wake-word system.

The immediate goal is not model training inside the website. The goal is to collect structured, clean, reviewable audio data that can later support:

- wake-word detection experiments
- ASR fine-tuning
- audio embedding similarity experiments
- future batch ASR quality checks
- prompt-based dataset export

The product name and wake word is **Vigil**. The website must not confuse it with “Visual.”

## 2. What This Demo Currently Supports

### Participant Side

The participant flow currently includes:

1. Welcome page
2. Email login with verification code
3. Participant workspace
4. Consent and privacy confirmation
5. Low-burden metadata collection
6. Local microphone test
7. Prompt-based recording
8. Multiple recordings per prompt
9. Local playback before submission
10. Final bulk upload and session submission
11. Participant session history
12. Participant-owned clip review and deletion

Important behavior:

- The microphone test is local only and is not uploaded.
- Accepted recordings stay in the browser until final submission.
- The backend only creates the participant/session and saves audio after final `Upload All & Submit Session`.
- If a participant starts but never submits, incomplete recordings are not saved to the backend dataset.

### Current Login Behavior

The demo uses email verification instead of passwords.

- If an email is new, verification effectively creates an account.
- If an email already exists, verification logs the participant in.
- The same device keeps a one-hour login token.
- Recent emails are remembered locally on the same device.

This is an MVP-level authentication flow. It is not yet a full production auth system.

### Recording Behavior

For debugging, the current prompt list is intentionally short:

1. Vigil
2. Hey Vigil
3. Hi Vigil
4. Vigil next
5. Vigil previous

Each prompt currently requires at least two accepted recordings. Participants can record more than two if needed.

### Admin Side

The admin dashboard currently supports:

- overall summary metrics
- account/client list
- per-client drill-down
- session list per client
- clip list per client
- playback for submitted clips
- deletion of individual bad clips
- deletion of an entire client and associated data
- export of the dataset ZIP
- needs-review table for flagged clips

Admin has higher privileges than participants. Participants can only access their own session/clip data through a token-scoped endpoint.

## 3. Data Currently Collected

### Participant-provided metadata

The participant only provides low-burden metadata:

- email login identity
- native/non-native/prefer-not-to-say English status
- recording device type

The demo intentionally avoids asking participants for difficult or unreliable metadata such as:

- exact microphone distance
- exact environment
- detailed background noise level
- speaking speed
- speaking volume
- microphone model

### System-generated metadata

The backend generates:

- participant ID
- session ID
- clip ID
- prompt ID
- batch ID
- timestamps
- raw audio path
- processed WAV path
- clip duration
- file size
- QC status
- QC flags
- segment count fields

## 4. Current Audio Processing

When a submitted recording is uploaded, the backend:

1. Saves the raw browser upload.
2. Converts it to 16 kHz mono WAV using FFmpeg.
3. Computes duration and file size.
4. Runs technical automatic QC.
5. Saves metadata to SQLite.

Current QC checks are technical, not semantic:

- empty audio
- too short duration
- suspiciously short or long duration
- FFmpeg conversion failure
- very low volume
- severe clipping
- segmentation mismatch for repeated prompts

Current QC does **not** yet verify whether the participant actually said the correct prompt.

## 5. Current Storage Layout

The current local backend stores files mainly by participant and session:

```text
backend/storage/
  raw_audio/{participant_id}/{session_id}/{clip_id}.webm
  processed_wav/{participant_id}/{session_id}/{clip_id}.wav
  segments/{participant_id}/{session_id}/...
  exports/...
  vigil_recorder.sqlite3
```

This layout is useful for:

- deleting a participant
- deleting a session
- tracing dataset provenance
- auditing who submitted what
- recovering from bad submissions

The export ZIP also creates a prompt-organized view:

```text
by_prompt/POS_SINGLE_001/raw_audio/
by_prompt/POS_SINGLE_001/processed_wav/
by_prompt/POS_SINGLE_002/raw_audio/
by_prompt/POS_SINGLE_002/processed_wav/
```

So the training dataset can be consumed by prompt, even if the internal storage keeps participant/session provenance.

## 6. Current Export Format

The admin export ZIP includes:

```text
README.md
prompts/prompts_v0_1.csv
metadata/accounts.csv
metadata/participants.csv
metadata/sessions.jsonl
metadata/clips.jsonl
metadata/segments.jsonl
metadata/qc_report.csv
raw_audio/
processed_wav/
segments/
by_prompt/
```

This gives both:

- traceable source data
- prompt-grouped data for model work

## 7. Important Current Limitations

This demo is functional, but still an MVP.

Current limitations:

- It runs on a local machine unless deployed.
- Microphone access for remote users requires HTTPS.
- Admin login is not implemented yet.
- SQLite is fine for MVP but not ideal for production concurrency.
- Local filesystem storage is not sufficient for real distributed collection.
- ASR-based semantic review is not implemented yet.
- Browser-local accepted recordings can be lost if the participant refreshes before final submit.
- Generated export ZIP files are static snapshots and are not automatically invalidated after later deletions.

## 8. ASR Review Discussion

The important future QC question is semantic correctness:

Did the participant say the expected prompt?

Examples:

- Expected: “Hi Vigil”
- Acceptable variants might include accent or pronunciation differences.
- Bad sample: “Hi Apple”

Current technical QC cannot detect this. A future ASR review step could:

1. Wait until a participant submits all recordings.
2. Run ASR over the submitted clips.
3. Compare ASR output against expected prompt text.
4. Flag clips that are too different.
5. Ask the participant to re-record only the failed clips.

Potential ASR model:

- Qwen/Qwen3-ASR-1.7B

Key infrastructure implication:

- Running this model likely requires a GPU server or a separate GPU worker.
- We do not need to run ASR after every single clip.
- Batch ASR after final submission is likely cheaper and simpler.

## 9. Deployment Options To Discuss

### Option A: Local machine only

Pros:

- cheapest
- easy for internal debugging

Cons:

- not reliable
- not accessible remotely unless using tunnel/VPN
- not production-safe
- microphone access is problematic without HTTPS

### Option B: VPS with Docker

Pros:

- predictable cost
- can run frontend/backend together
- can use attached disk storage
- easy to add HTTPS with Nginx/Caddy

Cons:

- we manage backups
- we manage security updates
- storage scaling is manual
- no managed auth/database

### Option C: Supabase for auth/database/storage

Supabase could provide:

- email auth
- Postgres database
- object storage for audio
- dashboard tools
- access rules

Pros:

- faster productionization
- managed auth
- managed Postgres
- easier permission model
- easier remote deployment story

Cons:

- cost can grow with storage, bandwidth, and active usage
- audio files are large compared with normal app data
- export/download bandwidth may become expensive
- vendor lock-in
- GPU ASR still needs a separate service

### Option D: Hybrid architecture

Possible hybrid:

- Supabase Auth for accounts
- Supabase Postgres for metadata
- Cloudflare R2 / S3 for audio storage
- Separate GPU worker for ASR
- Frontend on Vercel/Cloudflare Pages
- Backend on small VPS or serverless API

Pros:

- separates metadata, audio, and ASR compute
- object storage can be cheaper than database storage
- easier to scale components independently

Cons:

- more moving parts
- more integration work
- more operational complexity

## 10. Cost Questions For Professor

We should ask the professor the following before choosing the final architecture.

### Dataset scale

- How many participants do we expect?
- How many sessions per participant?
- How many prompts per session?
- How many recordings per prompt?
- What is the target total number of clips?
- What is the expected average clip duration?

### Storage and retention

- How long do we need to keep raw browser audio?
- Do we need to keep both raw audio and processed WAV permanently?
- Can we delete raw files after WAV conversion and export?
- Should deleted participant data also be removed from old export ZIPs?
- Do we need automatic backups?
- How many backup copies are required?

### Privacy and compliance

- Is this internal research only or will external participants use it?
- Do we need IRB review or consent language approval?
- Are voice recordings considered sensitive data under our project policy?
- Do we need a formal data retention policy?
- Who should have admin access?
- Do participants need the right to delete all their data?

### Authentication

- Is email-code login enough?
- Do we need institutional login?
- Should admin use a separate login system?
- Should participants be able to edit or delete already submitted clips?
- Should there be an audit log for admin deletions?

### Storage architecture

- Should we use Supabase for auth, database, and audio storage?
- If Supabase storage costs grow, what is the budget limit?
- Would S3, Cloudflare R2, or MinIO be better for audio files?
- Should metadata and audio live in separate systems?
- Do we need region-specific storage?

### Cost ownership

- Who pays for hosting?
- Is there a monthly budget cap?
- Should we optimize for minimum cost, fastest development, or reliability?
- Is occasional downtime acceptable during data collection?

### ASR review

- Do we need semantic ASR review before accepting submitted data?
- Is Qwen/Qwen3-ASR-1.7B the intended ASR model?
- Do we have access to a GPU server?
- Should ASR run immediately after final submission or as an offline batch job?
- Should participants wait for ASR results and re-record failed clips in the same session?
- What false reject rate is acceptable?
- Should ASR output be stored as metadata?

### Dataset export format

- Should the canonical dataset be organized by prompt, participant, session, or all three?
- Should each prompt folder include raw audio, WAV, or only WAV?
- What naming convention does the downstream training pipeline expect?
- Should metadata be CSV, JSONL, Parquet, or all of them?

## 11. Suggested Recommendation

For the next stage, the safest path is:

1. Keep the current MVP for workflow debugging.
2. Add admin authentication before any real external use.
3. Decide expected dataset scale and monthly budget.
4. Use managed auth/database if we expect external participants.
5. Store audio in object storage rather than inside the database.
6. Keep prompt-grouped export as the training-facing format.
7. Add ASR review as a separate batch worker, not inside the frontend.

If the professor wants the fastest path to a real pilot, a reasonable architecture is:

```text
Frontend: Vercel or Cloudflare Pages
Auth: Supabase Auth
Metadata DB: Supabase Postgres
Audio Storage: Supabase Storage or Cloudflare R2/S3
Backend API: small VPS or serverless functions
ASR Review: separate GPU worker
```

The biggest cost drivers will likely be:

- audio storage volume
- audio download/export bandwidth
- GPU ASR compute
- backup retention
- managed platform pricing limits

## 12. Demo Talking Points

Short presentation script:

> This demo is a full-stack MVP for collecting Vigil wake-word audio data. Participants log in with email, complete consent and low-burden metadata, test their microphone locally, record each prompt multiple times, review recordings locally, and only submit once the full session is complete. The backend then stores raw audio, converts it to 16 kHz mono WAV, writes metadata, runs technical QC, and supports admin review and dataset export.

> The admin view provides global metrics, client-level drill-down, clip playback, deletion of bad clips, deletion of a client’s data, and export. Internally we keep participant/session provenance, while the export also creates prompt-grouped folders for training use.

> The major open design questions are deployment, storage, cost, privacy, and semantic quality control. In particular, we need to decide whether to use Supabase or a hybrid stack, where audio should live, how much budget we have, and whether ASR review should run with a GPU worker after final submission.
