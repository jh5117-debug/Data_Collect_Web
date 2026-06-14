# Vigil Recorder Product Requirements Document

Status: MVP pilot ready
Last updated: 2026-06-14
Repository: `jh5117-debug/Data_Collect_Web`

## 1. Overview

Vigil Recorder is a web-based data collection system for gathering clean voice samples for the Vigil wake-word project. The system collects raw browser audio and structured metadata online, then supports offline conversion, ASR review, manual review, and final dataset generation.

The product name and wake word is `Vigil`. The word `visual` is allowed only as a negative example prompt.

## 2. Problem

The project needs a repeatable way to collect audio examples for a wake-word or voice-trigger system. A simple local demo is not enough because remote participants need a stable link, browser microphone access, account-specific workspaces, and a way for coordinators to review and export submitted data.

The system must preserve the relationship between:

- who recorded the audio
- which session the audio belongs to
- which prompt group the audio belongs to
- the exact transcript the participant intended to say
- whether the clip is a positive Vigil example or a negative example

## 3. Goals

- Collect raw voice recordings from remote participants through a browser.
- Keep participant flow simple and low-friction.
- Support four prompt groups for Vigil positive and negative examples.
- Store metadata in a database and raw audio files in object storage.
- Allow participants to view, play, and delete only their own submitted data.
- Allow admin/coordinator users to review accounts, sessions, clips, and summary metrics.
- Allow admin users to delete bad clips, sessions, or accounts.
- Export raw data and metadata for offline processing.
- Keep online collection lightweight and avoid GPU-dependent processing during collection.

## 4. Non-Goals

- Do not train a model inside the website.
- Do not run Qwen ASR online during every upload.
- Do not convert all files to WAV online during participant upload.
- Do not implement temporal segmentation in the current UI.
- Do not store SMTP, Supabase, or database secrets in frontend code or Git.
- Do not treat the current email-code login as a full enterprise authentication system.

## 5. Users

### Participant

The participant records audio examples and submits one or more sessions. They should be able to:

- log in with email
- continue on the same device without repeated login during a short active window
- start a new recording session
- complete consent and minimal metadata
- test microphone locally
- record examples under each prompt group
- play recordings before upload
- delete recordings
- submit the session
- view their own submitted sessions and clips

Participants must not be able to access other participants' data.

### Admin / Coordinator

The admin monitors collection quality and dataset status. They should be able to:

- view global summary metrics
- view prompt group counts
- view accounts
- open one account and inspect sessions
- open one session and inspect clips
- play submitted clips
- filter clips
- delete bad clips
- delete sessions
- delete accounts and associated data
- export raw data and metadata

## 6. Current Architecture

Production currently uses three hosted services:

```text
Participant/Admin browser
  -> Vercel frontend
  -> Render FastAPI backend
  -> Supabase Postgres for metadata
  -> Supabase Storage for raw audio
```

### Vercel

Vercel hosts the React/Vite frontend.

Responsibilities:

- participant UI
- admin UI
- browser recording controls
- playback before upload
- upload requests to backend

Vercel does not store audio or metadata.

Connection:

```text
VITE_API_BASE_URL=https://data-collect-web.onrender.com
```

### Render

Render hosts the Python FastAPI backend.

Responsibilities:

- email-code login API
- session creation
- prompt validation
- raw audio upload handling
- metadata writes
- admin summary APIs
- playback/download proxy routes
- deletion logic
- export generation

Connection:

```text
Frontend -> https://data-collect-web.onrender.com/api/...
```

Render must allow the Vercel frontend through CORS.

### Supabase

Supabase provides managed storage and metadata persistence.

Responsibilities:

- Supabase Postgres stores accounts, sessions, clips, transcript fields, labels, and storage paths.
- Supabase Storage stores raw browser audio files.

Supabase service keys must only exist in backend environment variables.

## 7. Prompt Groups

The current prompt design has four groups.

### Prompt 1: VIGIL Only

Prompt group:

```text
P1_vigil_only
```

Participant instruction:

```text
Please say "VIGIL" once per recording.
You can upload as many recordings as you like. The more the better.
```

Transcript:

```text
Vigil
```

Labels:

```text
contains_vigil = true
wake_intent = true
is_negative = false
```

No transcript input is required.

### Prompt 2: Phrase/Sentence + VIGIL

Prompt group:

```text
P2_phrase_plus_vigil
```

Participant instruction:

```text
Please say a phrase or sentence ending with or followed by "VIGIL" in one recording.
You can upload as many recordings as you like, with the same or different phrases/sentences.
```

Examples:

- Hi VIGIL.
- Hey VIGIL.
- Hello VIGIL.
- Next, VIGIL.
- What's next, VIGIL?
- Am I doing right, VIGIL?

Labels:

```text
contains_vigil = true
wake_intent = true
is_negative = false
```

Validation:

- transcript must be non-empty
- transcript must contain exact word `Vigil`, case-insensitively
- validation must use word boundary matching, not simple substring matching

### Prompt 3: VIGIL + Phrase/Sentence

Prompt group:

```text
P3_vigil_plus_phrase
```

Participant instruction:

```text
Please say "VIGIL" plus a phrase or sentence in one recording.
You can upload as many recordings as you like, with the same or different phrases/sentences.
```

Examples:

- VIGIL, next.
- VIGIL, go back.
- VIGIL, what's next?
- VIGIL, am I doing right?

Labels:

```text
contains_vigil = true
wake_intent = true
is_negative = false
```

Validation:

- transcript must be non-empty
- transcript must contain exact word `Vigil`, case-insensitively
- starting with `Vigil` is preferred but not strictly required

### Prompt 4: Negative Examples

Prompt group:

```text
P4_negative
```

Participant instruction:

```text
Please record confusing common words or sentences.
These recordings should NOT wake up Vigil.
Do not say the exact word "Vigil" in this section.
```

Examples:

- visual
- visuals
- visible
- digital
- individual
- residual
- video
- vital
- vigilant
- This is a visual input.
- The video is clear.
- The image is visible.
- This is a digital system.
- The individual is moving.
- The vital signs are normal.

Labels:

```text
contains_vigil = false
wake_intent = false
is_negative = true
```

Validation:

- transcript must be non-empty
- transcript must not contain exact word `Vigil`, case-insensitively
- `vigilant` is allowed because it is not the exact word `Vigil`

## 8. Participant Requirements

### Login

- The participant logs in with an email verification code.
- New email verification creates an account.
- Existing email verification logs into the existing account.
- The same device may keep a short-lived login token.
- Recent email addresses may be remembered locally on the same device.

### Session Flow

The participant flow should be:

1. Welcome page
2. Email login
3. Participant workspace
4. Start new session
5. Consent
6. Participant details
7. Local microphone check
8. Four-card recording workspace
9. Review accepted recordings per card
10. Upload all and submit session
11. Session history and session detail

### Recording Workspace

The recording page should show four large prompt cards in one workspace, not a strict linear next-prompt flow.

Each card should include:

- prompt group title
- instruction
- example chips where applicable
- transcript input where applicable
- record / stop / playback / accept / redo controls
- accepted recordings table
- count badge

Accepted recordings table columns:

```text
Transcript | Take | Playback | Status | Delete
```

The accepted recordings area should be scrollable after approximately five rows so the page remains usable.

### Recording Counts

Each prompt card should show a count badge:

- `0`: neutral/empty style
- `1`: light green style
- `2+`: stronger green style

Example chips should also show how many times that exact transcript has been accepted.

### Upload Behavior

Current intended behavior:

- Accepted recordings remain local drafts until final session submission.
- Final submission uploads all accepted clips together.
- Draft status should be visible before upload.
- Uploaded status should be visible after successful upload.
- Failed uploads should remain visible and count toward failed upload summary.

If a previously uploaded clip is deleted, deletion must call the backend and remove both metadata and raw audio.

## 9. Admin Requirements

### Admin Homepage

The admin homepage should show:

- summary metrics
- prompt group summary
- accounts table
- needs review table

The admin homepage should not show all clips directly. Clip-level inspection belongs inside account/session detail pages.

Summary metrics:

- accounts
- sessions
- submitted sessions
- total clips
- positive clips
- negative clips
- auto accepted
- flagged
- rejected

Prompt group summary:

- P1_vigil_only
- P2_phrase_plus_vigil
- P3_vigil_plus_phrase
- P4_negative
- legacy, if any

Accounts table:

```text
Email | Verified | Sessions | Submitted | Clips | Positive | Negative | Last login | Open
```

### Account Detail

The admin account detail should show:

- account email
- session table
- delete account button
- delete all sessions button

Session table:

```text
Session | Status | Total clips | Positive clips | Negative clips | Submitted | Open | Delete Session
```

### Session Detail

The admin session detail should show clips within one session.

Columns:

```text
Clip | Prompt Group | Transcript | Contains Vigil | Wake Intent | Status | Flags | Duration | Size | Created | Playback | Delete
```

Filters:

- All
- Positive only
- Negative only
- P1
- P2
- P3
- P4
- Flagged only

### Deletion

All destructive actions require confirmation.

Deletion requirements:

- delete clip removes metadata row and raw audio object
- delete session removes session, clips, and raw audio objects
- delete account removes account, sessions, clips, raw audio objects, and login tokens
- counts update immediately after deletion
- old export snapshots should not be treated as canonical after deletion

## 10. Data Model

The backend should keep these conceptual entities.

### Account

- account_id
- email
- verified
- last_login_at
- created_at

### Participant

- participant_id
- account_id
- participant details
- created_at

### Session

- session_id
- account_id
- participant_id
- batch_id
- status
- submitted_at
- created_at

### Clip

Required fields:

- clip_id
- account_id
- participant_id
- session_id
- prompt_group
- prompt_title
- transcript
- normalized_transcript
- contains_vigil
- wake_intent
- is_negative
- storage_path
- original_filename
- content_type
- duration_seconds, if available
- file_size_bytes
- status
- flags
- created_at

### Legacy Clip Handling

Existing `POS_SINGLE_xxx` clips should not crash the app.

Legacy fallback:

```text
prompt_group = legacy
prompt_title = old prompt_id
transcript = expected transcript if available, otherwise old prompt_id
is_negative = false
```

## 11. Backend Validation

The frontend may send `prompt_group` and `transcript`, but the backend must derive canonical fields.

Canonical mapping:

```text
P1_vigil_only:
  transcript = Vigil
  prompt_title = VIGIL Only
  contains_vigil = true
  wake_intent = true
  is_negative = false

P2_phrase_plus_vigil:
  prompt_title = Phrase/Sentence + VIGIL
  transcript = participant provided
  contains_vigil = true
  wake_intent = true
  is_negative = false

P3_vigil_plus_phrase:
  prompt_title = VIGIL + Phrase/Sentence
  transcript = participant provided
  contains_vigil = true
  wake_intent = true
  is_negative = false

P4_negative:
  prompt_title = Negative Examples
  transcript = participant provided
  contains_vigil = false
  wake_intent = false
  is_negative = true
```

Transcript normalization:

- trim leading/trailing whitespace
- collapse multiple whitespace characters into one space
- normalize standalone `VIGIL` / `vigil` to `Vigil`

## 12. Quality Control

Online QC should remain lightweight:

- reject empty audio
- flag missing transcript
- flag invalid prompt group
- flag P2/P3 transcript missing exact word `Vigil`
- flag P4 transcript containing exact word `Vigil`
- flag missing audio file
- flag upload failure

Online QC should not require manual review for every clip.

Semantic QC is deferred to offline processing:

- download raw audio and metadata
- convert to WAV
- run Qwen ASR
- compare ASR output with intended transcript
- manually review failures
- decide whether participant re-recording is needed

## 13. Export Requirements

Export package should include:

```text
vigil_dataset_export/
  README.md
  metadata/
    accounts.csv
    sessions.jsonl
    clips.csv
    clips.jsonl
    qc_report.csv
  raw_audio/
  audio_raw/
  by_prompt_group/
    P1_vigil_only/
    P2_phrase_plus_vigil/
    P3_vigil_plus_phrase/
    P4_negative/
    legacy/
  qwen_asr/
    train.jsonl
    eval.jsonl
  keyword_spotting/
    kws_train.jsonl
    kws_eval.jsonl
```

Qwen ASR JSONL should use transcript text only:

```json
{"audio":"audio_raw/C000001.webm","text":"language English<asr_text>Vigil"}
```

Keyword spotting JSONL should include labels:

```json
{"audio":"audio_raw/C000001.webm","transcript":"Vigil","prompt_group":"P1_vigil_only","contains_vigil":true,"wake_intent":true,"is_negative":false}
```

For MVP, train/eval split may be deterministic. Prefer account-independent split when practical.

## 14. Security and Privacy Requirements

- Do not commit `.env` files.
- Do not commit audio files.
- Do not commit export ZIP files.
- Do not expose service role keys or Supabase secret keys in frontend code.
- Use HTTPS for production microphone access.
- Participants can only access their own data.
- Admin routes should be protected before external data collection at scale.
- Deleted data should be removed from active storage.
- Export snapshots should be handled carefully because they can preserve deleted data.

## 15. Cost and Infrastructure Considerations

Current MVP stack:

- Vercel: frontend hosting
- Render: FastAPI backend hosting
- Supabase: Postgres metadata and Storage audio files

Main cost drivers:

- raw audio storage volume
- audio upload/download bandwidth
- database size
- backend uptime
- export download frequency
- future GPU ASR review
- backup retention

AWS alternative:

```text
Vercel frontend -> AWS Amplify Hosting or S3 + CloudFront
Render backend -> ECS/Fargate, App Runner, Lambda, or Elastic Beanstalk
Supabase Postgres -> RDS PostgreSQL or Aurora PostgreSQL
Supabase Storage -> S3
Email code -> SES
Auth -> Cognito or custom email-code login
```

AWS can host the whole system under one vendor, but it increases configuration work around IAM, networking, monitoring, database operations, and cost control.

## 16. Success Criteria

The MVP is successful when:

- remote participants can open the Vercel link
- participants can log in with email code
- participants can record P1 without typing transcript
- participants can use P2 example chips such as `Hi VIGIL.`
- participants can use P3 example chips such as `VIGIL, go back.`
- participants can record P4 negative examples such as `visual`
- P4 rejects exact word `Vigil`
- participant uploads persist in Supabase Storage and Postgres
- admin can see positive and negative counts
- admin can play and delete clips
- admin can export raw audio and metadata
- deleted clips disappear from active metadata and storage

## 17. Open Questions For Professor

- How many participants are expected?
- How many sessions per participant are expected?
- How many total clips are needed?
- How long should raw audio be retained?
- Should raw audio be deleted after WAV conversion and final dataset generation?
- Does the project require IRB review or formal consent language?
- Who should have admin access?
- Is email-code login enough for pilot data collection?
- Should admin authentication use institutional login?
- What monthly budget is acceptable?
- Is AWS preferred because of lab or university credits?
- Should semantic ASR review happen offline only, or should participants be asked to re-record failed clips online?
- What false reject rate is acceptable for ASR review?
- What final dataset format does the downstream Qwen/wake-word training pipeline expect?

