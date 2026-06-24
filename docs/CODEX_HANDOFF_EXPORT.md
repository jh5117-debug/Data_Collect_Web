# Codex Handoff: Admin Export

## 2026-06-24

- Replaced the synchronous Admin export request with a database-backed export job.
- `POST /api/admin/export` now returns a `job_id` immediately.
- A bounded in-process worker generates the ZIP and persists progress in `export_jobs`.
- The frontend polls `/api/admin/export/jobs/{job_id}` and shows phase, item count, percent, warnings, size, and a download button when complete.
- Export ZIPs now keep one canonical raw audio copy under `audio_raw/`; prompt-group folders contain lightweight `clips.jsonl` views instead of duplicate audio.
- Supabase export downloads no longer do `HEAD` before `GET`; missing files are tracked as export warnings.

Verification completed in this turn:

- `LD_PRELOAD=/home/hj/miniconda/pkgs/libsqlite-3.53.2-h0c1763c_0/lib/libsqlite3.so.0 PYTHONPATH=. pytest -q` from `backend/`
- `npm run build` from `frontend/`
- Local FastAPI export smoke with temp SQLite/temp storage, one uploaded test clip, async job polling, ZIP download, and ZIP structure inspection

Note: the base conda Python currently loads an older `libsqlite3.so.0` symlink and needs the `LD_PRELOAD` above for sqlite-backed tests.
