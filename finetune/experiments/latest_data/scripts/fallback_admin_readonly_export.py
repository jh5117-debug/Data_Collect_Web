#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from vigil_latest.export_download import DEFAULT_BACKEND_URL, api_url, get_summary
from vigil_latest.utils import ensure_dir, sha256_file, write_json


ACCOUNT_FIELDS = ["email", "created_at_utc", "last_login_at_utc", "verified"]
PARTICIPANT_FIELDS = ["participant_id", "user_email", "english_native_speaker", "recording_device_type", "created_at_utc"]
SESSION_FIELDS = ["session_id", "participant_id", "batch_id", "status", "created_at_utc", "submitted_at_utc"]
CLIP_FIELDS = [
    "clip_id",
    "participant_id",
    "session_id",
    "prompt_id",
    "prompt_group",
    "prompt_title",
    "transcript",
    "normalized_transcript",
    "contains_vigil",
    "wake_intent",
    "is_negative",
    "clip_type",
    "raw_audio_path",
    "processed_wav_path",
    "duration_sec",
    "file_size_bytes",
    "sample_rate_processed",
    "channels_processed",
    "auto_qc_status",
    "auto_qc_flags",
    "segmentation_status",
    "detected_segment_count",
    "expected_segment_count",
    "created_at_utc",
    "review_status",
    "review_note",
]


def request_json(url: str, *, timeout: float = 180.0) -> Any:
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_bytes(url: str, *, timeout: float = 180.0, attempts: int = 3) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), dict(resp.headers.items())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"audio download failed after {attempts} attempts: {last_error}")


def rows_to_csv(rows: list[dict[str, Any]], fields: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buf.getvalue()


def rows_to_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=True, default=str) + "\n" for row in rows)


def clip_audio_suffix(clip: dict[str, Any]) -> str:
    raw = str(clip.get("raw_audio_path") or "")
    suffix = Path(raw).suffix
    return suffix if suffix else ".webm"


def fetch_all_metadata(base_url: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    clients = request_json(api_url(base_url, "/api/admin/clients"))
    accounts: list[dict[str, Any]] = []
    participants_by_id: dict[str, dict[str, Any]] = {}
    sessions_by_id: dict[str, dict[str, Any]] = {}
    clips_by_id: dict[str, dict[str, Any]] = {}
    for client in clients:
        email = str(client.get("email") or "")
        accounts.append(
            {
                "email": email,
                "created_at_utc": client.get("account_created_at_utc"),
                "last_login_at_utc": client.get("last_login_at_utc"),
                "verified": client.get("verified"),
            }
        )
        encoded_email = urllib.parse.quote(email, safe="")
        sessions = request_json(api_url(base_url, f"/api/admin/clients/{encoded_email}/sessions"))
        for session in sessions:
            session_id = str(session["session_id"])
            session_clips = request_json(api_url(base_url, f"/api/admin/sessions/{urllib.parse.quote(session_id, safe='')}/clips"))
            participant_id = str(session_clips[0]["participant_id"]) if session_clips else ""
            sessions_by_id[session_id] = {
                "session_id": session_id,
                "participant_id": participant_id,
                "batch_id": session.get("batch_id"),
                "status": session.get("status"),
                "created_at_utc": session.get("created_at_utc"),
                "submitted_at_utc": session.get("submitted_at_utc"),
            }
            if participant_id:
                participants_by_id.setdefault(
                    participant_id,
                    {
                        "participant_id": participant_id,
                        "user_email": email,
                        "english_native_speaker": "",
                        "recording_device_type": "",
                        "created_at_utc": session.get("created_at_utc"),
                    },
                )
            for clip in session_clips:
                clips_by_id[str(clip["clip_id"])] = dict(clip)
    return accounts, list(participants_by_id.values()), list(sessions_by_id.values()), list(clips_by_id.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--data-dir", default="finetune/data")
    parser.add_argument("--report-dir", default="finetune/experiments/latest_data/reports")
    args = parser.parse_args()
    reports = ensure_dir(args.report_dir)
    summary = get_summary(args.backend_url)
    accounts, participants, sessions, clips = fetch_all_metadata(args.backend_url)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    root = f"vigil_dataset_export_latest_readonly_{timestamp}"
    zip_path = ensure_dir(args.data_dir) / f"{root}.zip"
    warnings: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/README.md", "# VIGIL read-only Admin fallback export\n")
        archive.writestr(f"{root}/metadata/accounts.csv", rows_to_csv(accounts, ACCOUNT_FIELDS))
        archive.writestr(f"{root}/metadata/participants.csv", rows_to_csv(participants, PARTICIPANT_FIELDS))
        archive.writestr(f"{root}/metadata/sessions.jsonl", rows_to_jsonl(sorted(sessions, key=lambda r: str(r.get("session_id")))))
        normalized_clips: list[dict[str, Any]] = []
        for idx, clip in enumerate(sorted(clips, key=lambda r: str(r.get("clip_id"))), start=1):
            clip_id = str(clip["clip_id"])
            suffix = clip_audio_suffix(clip)
            audio_member = f"audio_raw/{clip_id}{suffix}"
            row = {field: clip.get(field) for field in CLIP_FIELDS}
            row["raw_audio_path"] = audio_member
            row["processed_wav_path"] = ""
            row["sample_rate_processed"] = ""
            row["channels_processed"] = ""
            row["review_status"] = ""
            row["review_note"] = ""
            normalized_clips.append(row)
            try:
                audio, headers = request_bytes(api_url(args.backend_url, f"/api/admin/clips/{urllib.parse.quote(clip_id, safe='')}/audio"))
                archive.writestr(f"{root}/{audio_member}", audio)
                if idx % 100 == 0:
                    print(json.dumps({"downloaded_audio": idx, "total_clips": len(clips)}, sort_keys=True), flush=True)
            except Exception as exc:
                warnings.append({"clip_id": clip_id, "reason": type(exc).__name__, "message": str(exc)[:500]})
        archive.writestr(f"{root}/metadata/clips.jsonl", rows_to_jsonl(normalized_clips))
        archive.writestr(f"{root}/metadata/export_warnings.jsonl", rows_to_jsonl(warnings))
    report = {
        "status": "ok" if not warnings and len(clips) == int(summary.get("total_clips", -1)) else "incomplete",
        "method": "readonly_admin_api_fallback",
        "summary": summary,
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": sha256_file(zip_path),
        "accounts": len(accounts),
        "participants": len(participants),
        "sessions": len(sessions),
        "clips": len(clips),
        "warnings": len(warnings),
    }
    write_json(reports / "readonly_fallback_export_report.json", report)
    print(zip_path.resolve())
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
