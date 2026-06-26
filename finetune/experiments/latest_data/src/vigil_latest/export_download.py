from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ensure_dir, sha256_file, write_json


DEFAULT_BACKEND_URL = "https://data-collect-web.onrender.com"


@dataclass(frozen=True)
class DownloadResult:
    zip_path: Path
    sha256: str
    content_length: int | None
    job: dict[str, Any]


def api_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def request_json(method: str, url: str, *, timeout: float = 120.0) -> dict[str, Any]:
    req = urllib.request.Request(url, method=method)
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {body[:500]}") from exc


def get_summary(base_url: str = DEFAULT_BACKEND_URL) -> dict[str, Any]:
    return request_json("GET", api_url(base_url, "/api/admin/summary"))


def create_export_job(base_url: str = DEFAULT_BACKEND_URL) -> dict[str, Any]:
    return request_json("POST", api_url(base_url, "/api/admin/export"))


def poll_export_job(base_url: str, job_id: str, *, interval_sec: float = 5.0, timeout_sec: float = 3600.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = request_json("GET", api_url(base_url, f"/api/admin/export/jobs/{job_id}"))
        status = str(last.get("status"))
        if status in {"completed", "failed"}:
            return last
        time.sleep(interval_sec)
    raise TimeoutError(f"export job {job_id} did not finish before timeout; last status={last}")


def download_export(base_url: str, job: dict[str, Any], output_dir: Path | str, *, output_name: str | None = None) -> DownloadResult:
    if job.get("status") != "completed":
        raise ValueError(f"export job is not completed: {job.get('status')}")
    download_path = job.get("download_path")
    if not download_path:
        raise ValueError("completed export job did not include download_path")
    file_name = output_name or str(job.get("file_name") or "vigil_dataset_export_latest.zip")
    out_dir = ensure_dir(output_dir)
    zip_path = out_dir / file_name
    req = urllib.request.Request(api_url(base_url, str(download_path)), method="GET")
    with urllib.request.urlopen(req, timeout=1800.0) as resp:
        content_length = resp.headers.get("Content-Length")
        with zip_path.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    digest = sha256_file(zip_path)
    return DownloadResult(
        zip_path=zip_path,
        sha256=digest,
        content_length=int(content_length) if content_length and content_length.isdigit() else None,
        job=job,
    )


def write_download_report(path: Path | str, summary: dict[str, Any], job: dict[str, Any], result: DownloadResult | None) -> None:
    payload = {
        "summary": summary,
        "job": job,
        "download": None
        if result is None
        else {
            "zip_path": str(result.zip_path),
            "sha256": result.sha256,
            "content_length": result.content_length,
            "file_size_bytes": result.zip_path.stat().st_size,
        },
    }
    write_json(path, payload)
