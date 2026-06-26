#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from vigil_latest.export_download import (
    DEFAULT_BACKEND_URL,
    create_export_job,
    download_export,
    get_summary,
    poll_export_job,
    write_download_report,
)
from vigil_latest.utils import ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--data-dir", default="finetune/data")
    parser.add_argument("--report-dir", default="finetune/experiments/latest_data/reports")
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    args = parser.parse_args()
    reports = ensure_dir(args.report_dir)
    summary = get_summary(args.backend_url)
    write_json(reports / "admin_summary_before_export.json", summary)
    job = create_export_job(args.backend_url)
    write_json(reports / "export_job_create.json", job)
    final_job = poll_export_job(
        args.backend_url,
        str(job["job_id"]),
        interval_sec=args.poll_interval_sec,
        timeout_sec=args.timeout_sec,
    )
    write_json(reports / f"export_job_status_{job['job_id']}.json", final_job)
    if final_job.get("status") != "completed":
        write_download_report(reports / "latest_export_download_report.json", summary, final_job, None)
        return 2
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    output_name = f"vigil_dataset_export_latest_{timestamp}.zip"
    result = download_export(args.backend_url, final_job, Path(args.data_dir), output_name=output_name)
    write_download_report(reports / "latest_export_download_report.json", summary, final_job, result)
    print(result.zip_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
