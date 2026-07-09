from __future__ import annotations

import csv
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ZIP = Path("finetune/experiments/false_trigger_regression/private/rosbag-trigger-word.zip")
DEFAULT_EXTRACT_DIR = Path("finetune/experiments/false_trigger_regression/private/rosbag-trigger-word_extracted")


def expected_label_for_bag_name(name: str) -> int:
    normalized = name.lower()
    if "false" in normalized or "negative" in normalized:
        return 0
    if "true" in normalized or "positive" in normalized:
        return 1
    raise ValueError(f"cannot infer expected label from bag name: {name}")


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> list[str]:
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            if not str(target).startswith(str(extract_dir.resolve())):
                raise ValueError(f"unsafe zip member path: {member.filename}")
            archive.extract(member, extract_dir)
            extracted.append(member.filename)
    return extracted


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _bag_info(metadata: dict[str, Any]) -> dict[str, Any]:
    return dict(metadata.get("rosbag2_bagfile_information") or {})


def metadata_topics(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    info = _bag_info(metadata)
    rows = []
    for item in info.get("topics_with_message_count", []) or []:
        topic = dict(item.get("topic_metadata") or {})
        rows.append(
            {
                "name": topic.get("name"),
                "type": topic.get("type"),
                "serialization_format": topic.get("serialization_format"),
                "message_count": int(item.get("message_count") or 0),
            }
        )
    return rows


def parse_metadata_summary(metadata_path: Path) -> dict[str, Any]:
    metadata = load_metadata(metadata_path)
    info = _bag_info(metadata)
    duration_ns = int((info.get("duration") or {}).get("nanoseconds") or 0)
    return {
        "metadata_path": str(metadata_path),
        "version": info.get("version"),
        "storage_identifier": info.get("storage_identifier"),
        "duration_nanoseconds": duration_ns,
        "duration_sec": duration_ns / 1e9,
        "starting_time_nanoseconds_since_epoch": (info.get("starting_time") or {}).get("nanoseconds_since_epoch"),
        "message_count": int(info.get("message_count") or 0),
        "relative_file_paths": list(info.get("relative_file_paths") or []),
        "files": list(info.get("files") or []),
        "topics": metadata_topics(metadata),
    }


def _query_with_python_sqlite(db_path: Path, sql: str) -> list[dict[str, Any]]:
    import sqlite3

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(sql).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def _query_with_sqlite_cli(db_path: Path, sql: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["sqlite3", "-csv", "-header", str(db_path), sql],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not completed.stdout.strip():
        return []
    reader = csv.DictReader(io.StringIO(completed.stdout))
    return [dict(row) for row in reader]


def query_sqlite(db_path: Path, sql: str) -> list[dict[str, Any]]:
    try:
        return _query_with_python_sqlite(Path(db_path), sql)
    except Exception:
        return _query_with_sqlite_cli(Path(db_path), sql)


def read_topics_table(db_path: Path) -> list[dict[str, Any]]:
    rows = query_sqlite(db_path, "select id, name, type, serialization_format from topics order by id")
    normalized = []
    for row in rows:
        normalized.append(
            {
                "id": int(row["id"]),
                "name": row["name"],
                "type": row["type"],
                "serialization_format": row["serialization_format"],
            }
        )
    return normalized


def read_message_counts(db_path: Path) -> dict[str, Any]:
    total_rows = query_sqlite(db_path, "select count(*) as n from messages")
    per_topic = query_sqlite(db_path, "select topic_id, count(*) as n from messages group by topic_id order by topic_id")
    return {
        "messages_table_rows": int(total_rows[0]["n"]) if total_rows else 0,
        "messages_by_topic_id": {str(int(row["topic_id"])): int(row["n"]) for row in per_topic},
    }


def inspect_bag_dir(bag_dir: Path) -> dict[str, Any]:
    bag_dir = Path(bag_dir)
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata.yaml under {bag_dir}")
    summary = parse_metadata_summary(metadata_path)
    db3_files = sorted(bag_dir.glob("*.db3"))
    db3_summaries = []
    for db_path in db3_files:
        db3_summaries.append(
            {
                "path": str(db_path),
                "file_name": db_path.name,
                "size_bytes": db_path.stat().st_size,
                "topics_table": read_topics_table(db_path),
                **read_message_counts(db_path),
            }
        )
    source_bag = bag_dir.name
    return {
        "source_bag": source_bag,
        "expected_label": expected_label_for_bag_name(source_bag),
        "bag_dir": str(bag_dir),
        "metadata": summary,
        "db3_files": db3_summaries,
    }


def find_bag_dirs(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(path.parent for path in root.rglob("metadata.yaml"))


def inspect_extracted_root(root: Path) -> dict[str, Any]:
    bag_dirs = find_bag_dirs(root)
    return {
        "root": str(root),
        "bag_count": len(bag_dirs),
        "bags": [inspect_bag_dir(path) for path in bag_dirs],
    }


def inspect_zip(zip_path: Path = DEFAULT_ZIP, extract_dir: Path = DEFAULT_EXTRACT_DIR) -> dict[str, Any]:
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extracted_members = safe_extract_zip(zip_path, extract_dir)
    inspection = inspect_extracted_root(extract_dir)
    inspection["zip_path"] = str(zip_path)
    inspection["zip_size_bytes"] = zip_path.stat().st_size
    inspection["extracted_members"] = extracted_members
    return inspection


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
