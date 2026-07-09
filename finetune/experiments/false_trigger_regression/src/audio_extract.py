from __future__ import annotations

import json
import re
import struct
import time
import wave
from pathlib import Path
from typing import Any

from rosbag_index import find_bag_dirs, read_topics_table


def ros2_decode_status() -> dict[str, Any]:
    missing = []
    for module in ("rosbag2_py", "rosidl_runtime_py", "rclpy", "vigil_msgs"):
        try:
            __import__(module)
        except Exception as exc:
            missing.append({"module": module, "error": f"{type(exc).__name__}: {exc}"})
    return {"available": not missing, "missing": missing}


def exact_ros2_command(repo_root: Path | str = "/home/hj/Data_Collect_Web") -> str:
    root = Path(repo_root)
    return (
        f"cd {root} && "
        "PYTHONPATH=finetune/experiments/false_trigger_regression/src:. "
        "python finetune/experiments/false_trigger_regression/scripts/extract_rosbag_audio.py "
        "--ros2-decode"
    )


def _query_messages(db_path: Path, topic_id: int) -> list[tuple[int, int, bytes]]:
    import sqlite3

    connection = sqlite3.connect(str(db_path))
    try:
        return [
            (int(row[0]), int(row[1]), bytes(row[2]))
            for row in connection.execute(
                "select id, timestamp, data from messages where topic_id = ? order by timestamp, id",
                (topic_id,),
            )
        ]
    finally:
        connection.close()


def _topic_id_by_name(db_path: Path) -> dict[str, int]:
    return {row["name"]: int(row["id"]) for row in read_topics_table(db_path)}


def _find_int16_payload(blob: bytes, sample_rate: int = 16000) -> tuple[bytes, int]:
    rate_bytes = int(sample_rate).to_bytes(4, "little", signed=False)
    for rate_offset in range(0, min(len(blob), 128) - 4):
        if blob[rate_offset : rate_offset + 4] != rate_bytes:
            continue
        for length_offset in range(rate_offset + 4, min(len(blob), rate_offset + 24) - 4):
            sample_count = int.from_bytes(blob[length_offset : length_offset + 4], "little", signed=False)
            data_offset = length_offset + 4
            if sample_count <= 0:
                continue
            if data_offset + sample_count * 2 == len(blob):
                return blob[data_offset:], sample_rate
    raise ValueError("could not locate 16 kHz int16 PCM payload")


def _extract_printable_cdr_strings(blob: bytes) -> list[str]:
    values: list[str] = []
    for offset in range(4, max(4, len(blob) - 4)):
        length = int.from_bytes(blob[offset : offset + 4], "little", signed=False)
        if not 1 <= length <= 256:
            continue
        end = offset + 4 + length
        if end > len(blob):
            continue
        raw = blob[offset + 4 : end]
        if not raw.endswith(b"\0"):
            continue
        payload = raw[:-1]
        if payload and all(32 <= byte < 127 for byte in payload):
            values.append(payload.decode("utf-8", errors="replace"))
    return values


def _transcript_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown"


def _expected_label_for_bag(source_bag: str) -> int:
    return 0 if "false" in source_bag.lower() or "negative" in source_bag.lower() else 1


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def _slice_pcm_by_time(
    pcm: bytes,
    *,
    first_audio_timestamp_ns: int,
    event_timestamp_ns: int,
    sample_rate: int,
    window_sec: float = 2.0,
) -> tuple[bytes, float, float]:
    total_samples = len(pcm) // 2
    event_sample = round(((event_timestamp_ns - first_audio_timestamp_ns) / 1e9) * sample_rate)
    event_sample = max(0, min(total_samples, event_sample))
    window_samples = int(round(window_sec * sample_rate))
    start_sample = max(0, event_sample - window_samples)
    end_sample = min(total_samples, max(event_sample, start_sample + min(window_samples, total_samples - start_sample)))
    return pcm[start_sample * 2 : end_sample * 2], start_sample / sample_rate, end_sample / sample_rate


def _extract_bag_audio_cases(bag_dir: Path, run_dir: Path) -> dict[str, Any]:
    source_bag = bag_dir.name
    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        raise FileNotFoundError(f"missing db3 under {bag_dir}")
    db_path = db3_files[0]
    topics = _topic_id_by_name(db_path)
    audio_topic_id = topics.get("/microphone/audio")
    transcript_topic_id = topics.get("/microphone/audio/transcription")
    if audio_topic_id is None:
        raise ValueError(f"missing /microphone/audio in {db_path}")
    audio_rows = _query_messages(db_path, audio_topic_id)
    if not audio_rows:
        raise ValueError(f"no audio messages in {db_path}")
    sample_rate = 16000
    pcm_chunks = []
    for _, _, blob in audio_rows:
        chunk, sample_rate = _find_int16_payload(blob)
        pcm_chunks.append(chunk)
    full_pcm = b"".join(pcm_chunks)
    audio_dir = run_dir / "extracted_audio"
    full_wav = audio_dir / f"{source_bag}_full.wav"
    _write_wav(full_wav, full_pcm, sample_rate)
    transcript_rows = _query_messages(db_path, transcript_topic_id) if transcript_topic_id is not None else []
    cases = []
    for index, (_, timestamp_ns, blob) in enumerate(transcript_rows, start=1):
        strings = _extract_printable_cdr_strings(blob)
        transcript = strings[-1] if strings else ""
        case_id = f"{source_bag}_{index:03d}_{_transcript_slug(transcript)}"
        window_pcm, start_sec, end_sec = _slice_pcm_by_time(
            full_pcm,
            first_audio_timestamp_ns=audio_rows[0][1],
            event_timestamp_ns=timestamp_ns,
            sample_rate=sample_rate,
        )
        window_wav = audio_dir / f"{case_id}.wav"
        _write_wav(window_wav, window_pcm, sample_rate)
        cases.append(
            {
                "case_id": case_id,
                "source_bag": source_bag,
                "expected_label": _expected_label_for_bag(source_bag),
                "transcript_hint": transcript,
                "wav_path": str(window_wav),
                "duration_sec": round(len(window_pcm) / 2 / sample_rate, 6),
                "window_start_sec": round(start_sec, 6),
                "window_end_sec": round(end_sec, 6),
                "full_wav_path": str(full_wav),
                "sample_rate": sample_rate,
                "audio_messages": len(audio_rows),
                "transcript_message_timestamp_ns": timestamp_ns,
            }
        )
    if not cases:
        cases.append(
            {
                "case_id": f"{source_bag}_full",
                "source_bag": source_bag,
                "expected_label": _expected_label_for_bag(source_bag),
                "transcript_hint": "",
                "wav_path": str(full_wav),
                "duration_sec": round(len(full_pcm) / 2 / sample_rate, 6),
                "window_start_sec": 0.0,
                "window_end_sec": round(len(full_pcm) / 2 / sample_rate, 6),
                "full_wav_path": str(full_wav),
                "sample_rate": sample_rate,
                "audio_messages": len(audio_rows),
                "transcript_message_timestamp_ns": None,
            }
        )
    return {
        "source_bag": source_bag,
        "db_path": str(db_path),
        "sample_rate": sample_rate,
        "full_wav_path": str(full_wav),
        "duration_sec": round(len(full_pcm) / 2 / sample_rate, 6),
        "audio_messages": len(audio_rows),
        "transcript_messages": len(transcript_rows),
        "cases": cases,
    }


def _write_manifest(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cases), encoding="utf-8")


def extract_known_vigil_rosbag_audio(extracted_root: Path, output_root: Path) -> dict[str, Any]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    bag_results = [_extract_bag_audio_cases(path, run_dir) for path in find_bag_dirs(extracted_root)]
    cases = [case for bag in bag_results for case in bag["cases"]]
    manifest_path = run_dir / "rosbag_cases.jsonl"
    _write_manifest(manifest_path, cases)
    return {
        "status": "ok",
        "reason": "Decoded known VIGIL rosbag CDR layout into local WAV windows without ROS 2 message imports.",
        "ros2_decode_status": ros2_decode_status(),
        "extracted_root": str(extracted_root),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "manifest_created": True,
        "case_count": len(cases),
        "bag_results": bag_results,
        "exact_resume_command": (
            "cd /home/hj/Data_Collect_Web && "
            f"PYTHONPATH=finetune/src:finetune/experiments/false_trigger_regression/src:. "
            "python finetune/experiments/false_trigger_regression/scripts/run_false_trigger_score_audit.py "
            f"--manifest {manifest_path}"
        ),
    }


def attempt_audio_extraction(
    extracted_root: Path,
    output_root: Path,
    *,
    repo_root: Path | str = "/home/hj/Data_Collect_Web",
    force_ros2_decode: bool = False,
) -> dict[str, Any]:
    status = ros2_decode_status()
    try:
        return extract_known_vigil_rosbag_audio(Path(extracted_root), Path(output_root))
    except Exception as exc:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_root) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "rosbag_cases.jsonl"
        known_decoder_error = f"{type(exc).__name__}: {exc}"
    if not status["available"]:
        result = {
            "status": "blocked",
            "reason": "Known-layout extraction failed and ROS 2 decoding is not available in this Python environment.",
            "known_decoder_error": known_decoder_error,
            "ros2_decode_status": status,
            "extracted_root": str(extracted_root),
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "manifest_created": False,
            "exact_resume_command": exact_ros2_command(repo_root),
        }
        return result
    if not force_ros2_decode:
        return {
            "status": "blocked",
            "reason": "ROS 2 modules are importable, but message decoding is intentionally gated behind --ros2-decode.",
            "ros2_decode_status": status,
            "extracted_root": str(extracted_root),
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "manifest_created": False,
            "exact_resume_command": exact_ros2_command(repo_root),
        }
    return {
        "status": "blocked",
        "reason": "ROS 2 modules are available, but typed vigil_msgs CDR audio reconstruction is not implemented in this non-ROS audit package.",
        "ros2_decode_status": status,
        "extracted_root": str(extracted_root),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "manifest_created": False,
        "exact_resume_command": exact_ros2_command(repo_root),
    }


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
