from __future__ import annotations

import subprocess
from pathlib import Path

from audio_extract import _extract_printable_cdr_strings, _find_int16_payload
from rosbag_index import expected_label_for_bag_name, parse_metadata_summary, read_message_counts, read_topics_table


def test_expected_label_assignment() -> None:
    assert expected_label_for_bag_name("false_positive") == 0
    assert expected_label_for_bag_name("true_positive") == 1


def test_metadata_parser(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.yaml"
    metadata.write_text(
        """
rosbag2_bagfile_information:
  version: 5
  storage_identifier: sqlite3
  duration:
    nanoseconds: 2000000000
  message_count: 2
  topics_with_message_count:
    - topic_metadata:
        name: /microphone/audio
        type: vigil_msgs/msg/AudioMessage
        serialization_format: cdr
      message_count: 2
  relative_file_paths:
    - sample.db3
""",
        encoding="utf-8",
    )
    summary = parse_metadata_summary(metadata)
    assert summary["duration_sec"] == 2.0
    assert summary["message_count"] == 2
    assert summary["topics"][0]["name"] == "/microphone/audio"


def test_sqlite_topic_reader_with_cli_created_db(tmp_path: Path) -> None:
    db = tmp_path / "bag.db3"
    subprocess.run(
        [
            "sqlite3",
            str(db),
            "create table topics(id integer primary key, name text, type text, serialization_format text);"
            "create table messages(id integer primary key, topic_id integer, timestamp integer, data blob);"
            "insert into topics values (1, '/microphone/audio', 'vigil_msgs/msg/AudioMessage', 'cdr');"
            "insert into messages(topic_id, timestamp, data) values (1, 10, x'00');"
            "insert into messages(topic_id, timestamp, data) values (1, 20, x'01');",
        ],
        check=True,
    )
    assert read_topics_table(db)[0]["name"] == "/microphone/audio"
    assert read_message_counts(db)["messages_by_topic_id"] == {"1": 2}


def test_known_audio_cdr_payload_decoder() -> None:
    samples = (1).to_bytes(2, "little", signed=True) + (-2).to_bytes(2, "little", signed=True)
    blob = b"\x00\x01\x00\x00" + b"header-padding" + (16000).to_bytes(4, "little") + (2).to_bytes(4, "little") + samples
    payload, sample_rate = _find_int16_payload(blob)
    assert sample_rate == 16000
    assert payload == samples


def test_known_transcript_cdr_string_decoder() -> None:
    blob = b"\x00\x01\x00\x00" + (5).to_bytes(4, "little") + b"Joe.\0"
    assert _extract_printable_cdr_strings(blob) == ["Joe."]
