from __future__ import annotations

from librispeech import smoke_subset


def _rows() -> list[dict]:
    rows = []
    for speaker in range(12):
        for item in range(5):
            rows.append(
                {
                    "id": f"{speaker:04d}-0001-{item:04d}",
                    "speaker_id": f"{speaker:04d}",
                    "chapter_id": "0001",
                    "duration_sec": 1.0 + speaker * 0.1 + item,
                }
            )
    return rows


def test_smoke_subset_is_deterministic_and_spread() -> None:
    rows = _rows()
    first = smoke_subset(rows, count=32, seed=20260620)
    second = smoke_subset(rows, count=32, seed=20260620)
    assert first == second
    assert len(first) == 32
    assert len({row["speaker_id"] for row in first}) >= 10
    assert [row["id"] for row in first] != [row["id"] for row in rows[:32]]
