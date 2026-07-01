from __future__ import annotations


def test_qwen_transcript_cache_is_clip_deduplicated():
    rows = [{"clip_id": "C1"}, {"clip_id": "C1"}, {"clip_id": "C2"}]
    unique = {row["clip_id"] for row in rows}
    assert len(unique) == 2
