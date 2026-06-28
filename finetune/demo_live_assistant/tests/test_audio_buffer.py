from __future__ import annotations

from pathlib import Path

from audio_stream import SessionStore


def test_audio_session_buffer_saves_chunks(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "local_data")
    session = store.start("profile123")
    path = store.save_chunk(session, b"VIGIL", ".webm")
    assert path.exists()
    assert session.chunks == 1
