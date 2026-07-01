from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transcript import rolling_append
from trigger import TriggerState


@dataclass
class AssistantSession:
    profile_id: str
    session_id: str
    directory: Path
    trigger: TriggerState = field(default_factory=TriggerState)
    rolling_transcript: str = ""
    chunks: int = 0
    active: bool = True

    def add_transcript(self, text: str) -> None:
        self.rolling_transcript = rolling_append(self.rolling_transcript, text)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "assistant_session_id": self.session_id,
            "assistant_state": self.trigger.state,
            "rolling_transcript": self.rolling_transcript,
            "chunks": self.chunks,
            "active": self.active,
        }


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sessions: dict[str, AssistantSession] = {}

    def start(self, profile_id: str) -> AssistantSession:
        session_id = uuid.uuid4().hex
        directory = self.root / "profiles" / profile_id / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        session = AssistantSession(profile_id=profile_id, session_id=session_id, directory=directory)
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> AssistantSession:
        if session_id not in self.sessions:
            raise KeyError(session_id)
        return self.sessions[session_id]

    def stop(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        session.active = False
        session.trigger.state = "IDLE"
        return session.as_dict()

    def reset(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        session.trigger.reset()
        session.rolling_transcript = ""
        return session.as_dict()

    def save_chunk(self, session: AssistantSession, data: bytes, suffix: str = ".webm") -> Path:
        session.chunks += 1
        if suffix.lower() not in {".webm", ".wav", ".m4a", ".ogg"}:
            suffix = ".webm"
        path = session.directory / f"chunk_{session.chunks:04d}{suffix}"
        path.write_bytes(data)
        (session.directory / "session.json").write_text(json.dumps(session.as_dict(), indent=2) + "\n", encoding="utf-8")
        return path
