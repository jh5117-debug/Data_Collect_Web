from __future__ import annotations

from pydantic import BaseModel, Field


class ProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ProfileOut(BaseModel):
    profile_id: str
    display_name: str


class CalibrationRequest(BaseModel):
    profile_id: str


class AssistantStartRequest(BaseModel):
    profile_id: str


class AssistantSessionRequest(BaseModel):
    profile_id: str
    assistant_session_id: str


class HealthOut(BaseModel):
    status: str
    mode: str
    model_loaded: dict[str, bool]
    gpu: dict[str, str | int | bool | None]
    message: str
