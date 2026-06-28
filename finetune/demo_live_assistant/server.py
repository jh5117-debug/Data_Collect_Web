from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from audio_stream import SessionStore
from model_runtime import AssistantModelRuntime, DEFAULT_RUN_DIR
from onboarding import ProfileStore
from schemas import AssistantSessionRequest, AssistantStartRequest, CalibrationRequest, HealthOut, ProfileCreate


BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
DEFAULT_DATA = BASE / "local_data"


def create_app(
    *,
    data_root: Path | str = DEFAULT_DATA,
    run_dir: Path | str = DEFAULT_RUN_DIR,
    force_mock: bool = False,
    load_models: bool = True,
) -> FastAPI:
    app = FastAPI(title="VIGIL Local HAL Assistant Demo")
    data_root = Path(data_root)
    profiles = ProfileStore(data_root)
    sessions = SessionStore(data_root)
    runtime = AssistantModelRuntime(run_dir, force_mock=force_mock)
    if load_models:
        runtime.load()
    app.state.profiles = profiles
    app.state.sessions = sessions
    app.state.runtime = runtime
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/health", response_model=HealthOut)
    def health() -> dict[str, Any]:
        status = runtime.status()
        return {
            "status": "ok",
            "mode": status.mode,
            "model_loaded": status.model_loaded,
            "gpu": status.gpu,
            "message": status.message,
        }

    @app.post("/api/profile")
    def create_profile(payload: ProfileCreate) -> dict[str, Any]:
        return profiles.create_profile(payload.name)

    @app.post("/api/onboarding/clip")
    async def upload_clip(
        profile_id: str = Form(...),
        prompt_group: str = Form("P1_vigil_only"),
        transcript: str = Form(""),
        is_positive: bool = Form(True),
        accepted: bool = Form(True),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        try:
            data = await file.read()
            return profiles.save_clip(
                profile_id,
                file.filename or "clip.webm",
                data,
                {
                    "prompt_group": prompt_group,
                    "transcript": transcript,
                    "is_positive": is_positive,
                    "accepted": accepted,
                },
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="profile not found") from exc

    @app.get("/api/onboarding/clip/{clip_id}/audio")
    def clip_audio(clip_id: str, profile_id: str = Query(...)) -> FileResponse:
        try:
            _, audio_path = profiles.clip_paths(profile_id, clip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="clip not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FileResponse(audio_path)

    @app.delete("/api/onboarding/clip/{clip_id}")
    def delete_clip(clip_id: str, profile_id: str = Query(...)) -> dict[str, Any]:
        try:
            return profiles.delete_clip(profile_id, clip_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="clip not found") from exc

    @app.post("/api/onboarding/calibrate")
    def calibrate(payload: CalibrationRequest) -> dict[str, Any]:
        try:
            clips = profiles.accepted_positive_clips(payload.profile_id)
            scores = runtime.support_scores(len(clips))
            return profiles.calibrate(payload.profile_id, scores, runtime.theta2)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="profile not found") from exc

    @app.post("/api/assistant/start")
    def assistant_start(payload: AssistantStartRequest) -> dict[str, Any]:
        try:
            profiles.require_profile(payload.profile_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="profile not found") from exc
        session = sessions.start(payload.profile_id)
        session.trigger.state = "LISTENING"
        return session.as_dict()

    @app.post("/api/assistant/chunk")
    async def assistant_chunk(
        profile_id: str = Form(...),
        assistant_session_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        try:
            profiles.require_profile(profile_id)
            session = sessions.get(assistant_session_id)
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail="session or profile not found") from exc
        if session.profile_id != profile_id:
            raise HTTPException(status_code=400, detail="session/profile mismatch")
        data = await file.read()
        chunk_path = sessions.save_chunk(session, data, suffix=Path(file.filename or "chunk.webm").suffix)
        calibration = profiles.calibration(profile_id)
        result = runtime.analyze_audio(chunk_path, calibration)
        session.add_transcript(result.get("rolling_transcript") or "")
        trigger_update = session.trigger.update(candidate=bool(result["candidate"]), trigger_detected=bool(result["trigger_detected"]))
        return {
            "rolling_transcript": session.rolling_transcript,
            "stage1_score": result["stage1_score"],
            "stage2_score": result["stage2_score"],
            "calibrated_stage2_score": result["calibrated_stage2_score"],
            "theta_1": result["theta_1"],
            "theta_2": result["theta_2"],
            "trigger_detected": trigger_update["trigger_accepted"],
            "assistant_state": trigger_update["assistant_state"],
            "cooldown_active": trigger_update["cooldown_active"],
            "winning_window": result["winning_window"],
            "latency_ms": result["latency_ms"],
            "debug": {
                **result["debug"],
                "runtime_mode": runtime.mode,
                "qwen_extra_encoder_forward": bool(result["debug"].get("stage2_qwen_feature_path_used")),
                "llm_response_implemented": False,
            },
        }

    @app.post("/api/assistant/stop")
    def assistant_stop(payload: AssistantSessionRequest) -> dict[str, Any]:
        try:
            return sessions.stop(payload.assistant_session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @app.post("/api/assistant/reset")
    def assistant_reset(payload: AssistantSessionRequest) -> dict[str, Any]:
        try:
            return sessions.reset(payload.assistant_session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @app.get("/api/assistant/session/{session_id}")
    def assistant_session(session_id: str) -> dict[str, Any]:
        try:
            return sessions.get(session_id).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    return app


app = create_app(load_models=False, force_mock=True)
