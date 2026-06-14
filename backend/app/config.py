import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://data-collect-web.onrender.com",
    "https://data-collect-web.vercel.app",
]


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default.resolve()

    path = Path(value)
    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == "backend":
        return (REPO_DIR / path).resolve()

    return (BACKEND_DIR / path).resolve()


class Settings:
    app_name: str = "Vigil Recorder"
    prompt_version: str = "v0.1"
    default_batch_id: str = "vigil_batch_v0_1"
    storage_backend: str = os.getenv("STORAGE_BACKEND", "local")
    local_storage_root: Path = _resolve_path(
        os.getenv("LOCAL_STORAGE_ROOT"), BACKEND_DIR / "storage"
    )
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{local_storage_root / 'vigil_recorder.sqlite3'}"
    )
    cors_origins: list[str] = list(
        dict.fromkeys(
            DEFAULT_CORS_ORIGINS
            + [
                origin.strip()
                for origin in os.getenv("CORS_ORIGINS", "").split(",")
                if origin.strip()
            ]
        )
    )

    s3_endpoint_url: str | None = os.getenv("S3_ENDPOINT_URL")
    s3_bucket_name: str | None = os.getenv("S3_BUCKET_NAME")
    s3_access_key_id: str | None = os.getenv("S3_ACCESS_KEY_ID")
    s3_secret_access_key: str | None = os.getenv("S3_SECRET_ACCESS_KEY")

    supabase_url: str | None = os.getenv("SUPABASE_URL")
    supabase_secret_key: str | None = os.getenv(
        "SUPABASE_SECRET_KEY", os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
    supabase_storage_bucket: str = os.getenv("SUPABASE_STORAGE_BUCKET", "vigil-audio")


settings = Settings()
settings.local_storage_root.mkdir(parents=True, exist_ok=True)
