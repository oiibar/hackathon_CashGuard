from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


def _default_models() -> str:
    models = BACKEND_DIR / "models"
    paths = [models / "yolov8n.pt"]
    paths += [p for p in (models / "cash_world.pt", models / "cash_custom.pt") if p.exists()]
    return ",".join(str(p) for p in paths)


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("API_KEY", "change-me")

    database_url: str = os.getenv("DATABASE_URL", "")
    data_dir: Path = Path(os.getenv("DATA_DIR", BACKEND_DIR / "data"))
    keep_videos: bool = _env_bool("KEEP_VIDEOS", "false")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "2048"))

    model_paths: list[str] = field(default_factory=lambda: _env_list("MODEL_PATHS", _default_models()))
    device: str = os.getenv("DEVICE", "")
    imgsz: int = int(os.getenv("IMGSZ", "640"))
    conf: float = float(os.getenv("CONF", "0.35"))

    workers: int = int(os.getenv("WORKERS", "1"))
    job_deadline_sec: float = float(os.getenv("JOB_DEADLINE_SEC", "270"))
    target_fps: float = float(os.getenv("TARGET_FPS", "4"))
    min_fps: float = float(os.getenv("MIN_FPS", "0.5"))
    frame_width: int = int(os.getenv("FRAME_WIDTH", "1280"))

    rules_path: Path = Path(os.getenv("RULES_PATH", BACKEND_DIR / "config" / "rules.yaml"))
    zones_path: Path = Path(os.getenv("ZONES_PATH", BACKEND_DIR / "config" / "zones.yaml"))
    default_zone_profile: str = os.getenv("ZONE_PROFILE", "auto")

    evidence_max_width: int = int(os.getenv("EVIDENCE_MAX_WIDTH", "960"))
    evidence_jpeg_quality: int = int(os.getenv("EVIDENCE_JPEG_QUALITY", "80"))
    max_violations_per_type: int = int(os.getenv("MAX_VIOLATIONS_PER_TYPE", "10"))
    store_jpeg_quality: int = int(os.getenv("STORE_JPEG_QUALITY", "85"))
    retention_days: float = float(os.getenv("RETENTION_DAYS", "90"))

    learning_enabled: bool = _env_bool("LEARNING_ENABLED", "true")
    auto_retrain: bool = _env_bool("AUTO_RETRAIN", "true")
    retrain_min_reviews: int = int(os.getenv("RETRAIN_MIN_REVIEWS", "3"))
    cv_folds: int = int(os.getenv("CV_FOLDS", "10"))
    threshold_policy: str = os.getenv("THRESHOLD_POLICY", "mcc")

    frontend_dist: Path = Path(os.getenv("FRONTEND_DIST", BACKEND_DIR.parent / "frontend" / "dist"))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()
