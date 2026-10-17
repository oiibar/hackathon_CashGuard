"""Runtime settings, read once from environment variables.

Everything runs locally: no setting here points to a cloud service.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # Auth: jury sends this value in the X-API-Key header.
    api_key: str = os.getenv("API_KEY", "change-me")

    # Storage
    data_dir: Path = Path(os.getenv("DATA_DIR", BACKEND_DIR / "data"))
    keep_videos: bool = os.getenv("KEEP_VIDEOS", "false").lower() == "true"
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "2048"))

    # Models: comma-separated list of local .pt files. Detections from all
    # models are merged by class name, so a custom model (cash, door_open...)
    # can run next to the stock COCO model (person, cell phone, handbag...).
    model_paths: list[str] = field(
        default_factory=lambda: _env_list("MODEL_PATHS", str(BACKEND_DIR / "models" / "yolov8n.pt"))
    )
    device: str = os.getenv("DEVICE", "")  # "" = auto, "cpu", "0" (first GPU), "mps"
    imgsz: int = int(os.getenv("IMGSZ", "640"))
    conf: float = float(os.getenv("CONF", "0.35"))

    # Throughput / time budget
    workers: int = int(os.getenv("WORKERS", "1"))
    # Hard limit from the hackathon: 300 s per video. The budget counts from
    # upload (queue wait included) and keeps a safety margin.
    job_deadline_sec: float = float(os.getenv("JOB_DEADLINE_SEC", "270"))
    target_fps: float = float(os.getenv("TARGET_FPS", "4"))
    min_fps: float = float(os.getenv("MIN_FPS", "0.5"))
    frame_width: int = int(os.getenv("FRAME_WIDTH", "1280"))

    # Rules / zones
    rules_path: Path = Path(os.getenv("RULES_PATH", BACKEND_DIR / "config" / "rules.yaml"))
    zones_path: Path = Path(os.getenv("ZONES_PATH", BACKEND_DIR / "config" / "zones.yaml"))
    default_zone_profile: str = os.getenv("ZONE_PROFILE", "default")

    # Evidence frames
    evidence_max_width: int = int(os.getenv("EVIDENCE_MAX_WIDTH", "960"))
    evidence_jpeg_quality: int = int(os.getenv("EVIDENCE_JPEG_QUALITY", "80"))
    max_violations_per_type: int = int(os.getenv("MAX_VIOLATIONS_PER_TYPE", "10"))

    # Frontend build served by FastAPI (optional)
    frontend_dist: Path = Path(os.getenv("FRONTEND_DIST", BACKEND_DIR.parent / "frontend" / "dist"))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


settings = Settings()
