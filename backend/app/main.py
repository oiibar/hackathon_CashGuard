"""FastAPI entry point.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Routes (all /api/v1 routes except /health need the X-API-Key header):
    POST /api/v1/analyze        multipart file=<video.mov> -> {job_id}
    GET  /api/v1/jobs/{job_id}  -> {status: processing|done|failed, verdict, violations}
    GET  /api/v1/config         zone profiles + rule list (used by the web UI)
    GET  /api/v1/health         liveness, no auth
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .auth import require_api_key
from .config import settings
from .jobs import manager
from .pipeline.rules import load_yaml, wanted_classes

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cashguard")

ALLOWED_EXT = {e.strip().lower() for e in os.getenv("ALLOWED_EXT", ".mov,.mp4,.m4v,.avi,.mkv").split(",")}

state: dict = {}


def _build_analyzers(rules_cfg: dict) -> list:
    # Imported here so the API module loads fast and tests can stub it.
    from .pipeline.analyzer import Analyzer
    from .pipeline.detector import Detector

    analyzers = []
    for _ in range(max(settings.workers, 1)):
        det = Detector(settings.model_paths, settings.device, settings.imgsz, settings.conf, wanted_classes(rules_cfg))
        analyzers.append(Analyzer(det, rules_cfg, settings))
    return analyzers


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.api_key == "change-me":
        log.warning("API_KEY is the default 'change-me'. Set API_KEY env var before the demo.")
    rules_cfg = load_yaml(settings.rules_path)
    zones_cfg = load_yaml(settings.zones_path)
    state["rules"] = rules_cfg
    state["zones"] = zones_cfg.get("profiles", {})
    builder = app.state.analyzer_builder if hasattr(app.state, "analyzer_builder") else _build_analyzers
    analyzers = await asyncio.to_thread(builder, rules_cfg)
    state["detector_fps"] = [round(1 / a.sec_per_frame, 1) for a in analyzers if getattr(a, "sec_per_frame", 0)]
    await manager.start(analyzers)
    log.info("ready: %d worker(s), detector fps %s", len(analyzers), state["detector_fps"])
    yield
    await manager.stop()


app = FastAPI(title="CashGuard: cash discipline video analytics", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


def _resolve_zones(zones_json: str | None, profile: str | None) -> dict:
    if zones_json:
        try:
            zones = json.loads(zones_json)
            assert isinstance(zones, dict)
            for name, pts in zones.items():
                assert isinstance(pts, list) and len(pts) >= 3, name
                assert all(len(p) == 2 and 0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in pts), name
            return zones
        except Exception as exc:
            raise HTTPException(422,
                                f"zones must be JSON {{name: [[x,y],...]}} with 0..1 coords: {exc}")
    name = profile or settings.default_zone_profile
    if name not in state["zones"]:
        raise HTTPException(422, f"unknown zone profile: {name}")
    return state["zones"][name]


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "workers": len(manager.analyzers),
        "queued": manager.queue.qsize(),
        "detector_fps": state.get("detector_fps"),
    }


@app.post("/api/v1/analyze", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
async def analyze(
    file: UploadFile = File(..., description="Video file, .mov"),
    zones: str | None = Form(None, description='Optional JSON: {"cash_zone": [[x,y],...], "client_zone": [...]}'),
    profile: str | None = Form(None, description="Optional zone profile name from config/zones.yaml"),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"unsupported file type '{ext}', expected one of {sorted(ALLOWED_EXT)}")
    zone_map = _resolve_zones(zones, profile)

    job_id = uuid.uuid4().hex
    dest = settings.uploads_dir / f"{job_id}{ext}"
    limit = settings.max_upload_mb * 1024 * 1024

    def _save() -> int:
        size = 0
        with open(dest, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    break
                out.write(chunk)
        return size

    size = await asyncio.to_thread(_save)
    if size > limit:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"file larger than {settings.max_upload_mb} MB")
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")

    job = manager.submit(file.filename or dest.name, dest, zone_map, job_id=job_id)
    return {"job_id": job.id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job.public()


@app.get("/api/v1/config", dependencies=[Depends(require_api_key)])
async def get_config():
    rules = [
        {k: r.get(k) for k in ("type", "title", "kind", "zone", "watch_zone", "staff_zone", "classes", "watch_classes")}
        for r in state["rules"].get("rules", [])
        if r.get("enabled", True)
    ]
    return {"zone_profiles": state["zones"], "default_profile": settings.default_zone_profile,
            "rules": rules, "norms": state["rules"].get("norms", {})}


# Serve the built React app (frontend/dist) from the same port, if present.
if settings.frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
