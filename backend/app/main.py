from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import require_api_key
from .config import settings
from .jobs import manager
from .learner import learner
from .pipeline.rules import legal_basis, load_yaml, wanted_classes

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cashguard")

ALLOWED_EXT = {e.strip().lower() for e in os.getenv("ALLOWED_EXT", ".mov,.mp4,.m4v,.avi,.mkv").split(",")}

state: dict = {}


def _build_analyzers(rules_cfg: dict) -> list:
    from .pipeline.analyzer import Analyzer
    from .pipeline.detector import Detector

    analyzers = []
    for _ in range(max(settings.workers, 1)):
        det = Detector(settings.model_paths, settings.device, settings.imgsz, settings.conf, wanted_classes(rules_cfg),
                       aliases=rules_cfg.get("class_aliases"), class_conf=rules_cfg.get("class_conf"))
        analyzers.append(Analyzer(det, rules_cfg, settings))
    return analyzers


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.api_key == "change-me":
        log.warning("API_KEY is the default 'change-me'. Set API_KEY before exposing the service.")
    rules_cfg = load_yaml(settings.rules_path)
    state["rules"] = rules_cfg
    state["zones"] = load_yaml(settings.zones_path).get("profiles", {})
    builder = getattr(app.state, "analyzer_builder", _build_analyzers)
    analyzers = await asyncio.to_thread(builder, rules_cfg)
    state["detector_fps"] = [round(1 / a.sec_per_frame, 1) for a in analyzers if getattr(a, "sec_per_frame", 0)]
    det = getattr(analyzers[0], "detector", None) if analyzers else None
    state["classes"] = det.class_names if det else None
    state["models"] = (det.model_names if det else []) + (
        ["violation_clf (CLIP ViT-B/32)"] if analyzers and getattr(analyzers[0], "learned", None) else [])
    await manager.start(analyzers)
    await asyncio.to_thread(learner.start, manager.store, analyzers)
    log.info("ready: %d worker(s), detector fps %s, model v%d", len(analyzers), state["detector_fps"],
             learner.version)
    yield
    await manager.stop()


app = FastAPI(title="CashGuard", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


def _resolve_zones(zones_json: str | None, profile: str | None) -> dict | None:
    if zones_json:
        try:
            zones = json.loads(zones_json)
            assert isinstance(zones, dict)
            for name, pts in zones.items():
                assert isinstance(pts, list) and len(pts) >= 3, name
                assert all(len(p) == 2 and 0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in pts), name
            return zones
        except Exception as exc:
            raise HTTPException(422, f"zones must be JSON {{name: [[x,y],...]}} with 0..1 coords: {exc}")
    name = profile or settings.default_zone_profile
    if name not in state["zones"]:
        raise HTTPException(422, f"unknown zone profile: {name}")
    return state["zones"][name] or None


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "workers": len(manager.analyzers),
        "queued": manager.queue.qsize(),
        "storage": "postgres" if settings.database_url else "files",
        "detector_fps": state.get("detector_fps"),
        "models": state.get("models", []),
        "model_version": learner.version,
    }


@app.post("/api/v1/analyze", dependencies=[Depends(require_api_key)])
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

    def save() -> int:
        size = 0
        with open(dest, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    break
                out.write(chunk)
        return size

    size = await asyncio.to_thread(save)
    if size > limit:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"file larger than {settings.max_upload_mb} MB")
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")

    job = await asyncio.to_thread(manager.submit, file.filename or dest.name, dest, zone_map, job_id)
    return {"job_id": job.id, "status": "processing"}


@app.get("/api/v1/jobs", dependencies=[Depends(require_api_key)])
async def list_jobs(limit: int = 20):
    return {"jobs": await asyncio.to_thread(manager.recent, max(1, min(limit, 200)))}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_job(job_id: str):
    job = await asyncio.to_thread(manager.get, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job.public()


class ReviewIn(BaseModel):
    label: Literal["violation", "no_violation"]
    timecode_sec: float | None = Field(None, ge=0)
    type_id: str | None = Field(None, max_length=64)


@app.post("/api/v1/jobs/{job_id}/review", dependencies=[Depends(require_api_key)])
async def review_job(job_id: str, body: ReviewIn):
    job = await asyncio.to_thread(manager.get, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    if job.status != "done":
        raise HTTPException(status.HTTP_409_CONFLICT, "only finished jobs can be reviewed")
    if body.label == "violation" and body.timecode_sec is None:
        raise HTTPException(422, "timecode_sec is required when the video has a violation")
    duration = (job.result or {}).get("meta", {}).get("duration_sec")
    if body.timecode_sec is not None and duration and body.timecode_sec > duration + 1:
        raise HTTPException(422, f"timecode_sec is beyond the video length ({duration} s)")
    saved = await asyncio.to_thread(learner.review, job, body.label, body.timecode_sec, body.type_id)
    if await asyncio.to_thread(learner.should_retrain):
        saved["retraining"] = learner.schedule("reviews", manager.busy)
    return saved


@app.get("/api/v1/model")
async def model_info():
    return await asyncio.to_thread(learner.summary)


@app.post("/api/v1/model/retrain", dependencies=[Depends(require_api_key)])
async def retrain():
    if not learner.enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "learning is disabled or the base training set is missing")
    if not learner.schedule("manual", manager.busy):
        raise HTTPException(status.HTTP_409_CONFLICT, "retraining is already running")
    return {"status": "started"}


@app.post("/api/v1/model/versions/{version_id}/activate", dependencies=[Depends(require_api_key)])
async def activate(version_id: int):
    if not await asyncio.to_thread(learner.activate, version_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    return {"active_version": version_id}


@app.get("/api/v1/config")
async def get_config():
    norms = state["rules"].get("norms", {})
    available = state.get("classes")
    rules = []
    for r in state["rules"].get("rules", []):
        if not r.get("enabled", True):
            continue
        need = set(r.get("classes", [])) | set(r.get("watch_classes", []))
        rules.append({
            **{k: r.get(k) for k in ("type", "title", "kind", "severity", "min_duration_sec")},
            "why": " ".join((r.get("why") or "").split()),
            "legal_basis": legal_basis(r, norms),
            "active": None if available is None else (not need or bool(need & available)),
        })
    known = {r["type"] for r in rules}
    defs = {r["type"]: r for r in state["rules"].get("rules", [])}
    for vtype, d in state["rules"].get("learned_types", {}).items():
        d = {**defs.get(vtype, {}), **d}
        if vtype in known or "title" not in d:
            continue
        rules.append({"type": vtype, "title": d["title"], "kind": "learned", "severity": d.get("severity"),
                      "min_duration_sec": None, "why": " ".join((d.get("why") or "").split()),
                      "legal_basis": legal_basis(d, norms), "active": True})
    return {"zone_profiles": state["zones"], "default_profile": settings.default_zone_profile,
            "rules": rules, "norms": norms}


if settings.frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
