"""In-process job queue: asyncio.Queue + N worker coroutines.

Why not Celery/Redis: one GPU/CPU box, one model in memory, a hackathon
jury. An asyncio queue gives the same non-blocking POST with zero extra
services. The blocking YOLO loop runs in a thread via asyncio.to_thread, so
the event loop keeps answering GET /jobs/{id} while a video is processed.

Jobs are persisted as JSON files, so results survive a restart and
unfinished jobs are re-queued on startup.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import settings

log = logging.getLogger("cashguard.jobs")


@dataclass
class Job:
    id: str
    filename: str
    video_path: str
    zones: dict[str, Any]
    created_at: float
    deadline: float
    status: str = "queued"  # queued | processing | done | failed
    progress: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        """Shape returned by GET /api/v1/jobs/{id}."""
        status = "processing" if self.status in ("queued", "processing") else self.status
        body: dict[str, Any] = {"job_id": self.id, "status": status}
        if status == "processing":
            body["stage"] = self.status
            body["progress"] = round(self.progress, 3)
        elif status == "done" and self.result:
            body["verdict"] = self.result["verdict"]
            body["violations"] = self.result["violations"]
            body["meta"] = self.result.get("meta", {})
        elif status == "failed":
            body["error"] = self.error or "unknown error"
        body["filename"] = self.filename
        return body


@dataclass
class JobManager:
    analyzers: list[Any] = field(default_factory=list)
    jobs: dict[str, Job] = field(default_factory=dict)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _workers: list[asyncio.Task] = field(default_factory=list)

    # ---------------------------------------------------------- lifecycle

    async def start(self, analyzers: list[Any]) -> None:
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.analyzers = analyzers
        self.queue = asyncio.Queue()  # bound to the running loop
        self._workers = []
        self._restore()
        for i, analyzer in enumerate(analyzers):
            self._workers.append(asyncio.create_task(self._worker(i, analyzer), name=f"worker-{i}"))

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def _restore(self) -> None:
        for path in sorted(settings.jobs_dir.glob("*.json")):
            try:
                job = Job(**json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:  # corrupted file: skip, keep serving
                log.warning("cannot restore %s: %s", path.name, exc)
                continue
            self.jobs[job.id] = job
            if job.status in ("queued", "processing"):
                if Path(job.video_path).exists():
                    job.status, job.progress = "queued", 0.0
                    job.deadline = time.time() + settings.job_deadline_sec
                    self.queue.put_nowait(job.id)
                    log.info("re-queued %s after restart", job.id)
                else:
                    self._fail(job, "server restarted and the video file is gone")

    # ------------------------------------------------------------- public

    def submit(self, filename: str, video_path: Path, zones: dict[str, Any], job_id: str | None = None) -> Job:
        now = time.time()
        job = Job(
            id=job_id or uuid.uuid4().hex,
            filename=filename,
            video_path=str(video_path),
            zones=zones,
            created_at=now,
            deadline=now + settings.job_deadline_sec,
        )
        self.jobs[job.id] = job
        self._save(job)
        self.queue.put_nowait(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    # ------------------------------------------------------------ workers

    async def _worker(self, idx: int, analyzer: Any) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            try:
                if job is None:
                    continue
                await self._process(job, analyzer)
            except Exception as exc:  # never let a worker die
                log.exception("worker %d crashed on %s", idx, job_id)
                if job:
                    self._fail(job, f"internal error: {exc}")
            finally:
                self.queue.task_done()

    async def _process(self, job: Job, analyzer: Any) -> None:
        job.status = "processing"
        job.started_at = time.time()
        self._save(job)
        log.info("job %s started (%.0fs left before deadline)", job.id, job.deadline - job.started_at)

        def on_progress(p: float) -> None:
            job.progress = p

        try:
            result = await asyncio.to_thread(analyzer.run, Path(job.video_path), job.zones, job.deadline, on_progress)
        except ValueError as exc:  # unreadable video etc.
            self._fail(job, str(exc))
            return
        job.result = result
        job.status = "done"
        job.progress = 1.0
        job.finished_at = time.time()
        self._save(job)
        self._cleanup_video(job)
        log.info("job %s done in %.1fs: %s, %d violation(s)", job.id, job.finished_at - job.created_at,
                 result["verdict"], len(result["violations"]))

    # ------------------------------------------------------------ helpers

    def _fail(self, job: Job, msg: str) -> None:
        job.status = "failed"
        job.error = msg
        job.finished_at = time.time()
        self._save(job)
        self._cleanup_video(job)
        log.warning("job %s failed: %s", job.id, msg)

    def _cleanup_video(self, job: Job) -> None:
        if not settings.keep_videos:
            Path(job.video_path).unlink(missing_ok=True)

    def _save(self, job: Job) -> None:
        path = settings.jobs_dir / f"{job.id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash never leaves half a file


manager = JobManager()
