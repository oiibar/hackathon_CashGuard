from __future__ import annotations

import asyncio
import shutil
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import settings
from .learner import embeddings_path
from .store import Store, make_store

log = logging.getLogger("cashguard.jobs")


@dataclass
class Job:
    id: str
    filename: str
    video_path: str
    zones: dict[str, Any] | None
    created_at: float
    deadline: float
    status: str = "queued"
    progress: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
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
        body["created_at"] = self.created_at
        return body


@dataclass
class JobManager:
    analyzers: list[Any] = field(default_factory=list)
    active: dict[str, Job] = field(default_factory=dict)
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    store: Store | None = None
    _workers: list[asyncio.Task] = field(default_factory=list)

    async def start(self, analyzers: list[Any], store: Store | None = None) -> None:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or make_store(settings.database_url, settings.data_dir, Job)
        await asyncio.to_thread(self.store.open)
        self.analyzers = analyzers
        self.queue = asyncio.Queue()
        self.active = {}
        self._workers = []
        await asyncio.to_thread(self._restore)
        for i, analyzer in enumerate(analyzers):
            self._workers.append(asyncio.create_task(self._worker(i, analyzer), name=f"worker-{i}"))
        if settings.retention_days > 0:
            self._workers.append(asyncio.create_task(self._janitor(), name="janitor"))

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        if self.store is not None:
            await asyncio.to_thread(self.store.close)

    def busy(self) -> bool:
        return bool(self.active)

    async def _janitor(self, every_sec: float = 6 * 3600) -> None:
        while True:
            try:
                await asyncio.to_thread(self.purge)
            except Exception:
                log.exception("retention purge failed")
            await asyncio.sleep(every_sec)

    def purge(self, now: float | None = None) -> dict[str, list[str]]:
        before = (now or time.time()) - settings.retention_days * 86400
        out = self.store.purge(before)
        for job_id in out["deleted"]:
            shutil.rmtree(settings.data_dir / "evidence" / job_id, ignore_errors=True)
            embeddings_path(job_id).unlink(missing_ok=True)
        if out["deleted"] or out["stripped"]:
            log.info("retention: deleted %d job(s), removed evidence of %d reviewed job(s)",
                     len(out["deleted"]), len(out["stripped"]))
        return out

    def _restore(self) -> None:
        for job in self.store.unfinished():
            if Path(job.video_path).exists():
                job.status, job.progress = "queued", 0.0
                job.deadline = time.time() + settings.job_deadline_sec
                self.active[job.id] = job
                self.store.save(job)
                self.queue.put_nowait(job.id)
                log.info("re-queued %s after restart", job.id)
            else:
                self._fail(job, "server restarted and the video file is gone")

    def submit(self, filename: str, video_path: Path, zones: dict[str, Any] | None, job_id: str | None = None) -> Job:
        now = time.time()
        job = Job(id=job_id or uuid.uuid4().hex, filename=filename, video_path=str(video_path), zones=zones,
                  created_at=now, deadline=now + settings.job_deadline_sec)
        self.store.save(job)
        self.active[job.id] = job
        self.queue.put_nowait(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.active.get(job_id) or self.store.load(job_id)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.recent(limit)

    async def _worker(self, idx: int, analyzer: Any) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.active.get(job_id)
            try:
                if job is not None:
                    await self._process(job, analyzer)
            except Exception as exc:
                log.exception("worker %d crashed on %s", idx, job_id)
                if job:
                    await asyncio.to_thread(self._fail, job, f"internal error: {exc}")
            finally:
                self.queue.task_done()

    async def _process(self, job: Job, analyzer: Any) -> None:
        job.status = "processing"
        job.started_at = time.time()
        await asyncio.to_thread(self.store.save, job)
        log.info("job %s started (%.0fs left before deadline)", job.id, job.deadline - job.started_at)

        def on_progress(p: float) -> None:
            job.progress = p

        emb = embeddings_path(job.id) if settings.learning_enabled else None
        try:
            result = await asyncio.to_thread(analyzer.run, Path(job.video_path), job.zones, job.deadline,
                                             on_progress, emb)
        except ValueError as exc:
            await asyncio.to_thread(self._fail, job, str(exc))
            return
        job.result = result
        job.status = "done"
        job.progress = 1.0
        job.finished_at = time.time()
        await asyncio.to_thread(self._finish, job)
        log.info("job %s done in %.1fs: %s, %d violation(s)", job.id, job.finished_at - job.created_at,
                 result["verdict"], len(result["violations"]))

    def _finish(self, job: Job) -> None:
        self.store.save(job)
        self.active.pop(job.id, None)
        if not settings.keep_videos:
            Path(job.video_path).unlink(missing_ok=True)

    def _fail(self, job: Job, msg: str) -> None:
        job.status = "failed"
        job.error = msg
        job.finished_at = time.time()
        self._finish(job)
        log.warning("job %s failed: %s", job.id, msg)


manager = JobManager()
