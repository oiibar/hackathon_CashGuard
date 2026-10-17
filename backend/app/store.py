from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger("cashguard.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          text PRIMARY KEY,
    filename    text NOT NULL,
    video_path  text NOT NULL,
    zones       jsonb,
    status      text NOT NULL CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    verdict     text CHECK (verdict IN ('violation', 'no_violation')),
    error       text,
    meta        jsonb,
    created_at  timestamptz NOT NULL,
    deadline    timestamptz NOT NULL,
    started_at  timestamptz,
    finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS jobs_unfinished_idx ON jobs (created_at) WHERE status IN ('queued', 'processing');
CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs (created_at DESC);

CREATE TABLE IF NOT EXISTS violations (
    id           bigserial PRIMARY KEY,
    job_id       text NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    position     int  NOT NULL,
    type_id      text NOT NULL,
    title        text NOT NULL,
    source       text,
    severity     text,
    timecode_sec real NOT NULL,
    end_sec      real,
    duration_sec real,
    confidence   real,
    description  text NOT NULL,
    observed     text,
    why          text,
    legal_basis  jsonb NOT NULL DEFAULT '[]',
    UNIQUE (job_id, position)
);
CREATE INDEX IF NOT EXISTS violations_type_idx ON violations (type_id);

CREATE TABLE IF NOT EXISTS evidence_frames (
    violation_id  bigint NOT NULL REFERENCES violations (id) ON DELETE CASCADE,
    position      int    NOT NULL,
    timestamp_sec real   NOT NULL,
    path          text   NOT NULL,
    PRIMARY KEY (violation_id, position)
);

CREATE TABLE IF NOT EXISTS model_versions (
    id          int PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    status      text NOT NULL CHECK (status IN ('active', 'retired', 'rejected')),
    trigger     text NOT NULL,
    path        text NOT NULL,
    decision    text NOT NULL,
    params      jsonb NOT NULL,
    metrics     jsonb NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS model_versions_single_active ON model_versions (status) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS reviews (
    job_id        text PRIMARY KEY REFERENCES jobs (id) ON DELETE CASCADE,
    label         text NOT NULL CHECK (label IN ('violation', 'no_violation')),
    timecode_sec  real CHECK (timecode_sec >= 0),
    type_id       text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    model_version int REFERENCES model_versions (id) ON DELETE SET NULL,
    CHECK (label = 'no_violation' OR timecode_sec IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS reviews_pending_idx ON reviews (created_at) WHERE model_version IS NULL;
"""


class Store(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def save(self, job: Any) -> None: ...
    def load(self, job_id: str) -> Any | None: ...
    def unfinished(self) -> list[Any]: ...
    def recent(self, limit: int) -> list[dict[str, Any]]: ...
    def save_review(self, review: dict[str, Any]) -> None: ...
    def reviews(self) -> list[dict[str, Any]]: ...
    def mark_reviews(self, job_ids: list[str], version: int) -> None: ...
    def add_version(self, version: dict[str, Any]) -> int: ...
    def versions(self) -> list[dict[str, Any]]: ...
    def activate_version(self, version_id: int) -> bool: ...


def _summary(job: Any) -> dict[str, Any]:
    result = job.result or {}
    return {"job_id": job.id, "filename": job.filename, "created_at": job.created_at, "status": job.status,
            "verdict": result.get("verdict"), "violations": len(result.get("violations", []))}


class FileStore:
    def __init__(self, jobs_dir: Path, job_cls: type):
        self.dir = jobs_dir
        self.job_cls = job_cls
        self.registry = jobs_dir.parent / "learning.json"

    def open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        pass

    def save(self, job: Any) -> None:
        path = self.dir / f"{job.id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def load(self, job_id: str) -> Any | None:
        path = self.dir / f"{job_id}.json"
        if not path.is_file() or path.parent != self.dir:
            return None
        return self.job_cls(**json.loads(path.read_text(encoding="utf-8")))

    def _all(self) -> list[Any]:
        jobs = []
        for path in self.dir.glob("*.json"):
            try:
                jobs.append(self.job_cls(**json.loads(path.read_text(encoding="utf-8"))))
            except Exception as exc:
                log.warning("cannot read %s: %s", path.name, exc)
        return jobs

    def unfinished(self) -> list[Any]:
        return sorted((j for j in self._all() if j.status in ("queued", "processing")), key=lambda j: j.created_at)

    def recent(self, limit: int) -> list[dict[str, Any]]:
        return [_summary(j) for j in sorted(self._all(), key=lambda j: -j.created_at)[:limit]]

    def _registry(self) -> dict[str, Any]:
        if self.registry.is_file():
            return json.loads(self.registry.read_text(encoding="utf-8"))
        return {"reviews": {}, "versions": []}

    def _write_registry(self, data: dict[str, Any]) -> None:
        tmp = self.registry.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.registry)

    def save_review(self, review: dict[str, Any]) -> None:
        data = self._registry()
        data["reviews"][review["job_id"]] = {**review, "created_at": time.time(), "model_version": None}
        self._write_registry(data)

    def reviews(self) -> list[dict[str, Any]]:
        return sorted(self._registry()["reviews"].values(), key=lambda r: r["created_at"])

    def mark_reviews(self, job_ids: list[str], version: int) -> None:
        data = self._registry()
        for jid in job_ids:
            if jid in data["reviews"]:
                data["reviews"][jid]["model_version"] = version
        self._write_registry(data)

    def add_version(self, version: dict[str, Any]) -> int:
        data = self._registry()
        vid = max((v["id"] for v in data["versions"]), default=0) + 1
        if version["status"] == "active":
            for v in data["versions"]:
                if v["status"] == "active":
                    v["status"] = "retired"
        data["versions"].append({**version, "id": vid, "created_at": time.time()})
        self._write_registry(data)
        return vid

    def versions(self) -> list[dict[str, Any]]:
        return sorted(self._registry()["versions"], key=lambda v: -v["id"])

    def activate_version(self, version_id: int) -> bool:
        data = self._registry()
        if version_id and not any(v["id"] == version_id for v in data["versions"]):
            return False
        for v in data["versions"]:
            if v["id"] == version_id:
                v["status"] = "active"
            elif v["status"] == "active":
                v["status"] = "retired"
        self._write_registry(data)
        return True


class PgStore:
    def __init__(self, dsn: str, evidence_dir: Path, job_cls: type, connect_timeout: float = 60.0):
        self.dsn = dsn
        self.evidence_dir = evidence_dir
        self.job_cls = job_cls
        self.connect_timeout = connect_timeout
        self.pool = None

    def open(self) -> None:
        from psycopg_pool import ConnectionPool

        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.pool = ConnectionPool(self.dsn, min_size=1, max_size=4, open=False)
        self.pool.open(wait=True, timeout=self.connect_timeout)
        with self.pool.connection() as conn:
            conn.execute(SCHEMA)
        log.info("postgres ready")

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()

    def save(self, job: Any) -> None:
        result = job.result or {}
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """
                INSERT INTO jobs (id, filename, video_path, zones, status, verdict, error, meta,
                                  created_at, deadline, started_at, finished_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s),
                        to_timestamp(%s), to_timestamp(%s))
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status, verdict = EXCLUDED.verdict, error = EXCLUDED.error,
                    meta = EXCLUDED.meta, deadline = EXCLUDED.deadline,
                    started_at = EXCLUDED.started_at, finished_at = EXCLUDED.finished_at
                """,
                (job.id, job.filename, job.video_path, _json(job.zones), job.status, result.get("verdict"),
                 job.error, _json(result.get("meta")), job.created_at, job.deadline, job.started_at,
                 job.finished_at),
            )
            if job.status == "done" and job.result is not None:
                self._save_violations(conn, job.id, result.get("violations", []))

    def _save_violations(self, conn: Any, job_id: str, violations: list[dict[str, Any]]) -> None:
        conn.execute("DELETE FROM violations WHERE job_id = %s", (job_id,))
        folder = self.evidence_dir / job_id
        folder.mkdir(parents=True, exist_ok=True)
        for pos, v in enumerate(violations):
            vid = conn.execute(
                """
                INSERT INTO violations (job_id, position, type_id, title, source, severity, timecode_sec,
                                        end_sec, duration_sec, confidence, description, observed, why, legal_basis)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (job_id, pos, v.get("type_id") or v["type"], v["type"], v.get("source"), v.get("severity"),
                 v["timecode_sec"], v.get("end_sec"), v.get("duration_sec"), v.get("confidence"),
                 v["description"], v.get("observed"), v.get("why"), _json(v.get("legal_basis", []))),
            ).fetchone()[0]
            for fpos, frame in enumerate(v.get("evidence_frames", [])):
                path = folder / f"{pos:02d}_{fpos}.jpg"
                path.write_bytes(base64.b64decode(frame["image_base64"]))
                conn.execute(
                    "INSERT INTO evidence_frames (violation_id, position, timestamp_sec, path) VALUES (%s, %s, %s, %s)",
                    (vid, fpos, frame["timestamp_sec"], str(path)),
                )

    _JOB_COLS = """id, filename, video_path, zones, status, verdict, error, meta,
                   extract(epoch FROM created_at), extract(epoch FROM deadline),
                   extract(epoch FROM started_at), extract(epoch FROM finished_at)"""

    def _job(self, row: tuple) -> Any:
        (jid, filename, video_path, zones, status, verdict, error, meta, created, deadline, started, finished) = row
        job = self.job_cls(id=jid, filename=filename, video_path=video_path, zones=zones,
                           created_at=float(created), deadline=float(deadline), status=status, error=error,
                           started_at=_f(started), finished_at=_f(finished))
        if status == "done":
            job.progress = 1.0
            job.result = {"verdict": verdict, "violations": [], "meta": meta or {}}
        return job

    def load(self, job_id: str) -> Any | None:
        with self.pool.connection() as conn:
            row = conn.execute(f"SELECT {self._JOB_COLS} FROM jobs WHERE id = %s", (job_id,)).fetchone()
            if row is None:
                return None
            job = self._job(row)
            if job.result is not None:
                job.result["violations"] = self._violations(conn, job_id)
        return job

    def _violations(self, conn: Any, job_id: str) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT id, type_id, title, source, severity, timecode_sec, end_sec, duration_sec, confidence,
                   description, observed, why, legal_basis
            FROM violations WHERE job_id = %s ORDER BY position
            """,
            (job_id,),
        ).fetchall()
        frames: dict[int, list[dict[str, Any]]] = {}
        if rows:
            for vid, ts, path in conn.execute(
                "SELECT violation_id, timestamp_sec, path FROM evidence_frames "
                "WHERE violation_id = ANY(%s) ORDER BY violation_id, position",
                ([r[0] for r in rows],),
            ).fetchall():
                p = Path(path)
                if p.is_file():
                    frames.setdefault(vid, []).append(
                        {"timestamp_sec": round(float(ts), 2),
                         "image_base64": base64.b64encode(p.read_bytes()).decode("ascii")})
        out = []
        for (vid, type_id, title, source, severity, tc, end, dur, conf, desc, observed, why, basis) in rows:
            v = {"type": title, "type_id": type_id, "timecode_sec": round(float(tc), 2),
                 "end_sec": _r(end), "duration_sec": _r(dur), "confidence": _r(conf, 3), "severity": severity,
                 "source": source, "description": desc, "observed": observed, "why": why,
                 "legal_basis": basis or [], "evidence_frames": frames.get(vid, [])}
            out.append({k: val for k, val in v.items() if val is not None})
        return out

    def unfinished(self) -> list[Any]:
        with self.pool.connection() as conn:
            rows = conn.execute(f"SELECT {self._JOB_COLS} FROM jobs WHERE status IN ('queued', 'processing') "
                                "ORDER BY created_at").fetchall()
        return [self._job(r) for r in rows]

    def recent(self, limit: int) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT j.id, j.filename, extract(epoch FROM j.created_at), j.status, j.verdict,
                       count(v.id), r.label
                FROM jobs j
                LEFT JOIN violations v ON v.job_id = j.id
                LEFT JOIN reviews r ON r.job_id = j.id
                GROUP BY j.id, r.label ORDER BY j.created_at DESC LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [{"job_id": r[0], "filename": r[1], "created_at": float(r[2]), "status": r[3], "verdict": r[4],
                 "violations": int(r[5]), "review": r[6]} for r in rows]

    def save_review(self, review: dict[str, Any]) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO reviews (job_id, label, timecode_sec, type_id) VALUES (%s, %s, %s, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    label = EXCLUDED.label, timecode_sec = EXCLUDED.timecode_sec, type_id = EXCLUDED.type_id,
                    created_at = now(), model_version = NULL
                """,
                (review["job_id"], review["label"], review.get("timecode_sec"), review.get("type_id")),
            )

    def reviews(self) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT job_id, label, timecode_sec, type_id, extract(epoch FROM created_at), model_version "
                "FROM reviews ORDER BY created_at").fetchall()
        return [{"job_id": r[0], "label": r[1], "timecode_sec": _f(r[2]), "type_id": r[3], "created_at": float(r[4]),
                 "model_version": r[5]} for r in rows]

    def mark_reviews(self, job_ids: list[str], version: int) -> None:
        with self.pool.connection() as conn:
            conn.execute("UPDATE reviews SET model_version = %s WHERE job_id = ANY(%s) AND model_version IS NULL",
                         (version, job_ids))

    def add_version(self, version: dict[str, Any]) -> int:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("LOCK TABLE model_versions IN EXCLUSIVE MODE")
            vid = conn.execute("SELECT coalesce(max(id), 0) + 1 FROM model_versions").fetchone()[0]
            if version["status"] == "active":
                conn.execute("UPDATE model_versions SET status = 'retired' WHERE status = 'active'")
            conn.execute(
                "INSERT INTO model_versions (id, status, trigger, path, decision, params, metrics) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (vid, version["status"], version["trigger"], version["path"], version["decision"],
                 _json(version["params"]), _json(version["metrics"])),
            )
        return vid

    def versions(self) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, extract(epoch FROM created_at), status, trigger, path, decision, params, metrics "
                "FROM model_versions ORDER BY id DESC").fetchall()
        return [{"id": r[0], "created_at": float(r[1]), "status": r[2], "trigger": r[3], "path": r[4],
                 "decision": r[5], "params": r[6], "metrics": r[7]} for r in rows]

    def activate_version(self, version_id: int) -> bool:
        with self.pool.connection() as conn, conn.transaction():
            if version_id and conn.execute("SELECT 1 FROM model_versions WHERE id = %s",
                                           (version_id,)).fetchone() is None:
                return False
            conn.execute("UPDATE model_versions SET status = 'retired' WHERE status = 'active'")
            conn.execute("UPDATE model_versions SET status = 'active' WHERE id = %s", (version_id,))
        return True


def _json(value: Any) -> Any:
    if value is None:
        return None
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def _f(value: Any) -> float | None:
    return None if value is None else float(value)


def _r(value: Any, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def make_store(dsn: str, data_dir: Path, job_cls: type) -> Store:
    if dsn:
        return PgStore(dsn, data_dir / "evidence", job_cls)
    log.warning("DATABASE_URL is not set: jobs are stored as JSON files in %s", data_dir / "jobs")
    return FileStore(data_dir / "jobs", job_cls)
