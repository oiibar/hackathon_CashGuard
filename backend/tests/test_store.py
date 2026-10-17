import base64
import os
import time
import uuid

import pytest

from app.jobs import Job
from app.store import FileStore, PgStore

JPEG = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9").decode()


def report_job():
    now = time.time()
    job = Job(id=uuid.uuid4().hex, filename="Х-30.mov", video_path="/tmp/x.mov", zones=None,
              created_at=now, deadline=now + 270)
    return job


def finish(job):
    job.status, job.progress, job.finished_at = "done", 1.0, time.time()
    job.result = {
        "verdict": "violation",
        "meta": {"duration_sec": 18.8},
        "violations": [{
            "type": "Постороннее лицо в помещении кассы", "type_id": "outsider_entered", "source": "learned",
            "timecode_sec": 12.5, "end_sec": 15.0, "duration_sec": 2.5, "confidence": 0.95, "severity": "high",
            "description": "00:12.5–00:15.0: в помещение кассы входит человек…", "observed": "входит человек",
            "why": "Кассовый узел изолирован…",
            "legal_basis": [{"short": "Правила № 14", "clause": "п. 8", "quote": "…", "url": "https://adilet.zan.kz"}],
            "evidence_frames": [{"timestamp_sec": 12.5, "image_base64": JPEG},
                                {"timestamp_sec": 13.5, "image_base64": JPEG}],
        }],
    }


def roundtrip(store):
    job = report_job()
    store.save(job)
    assert job.id in [j.id for j in store.unfinished()]
    assert store.load(job.id).status == "queued"

    finish(job)
    store.save(job)
    store.save(job)
    back = store.load(job.id)
    assert back.status == "done" and back.result["verdict"] == "violation"
    v = back.result["violations"]
    assert len(v) == 1 and v[0]["type_id"] == "outsider_entered" and v[0]["timecode_sec"] == 12.5
    assert [f["timestamp_sec"] for f in v[0]["evidence_frames"]] == [12.5, 13.5]
    assert v[0]["evidence_frames"][0]["image_base64"] == JPEG
    assert v[0]["legal_basis"][0]["clause"] == "п. 8"
    assert back.public()["violations"][0]["type"] == "Постороннее лицо в помещении кассы"
    assert job.id not in [j.id for j in store.unfinished()]
    assert any(r["job_id"] == job.id and r["violations"] == 1 for r in store.recent(50))
    assert store.load("no-such-job") is None


def learning_roundtrip(store):
    job = report_job()
    finish(job)
    store.save(job)
    store.save_review({"job_id": job.id, "label": "violation", "timecode_sec": 12.0, "type_id": "outsider_entered"})
    store.save_review({"job_id": job.id, "label": "violation", "timecode_sec": 12.5, "type_id": "outsider_entered"})
    reviews = [r for r in store.reviews() if r["job_id"] == job.id]
    assert len(reviews) == 1 and reviews[0]["timecode_sec"] == 12.5 and reviews[0]["model_version"] is None

    first = store.add_version({"status": "active", "trigger": "test", "path": "/m/v1.npz", "decision": "ok",
                               "params": {"k": 16}, "metrics": {"mcc": 0.3}})
    second = store.add_version({"status": "rejected", "trigger": "test", "path": "/m/v2.npz",
                                "decision": "worse", "params": {}, "metrics": {}})
    store.mark_reviews([job.id], first)
    assert [r for r in store.reviews() if r["job_id"] == job.id][0]["model_version"] == first
    status = {v["id"]: v["status"] for v in store.versions()}
    assert status == {first: "active", second: "rejected"}
    assert store.activate_version(second)
    assert {v["id"]: v["status"] for v in store.versions()} == {first: "retired", second: "active"}
    assert store.activate_version(0)
    assert all(v["status"] != "active" for v in store.versions())
    assert not store.activate_version(999)


def test_file_store(tmp_path):
    store = FileStore(tmp_path / "jobs", Job)
    store.open()
    roundtrip(store)
    learning_roundtrip(store)


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_postgres_store(tmp_path):
    import psycopg

    dsn = os.environ["TEST_DATABASE_URL"]
    schema = f"cg_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in dsn else "?"
    store = PgStore(f"{dsn}{sep}options=-csearch_path%3D{schema}", tmp_path / "evidence", Job, connect_timeout=20)
    store.open()
    try:
        roundtrip(store)
        learning_roundtrip(store)
    finally:
        store.close()
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA {schema} CASCADE")
