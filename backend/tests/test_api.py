"""API contract tests with a stub analyzer (no YOLO needed, runs in seconds)."""
import time

from fastapi.testclient import TestClient

from app.main import app

KEY = {"X-API-Key": "test-key"}


class StubAnalyzer:
    sec_per_frame = 0.01

    def run(self, path, zones, deadline, progress=None):
        return {
            "verdict": "violation",
            "violations": [{
                "type": "mobile_phone_in_cash_zone",
                "timecode_sec": 3.5,
                "description": "Мобильный телефон. Нормативное основание: пост. НБРК № 231, п. 12",
                "evidence_frames": [{"timecode_sec": 3.5, "image_base64": "AAAA"}],
            }],
            "meta": {"zones": list(zones)},
        }


app.state.analyzer_builder = lambda rules: [StubAnalyzer()]


def wait_done(client, job_id, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/api/v1/jobs/{job_id}", headers=KEY).json()
        if body["status"] != "processing":
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_auth_required():
    with TestClient(app) as client:
        assert client.post("/api/v1/analyze", files={"file": ("a.mov", b"x")}).status_code == 401
        assert client.get("/api/v1/jobs/x", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/v1/health").status_code == 200


def test_analyze_and_poll_contract():
    with TestClient(app) as client:
        r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("cam1.MOV", b"fake-video-bytes")})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        body = wait_done(client, job_id)
        assert body["status"] == "done"
        assert body["verdict"] in ("violation", "no_violation")
        v = body["violations"][0]
        assert {"type", "timecode_sec", "description", "evidence_frames"} <= v.keys()
        assert "image_base64" in v["evidence_frames"][0]


def test_rejects_wrong_extension_and_unknown_job():
    with TestClient(app) as client:
        r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("notes.txt", b"hi")})
        assert r.status_code == 415
        assert client.get("/api/v1/jobs/does-not-exist", headers=KEY).status_code == 404


def test_custom_zones_passed_to_analyzer():
    zones = '{"cash_zone": [[0,0],[0.5,0],[0.5,1]], "client_zone": [[0.5,0],[1,0],[1,1]]}'
    with TestClient(app) as client:
        r = client.post("/api/v1/analyze", headers=KEY, data={"zones": zones}, files={"file": ("a.mov", b"x")})
        body = wait_done(client, r.json()["job_id"])
        assert body["meta"]["zones"] == ["cash_zone", "client_zone"]
        bad = client.post("/api/v1/analyze", headers=KEY, data={"zones": "[1,2]"}, files={"file": ("a.mov", b"x")})
        assert bad.status_code == 422
