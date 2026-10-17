import time

from fastapi.testclient import TestClient

from app.main import app

KEY = {"X-API-Key": "test-key"}


class StubAnalyzer:
    sec_per_frame = 0.01

    def run(self, path, zones, deadline, progress=None, embeddings_out=None):
        return {
            "verdict": "violation",
            "violations": [{
                "type": "Мобильный телефон на рабочем месте кассира",
                "type_id": "mobile_phone_in_cash_zone",
                "timecode_sec": 3.5,
                "description": "Мобильный телефон. Нормативное основание: пост. НБРК № 231, п. 12",
                "evidence_frames": [{"timestamp_sec": 3.5, "image_base64": "AAAA"}],
            }],
            "meta": {"zones": list(zones) if zones else None},
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
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        body = wait_done(client, job_id)
        assert body["status"] == "done"
        assert body["verdict"] in ("violation", "no_violation")
        v = body["violations"][0]
        assert {"type", "timecode_sec", "description", "evidence_frames"} <= v.keys()
        assert {"timestamp_sec", "image_base64"} <= v["evidence_frames"][0].keys()
        assert 1 <= len(v["evidence_frames"]) <= 3


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


def test_default_profile_is_auto_and_config_is_public():
    with TestClient(app) as client:
        r = client.post("/api/v1/analyze", headers=KEY, files={"file": ("a.mov", b"x")})
        assert wait_done(client, r.json()["job_id"])["meta"]["zones"] is None
        cfg = client.get("/api/v1/config")
        assert cfg.status_code == 200 and cfg.json()["default_profile"] == "auto"
        assert all(b["quote"] for rule in cfg.json()["rules"] for b in rule["legal_basis"])


def test_review_validation_and_model_endpoint():
    with TestClient(app) as client:
        job_id = client.post("/api/v1/analyze", headers=KEY, files={"file": ("a.mov", b"x")}).json()["job_id"]
        wait_done(client, job_id)
        url = f"/api/v1/jobs/{job_id}/review"
        assert client.post(url, json={"label": "no_violation"}).status_code == 401
        assert client.post(url, headers=KEY, json={"label": "violation"}).status_code == 422
        assert client.post(url, headers=KEY, json={"label": "maybe"}).status_code == 422
        assert client.post("/api/v1/jobs/nope/review", headers=KEY, json={"label": "no_violation"}).status_code == 404
        r = client.post(url, headers=KEY, json={"label": "violation", "timecode_sec": 3.0, "type_id": "x"})
        assert r.status_code == 200 and r.json()["label"] == "violation" and r.json()["trainable"] is False

        model = client.get("/api/v1/model").json()
        assert {"active_version", "metrics", "versions", "learning"} <= model.keys()
        assert model["learning"]["reviews"] >= 1
        assert client.post("/api/v1/model/retrain").status_code == 401
        assert client.post("/api/v1/model/versions/999/activate", headers=KEY).status_code == 404
