import time
import pytest
from fastapi.testclient import TestClient

EN_SRT = """1
00:00:01,000 --> 00:00:03,000
I have no kidney.
"""

KR_SRT = """1
00:00:01,000 --> 00:00:03,000
어이가 없네.
"""


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    monkeypatch.setenv("QC_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    from app.main import app
    return TestClient(app)


def _run_job(client, tmp_path):
    en = tmp_path / "en.srt"; en.write_text(EN_SRT, encoding="utf-8")
    kr = tmp_path / "kr.srt"; kr.write_text(KR_SRT, encoding="utf-8")
    res = client.post("/api/qc/run", json={
        "movie_title": "t", "en_srt_path": str(en), "kr_srt_path": str(kr),
    })
    assert res.status_code == 202
    job_id = res.json()["job_id"]
    for _ in range(50):
        job = client.get(f"/api/qc/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job_id, job
        time.sleep(0.1)
    pytest.fail("job did not finish")


def test_run_and_poll_job(client, tmp_path):
    job_id, job = _run_job(client, tmp_path)
    assert job["status"] == "done"
    assert job["result"]["verdict"]["status"] == "fail"
    assert len(job["result"]["pairs"]) == 1


def test_run_without_provider_returns_503(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("QC_PROVIDER", "gemini")
    from app.main import app
    client = TestClient(app)
    en = tmp_path / "en.srt"; en.write_text(EN_SRT, encoding="utf-8")
    res = client.post("/api/qc/run", json={"en_srt_path": str(en)})
    assert res.status_code == 503
    assert "GEMINI_API_KEY" in res.json()["detail"]


def test_feedback_recorded(client):
    res = client.post("/api/qc/feedback", json={
        "movie": "t", "segment_id": "pair_1", "korean": "어이가 없네",
        "dubbed": "I have no kidney", "finding_id": "f1",
        "reviewer_action": "modified", "final_text": "This is ridiculous.",
    })
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_export_csv(client, tmp_path):
    job_id, _ = _run_job(client, tmp_path)
    res = client.get(f"/api/qc/export/{job_id}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/csv")
    assert "kidney" in res.text


def test_reverdict_excluding_high_finding_lifts_fail(client, tmp_path):
    job_id, job = _run_job(client, tmp_path)
    assert job["result"]["verdict"]["status"] == "fail"
    high_ids = [f["id"] for f in job["result"]["findings"] if f["severity"] == "high"]
    res = client.post(f"/api/qc/jobs/{job_id}/reverdict",
                      json={"excluded_finding_ids": high_ids})
    assert res.status_code == 200
    assert res.json()["status"] != "fail"  # high 오탐 제외 → 반려 해제


def test_removed_endpoints_are_gone(client):
    assert client.get("/api/qc/mock-data").status_code == 404
    assert client.post("/api/qc/translate", json={"segments": []}).status_code == 404
    assert client.post("/api/qc/transcribe", json={"audio_path": "x"}).status_code == 404
    assert client.post("/api/qc/process", json={}).status_code == 404
    assert client.post("/api/qc/upload-video").status_code == 404
