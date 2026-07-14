import os
import csv
import io
import uuid
import asyncio

# .env 로더 — 기존 코드 그대로 유지
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0].strip()] = parts[1].strip().strip("\"'")

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.schemas import QCJobInput, FeedbackEntry, QCFinding
from app.core.pipeline import QCPipeline
from app.core.verdict import load_config, compute_axis_scores, decide
from app.providers.base import get_provider, ProviderNotConfiguredError
from app.feedback.store import FeedbackStore
import shutil
import tempfile
import subprocess
import struct

app = FastAPI(title="AI Dubbing QC API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

JOBS: dict = {}
VALID_ROLES = {"original", "dubbed", "stem", "srt_en", "srt_kr"}


def _feedback_store() -> FeedbackStore:
    path = os.getenv("QC_FEEDBACK_PATH",
                     os.path.join(os.path.dirname(__file__), "..", "data", "feedback.jsonl"))
    return FeedbackStore(path)


@app.get("/")
def read_root():
    return {"message": "AI Dubbing QC Backend API is running."}


@app.post("/api/qc/upload-media")
async def upload_media(file: UploadFile = File(...), role: str = "dubbed"):
    if role not in VALID_ROLES:
        raise HTTPException(400, f"role은 {sorted(VALID_ROLES)} 중 하나여야 합니다.")
    temp_dir = tempfile.gettempdir()
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
    media_path = os.path.join(temp_dir, f"{role}_{safe_filename}")
    with open(media_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if role.startswith("srt_"):
        return {"success": True, "role": role, "filename": file.filename,
                "media_path": media_path}

    # 미디어(원본/더빙본/스템): 16kHz mono WAV 추출 + waveform peaks — 기존 로직 그대로
    audio_filename = f"{role}_{os.path.splitext(safe_filename)[0]}.wav"
    audio_path = os.path.join(temp_dir, audio_filename)
    raw_audio_path = os.path.join(temp_dir, f"{role}_{os.path.splitext(safe_filename)[0]}_peaks.raw")
    try:
        subprocess.run(["ffmpeg", "-i", media_path, "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1", "-y", audio_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # NOTE: 저레이트 리샘플 금지 — 안티앨리어싱 필터가 음성 에너지를 제거함.
        # 시각화용 다운샘플은 아래 max-per-bin으로 수행.
        subprocess.run(["ffmpeg", "-i", media_path, "-f", "s16le", "-ac", "1",
                        "-ar", "16000", "-y", raw_audio_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        peaks = []
        if os.path.exists(raw_audio_path):
            with open(raw_audio_path, "rb") as f:
                raw_data = f.read()
            num_samples = len(raw_data) // 2
            if num_samples > 0:
                samples = struct.unpack(f"{num_samples}h", raw_data)
                bin_size = max(1, num_samples // 600)
                for i in range(0, num_samples, bin_size):
                    chunk = samples[i:i + bin_size]
                    if chunk:
                        peaks.append(round(max(abs(s) for s in chunk) / 32768.0, 3))
            os.remove(raw_audio_path)
        return {"success": True, "role": role, "filename": file.filename,
                "audio_path": audio_path, "media_path": media_path, "waveform": peaks}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def _run_job(job_id: str, job: QCJobInput):
    JOBS[job_id]["status"] = "running"

    def on_progress(stage, done, total):
        JOBS[job_id]["progress"] = {"stage": stage, "done": done, "total": total}

    try:
        pipeline = QCPipeline(provider=get_provider())
        result = await pipeline.run(job, on_progress=on_progress)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result.model_dump()
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.post("/api/qc/run", status_code=202)
async def run_qc(job: QCJobInput, background_tasks: BackgroundTasks):
    try:
        get_provider()  # 키 검증 — mock 자동 폴백 없음, 실패 시 즉시 거부
    except ProviderNotConfiguredError as e:
        raise HTTPException(503, str(e))
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "progress": None, "movie": job.movie_title}
    background_tasks.add_task(_run_job, job_id, job)
    return {"job_id": job_id}


@app.get("/api/qc/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "존재하지 않는 작업입니다.")
    return JOBS[job_id]


@app.post("/api/qc/feedback")
def post_feedback(entry: FeedbackEntry):
    _feedback_store().record(entry)
    return {"ok": True}


class ReverdictRequest(BaseModel):
    excluded_finding_ids: list[str] = []


@app.post("/api/qc/jobs/{job_id}/reverdict")
def reverdict(job_id: str, req: ReverdictRequest):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "완료된 작업이 아닙니다.")
    excluded = set(req.excluded_finding_ids)
    kept = [QCFinding(**f) for f in job["result"]["findings"] if f["id"] not in excluded]
    config = load_config()
    axis_scores = compute_axis_scores(kept, n_pairs=len(job["result"]["pairs"]), config=config)
    verdict = decide(axis_scores, kept, config)
    job["result"]["verdict"] = verdict.model_dump()  # 확정 판정으로 갱신
    return verdict.model_dump()


@app.get("/api/qc/export/{job_id}")
def export_csv(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "완료된 작업이 아닙니다.")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["타임코드", "화자", "축", "심각도", "유형", "한국어 원문",
                     "영어 더빙", "지적 사유", "수정안", "동의 수"])
    for f in job["result"]["findings"]:
        writer.writerow([
            f"{f['start_time']:.1f}-{f['end_time']:.1f}", f["speaker"], f["axis"],
            f["severity"], f["issue_type"], f["original_text"],
            f["current_translation"], f["description"], f["recommendation"],
            f["agreement"],
        ])
    return Response(
        content="﻿" + buf.getvalue(),  # UTF-8 BOM — 엑셀 한글 호환
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=qc_report_{job_id}.csv"},
    )
