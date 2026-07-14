import React, { useState, useRef, useEffect } from "react";
import { uploadMedia, runQC, getJob } from "../api";

const SLOTS = [
  { role: "original", label: "한국어 원본 영상", accept: "video/*,audio/*", required: true },
  { role: "dubbed", label: "영어 더빙 완성본", accept: "video/*", required: true },
  { role: "stem", label: "다이얼로그 사운드 (스템)", accept: "audio/*", required: true },
  { role: "srt_en", label: "영어 SRT 자막", accept: ".srt", required: true },
  { role: "srt_kr", label: "한국어 SRT (선택 — 있으면 STT 생략)", accept: ".srt", required: false },
];

const STAGE_LABELS = {
  ingest: "대본 수집 (SRT/STT)", align: "타임코드 정렬",
  rules: "결정론적 룰 체크", panel: "페르소나 패널 분석", verdict: "판정 계산",
};

export default function ProjectView({ uploads, setUploads, onJobComplete }) {
  const [movieTitle, setMovieTitle] = useState("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  // 탭 전환 등으로 컴포넌트가 언마운트돼도 폴링이 계속 돌지 않도록 정리한다
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleFile = async (role, file) => {
    if (!file) return;
    setUploads((u) => ({ ...u, [role]: { name: file.name, uploading: true } }));
    const res = await uploadMedia(file, role);
    setUploads((u) => ({
      ...u,
      [role]: res.success
        ? { name: file.name, ...res, uploading: false }
        : { name: file.name, error: res.error, uploading: false },
    }));
  };

  const requiredReady = SLOTS.filter((s) => s.required)
    .every((s) => uploads[s.role]?.media_path);

  const start = async () => {
    setError(null);
    setRunning(true);
    try {
      const { job_id } = await runQC({
        movie_title: movieTitle || "untitled",
        en_srt_path: uploads.srt_en.media_path,
        kr_srt_path: uploads.srt_kr?.media_path || null,
        kr_audio_path: uploads.original?.audio_path || null,
        stem_audio_path: uploads.stem?.audio_path || null,
      });
      pollRef.current = setInterval(async () => {
        const job = await getJob(job_id);
        setProgress(job.progress);
        if (job.status === "done") {
          clearInterval(pollRef.current);
          setRunning(false);
          onJobComplete(job_id, job.result, movieTitle || "untitled");
        } else if (job.status === "error") {
          clearInterval(pollRef.current);
          setRunning(false);
          setError(job.error);
        }
      }, 2000);
    } catch (e) {
      setRunning(false);
      setError(e.message);
    }
  };

  return (
    <div className="project-view">
      <h2>새 QC 프로젝트</h2>
      <input
        className="title-input" placeholder="작품명"
        value={movieTitle} onChange={(e) => setMovieTitle(e.target.value)}
      />
      <div className="upload-grid">
        {SLOTS.map((slot) => (
          <label key={slot.role} className={`upload-slot ${uploads[slot.role]?.media_path ? "done" : ""}`}>
            <span className="slot-label">
              {slot.label}{slot.required ? " *" : ""}
            </span>
            <span className="slot-file">
              {uploads[slot.role]?.uploading ? "업로드 중…"
                : uploads[slot.role]?.name || "파일 선택"}
            </span>
            <input type="file" accept={slot.accept} hidden
              onChange={(e) => handleFile(slot.role, e.target.files[0])} />
          </label>
        ))}
      </div>
      {error && <div className="error-banner">⚠ {error}</div>}
      {running && progress && (
        <div className="progress-panel">
          <div>{STAGE_LABELS[progress.stage] || progress.stage} — {progress.done}/{progress.total}</div>
          <div className="progress-bar">
            <div className="progress-fill"
              style={{ width: `${(progress.done / Math.max(progress.total, 1)) * 100}%` }} />
          </div>
        </div>
      )}
      <button className="start-btn" disabled={!requiredReady || running} onClick={start}>
        {running ? "분석 중…" : "QC 분석 시작"}
      </button>
    </div>
  );
}
