import React, { useState } from "react";
import { exportUrl, reverdict } from "../api";

const STATUS_META = {
  pass: { label: "통과", cls: "verdict-pass", icon: "✅" },
  conditional: { label: "조건부 통과", cls: "verdict-cond", icon: "⚠️" },
  fail: { label: "반려", cls: "verdict-fail", icon: "❌" },
};

export default function ReportView({ result, jobId, findings, reviewed }) {
  const [finalVerdict, setFinalVerdict] = useState(null);
  const [reverdicting, setReverdicting] = useState(false);
  const [reverdictError, setReverdictError] = useState(null);
  if (!result) return <div className="report-empty">완료된 QC 분석이 없습니다. 프로젝트 탭에서 분석을 실행하세요.</div>;

  // AI 가판정 → 검수자가 오탐을 반려한 뒤 "확정 재판정"하면 finalVerdict로 대체
  const verdict = finalVerdict || result.verdict;
  const meta = STATUS_META[verdict.status];

  const confirmVerdict = async () => {
    if (reverdicting) return;
    setReverdicting(true);
    setReverdictError(null);
    try {
      const excluded = Object.entries(reviewed)
        .filter(([, r]) => r.action === "rejected")
        .map(([id]) => id);
      setFinalVerdict(await reverdict(jobId, excluded));
    } catch (err) {
      setReverdictError(err.message || "재판정 요청이 실패했습니다.");
    } finally {
      setReverdicting(false);
    }
  };
  // 검수자가 반려(오탐)한 finding 제외 = 확정 지시서
  const confirmed = findings.filter((f) => reviewed[f.id]?.action !== "rejected");

  return (
    <div className="report-view">
      <div className={`verdict-banner ${meta.cls}`}>
        <span className="verdict-icon">{meta.icon}</span>
        <span className="verdict-label">{meta.label}</span>
        <span className="verdict-note">
          AI 가판정 기준 — 검수 확정 시 오탐 제외 후 재판정됩니다.
        </span>
      </div>
      {verdict.reasons.length > 0 && (
        <ul className="verdict-reasons">
          {verdict.reasons.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      )}
      <h3>5축 MOS 스코어카드</h3>
      <div className="mos-grid">
        {verdict.axis_scores.map((s) => (
          <div key={s.axis} className="mos-row">
            <span className="mos-axis">{s.axis}</span>
            <span className="mos-bar">
              {[1, 2, 3, 4, 5].map((n) => (
                <span key={n} className={`mos-cell ${n <= s.mos ? `filled mos-${s.mos}` : ""}`} />
              ))}
            </span>
            <span className="mos-value">{s.mos}</span>
          </div>
        ))}
      </div>
      <h3>수정 지시서 ({confirmed.length}건)</h3>
      <table className="report-table">
        <thead>
          <tr><th>타임코드</th><th>축</th><th>심각도</th><th>원문</th><th>더빙</th><th>수정안</th><th>상태</th></tr>
        </thead>
        <tbody>
          {confirmed.map((f) => (
            <tr key={f.id}>
              <td>{f.start_time.toFixed(1)}s</td>
              <td>{f.axis}</td>
              <td className={`sev-${f.severity}`}>{f.severity}</td>
              <td>{f.original_text}</td>
              <td>{f.current_translation}</td>
              <td>{reviewed[f.id]?.finalText || f.recommendation}</td>
              <td>{reviewed[f.id] ? reviewed[f.id].action : "미검수"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {reverdictError && <div className="review-error">{reverdictError}</div>}
      <div className="report-actions">
        <button className="export-btn" onClick={confirmVerdict} disabled={reverdicting}>
          {reverdicting ? "재판정 중…" : "검수 확정 재판정 (반려한 오탐 제외)"}
        </button>
        <a className="export-btn" href={exportUrl(jobId)} download>CSV 내보내기 (엑셀)</a>
        <button className="export-btn" onClick={() => window.print()}>인쇄 / PDF 저장</button>
      </div>
    </div>
  );
}
