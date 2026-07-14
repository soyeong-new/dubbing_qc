const API_BASE = "http://localhost:8000";

export async function uploadMedia(file, role) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/qc/upload-media?role=${role}`, {
    method: "POST", body: form,
  });
  return res.json();
}

export async function runQC(payload) {
  const res = await fetch(`${API_BASE}/api/qc/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.status === 503) {
    const body = await res.json();
    throw new Error(body.detail);
  }
  return res.json();
}

export async function getJob(jobId) {
  const res = await fetch(`${API_BASE}/api/qc/jobs/${jobId}`);
  return res.json();
}

export async function postFeedback(entry) {
  const res = await fetch(`${API_BASE}/api/qc/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(entry),
  });
  return res.json();
}

export async function reverdict(jobId, excludedFindingIds) {
  const res = await fetch(`${API_BASE}/api/qc/jobs/${jobId}/reverdict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ excluded_finding_ids: excludedFindingIds }),
  });
  return res.json();
}

export function exportUrl(jobId) {
  return `${API_BASE}/api/qc/export/${jobId}`;
}
