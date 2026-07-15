import asyncio
import math
import os
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List
import yaml
from app.schemas import AlignedPair, QCFinding
from app.providers.base import ModelProvider


def _finding(kind: str, pair: AlignedPair, severity: str, issue_type: str,
             axis: str, description: str, recommendation: str,
             category: str = "localization") -> QCFinding:
    anchor = pair.korean or pair.dubbed
    return QCFinding(
        id=f"rule_{kind}_{pair.id}", segment_id=pair.id, category=category,
        severity=severity, issue_type=issue_type,
        start_time=anchor.start, end_time=anchor.end, speaker=anchor.speaker,
        description=description,
        original_text=pair.korean.text if pair.korean else "",
        current_translation=pair.dubbed.text if pair.dubbed else "",
        recommendation=recommendation, confidence=1.0,
        axis=axis, source="rule",
    )


def check_missing(pairs: List[AlignedPair]) -> List[QCFinding]:
    findings = []
    for p in pairs:
        if p.korean is not None and (p.dubbed is None or not p.dubbed.text.strip()):
            findings.append(_finding(
                "missing", p, "high", "번역 누락", "언어 적합성",
                "해당 한국어 대사에 대응하는 영어 더빙 대사가 없습니다.",
                "Provide the missing dubbed line.",
            ))
    return findings


def check_pacing(pairs: List[AlignedPair], max_words_per_sec: float = 3.8) -> List[QCFinding]:
    findings = []
    for p in pairs:
        if not p.dubbed or not p.dubbed.text.strip():
            continue
        duration = p.dubbed.end - p.dubbed.start
        if duration <= 0:
            continue
        wps = len(p.dubbed.text.split()) / duration
        if wps > max_words_per_sec:
            findings.append(_finding(
                "pacing", p, "medium", "발화속도 초과", "싱크 정확도",
                f"발화속도가 초당 {wps:.1f}단어로 기준({max_words_per_sec})을 초과합니다. "
                "성우 발화가 빨라져 입 싱크가 어긋날 수 있습니다.",
                "Shorten the line to fit the timing.", category="voice",
            ))
    return findings


def check_sync_overflow(pairs: List[AlignedPair], tolerance: float = 0.5) -> List[QCFinding]:
    findings = []
    for p in pairs:
        if not p.korean or not p.dubbed:
            continue
        overflow = max(p.korean.start - p.dubbed.start, p.dubbed.end - p.korean.end)
        if overflow > tolerance:
            findings.append(_finding(
                "sync", p, "medium", "싱크 오버플로", "싱크 정확도",
                f"더빙 구간이 원본 대사 구간을 {overflow:.1f}초 벗어납니다.",
                "Re-time the dubbed line to match the original segment.",
                category="voice",
            ))
    return findings


def check_low_alignment(pairs: List[AlignedPair], min_confidence: float = 0.3) -> List[QCFinding]:
    findings = []
    for p in pairs:
        if p.korean and p.dubbed and p.alignment_confidence < min_confidence:
            findings.append(_finding(
                "lowalign", p, "low", "정렬 신뢰도 저하", "싱크 정확도",
                f"한↔영 세그먼트 정렬 신뢰도가 {p.alignment_confidence:.2f}로 낮습니다. "
                "타임코드 검토가 필요합니다.",
                "Verify the timecode mapping manually.", category="voice",
            ))
    return findings


def run_text_checks(pairs: List[AlignedPair]) -> List[QCFinding]:
    return (check_missing(pairs) + check_pacing(pairs)
            + check_sync_overflow(pairs) + check_low_alignment(pairs))


def read_wav_mono(path: str):
    """Read a 16-bit mono WAV file and return (samples, sample_rate)."""
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "16-bit mono WAV 필요"
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    samples = list(struct.unpack(f"{len(raw) // 2}h", raw))
    return samples, rate


def _rms(chunk) -> float:
    """Compute RMS (root mean square) of a chunk of samples."""
    if not chunk:
        return 0.0
    return math.sqrt(sum(s * s for s in chunk) / len(chunk))


def check_audio_quality(wav_path: str, pairs: List[AlignedPair]) -> List[QCFinding]:
    """Check audio quality: clipping, dropouts, SNR."""
    samples, rate = read_wav_mono(wav_path)
    findings: List[QCFinding] = []
    if not samples:
        return findings

    # 1) 클리핑: 최대치 근접 샘플 비율
    clipped = sum(1 for s in samples if abs(s) >= 32700)
    if clipped / len(samples) > 0.001:
        anchor = pairs[0] if pairs else None
        if anchor:
            findings.append(_finding(
                "clipping", anchor, "high", "클리핑", "음질",
                f"오디오 샘플의 {clipped / len(samples) * 100:.2f}%가 클리핑되었습니다. "
                "왜곡된 구간의 재녹음/마스터링 확인이 필요합니다.",
                "Re-master or re-record the clipped sections.", category="voice",
            ))

    # 2) 세그먼트 내 드롭아웃: 대사 구간인데 0.3초 이상 RMS<100 연속
    frame = rate // 10  # 100ms
    for p in pairs:
        seg = p.dubbed or p.korean
        if seg is None:
            continue
        lo, hi = int(seg.start * rate), min(int(seg.end * rate), len(samples))
        silent_run = 0
        found = False
        for i in range(lo, hi, frame):
            if _rms(samples[i:i + frame]) < 100:
                silent_run += 1
                if silent_run >= 3 and not found:  # 300ms 이상
                    findings.append(_finding(
                        "dropout", p, "high", "드롭아웃", "음질",
                        "대사 구간 안에 0.3초 이상의 완전 무음이 있습니다. "
                        "오디오 누락 여부를 확인하세요.",
                        "Check for missing audio in this segment.", category="voice",
                    ))
                    found = True
            else:
                silent_run = 0

    # 3) SNR: 상위 20% 프레임 RMS 대비 하위 10% 프레임 RMS
    frame_rms = sorted(_rms(samples[i:i + frame]) for i in range(0, len(samples), frame))
    if len(frame_rms) >= 10:
        noise = frame_rms[max(0, int(len(frame_rms) * 0.1) - 1)] or 1.0
        speech = frame_rms[int(len(frame_rms) * 0.8)]
        snr_db = 20 * math.log10(speech / noise) if noise > 0 and speech > 0 else 99
        if snr_db < 15:
            anchor = pairs[0] if pairs else None
            if anchor:
                findings.append(_finding(
                    "snr", anchor, "medium", "잡음", "음질",
                    f"추정 SNR이 {snr_db:.0f}dB로 낮습니다. 배경 잡음 확인이 필요합니다.",
                    "Reduce background noise in the dialogue stem.", category="voice",
                ))
    return findings


def _token_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between token sets of two strings."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def extract_clip(src: str, start: float, end: float) -> str:
    """Extract a clip from a WAV file using ffmpeg."""
    out = os.path.join(tempfile.gettempdir(), f"qc_clip_{start:.1f}_{end:.1f}.wav")
    subprocess.run(
        ["ffmpeg", "-i", src, "-ss", str(start), "-to", str(end),
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return out


async def check_srt_audio_match(pairs: List[AlignedPair], stem_wav_path: str,
                                provider: ModelProvider, extract_clip_fn=extract_clip,
                                sample_every: int = 10) -> List[QCFinding]:
    """Check if dubbed audio matches the SRT text via transcription."""
    findings = []
    targets = [p for p in pairs if p.dubbed and p.dubbed.text.strip()][::sample_every]
    for p in targets:
        # extract_clip_fn은 ffmpeg를 동기 호출한다 — asyncio 이벤트 루프를
        # 막지 않도록 스레드로 넘긴다.
        clip = await asyncio.to_thread(extract_clip_fn, stem_wav_path, p.dubbed.start, p.dubbed.end)
        heard = await provider.transcribe(clip, lang="en")
        heard_text = " ".join(s.text for s in heard)
        if _token_similarity(p.dubbed.text, heard_text) < 0.4:
            findings.append(_finding(
                "srtmatch", p, "medium", "자막-음성 불일치", "언어 적합성",
                f"SRT 자막과 실제 더빙 음성이 다르게 들립니다. "
                f"(음성 인식 결과: \"{heard_text[:80]}\") 누락/애드리브/다른 테이크 여부를 확인하세요.",
                "Verify the recorded line against the final script.",
            ))
    return findings


_DEFAULT_SENSITIVE_WORDS = Path(__file__).parent.parent / "knowledge" / "sensitive_words.yaml"


def load_sensitive_terms(path: str = None) -> List[tuple]:
    p = Path(path) if path else _DEFAULT_SENSITIVE_WORDS
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [(t["word"].lower(), t.get("category", "기타")) for t in data.get("terms", [])]


def check_sensitive_words(pairs: List[AlignedPair], terms: List[tuple] = None) -> List[QCFinding]:
    terms = terms if terms is not None else load_sensitive_terms()
    findings = []
    for p in pairs:
        if not p.dubbed or not p.dubbed.text.strip():
            continue
        text_lower = p.dubbed.text.lower()
        for word, category in terms:
            if word in text_lower:
                anchor = p.korean or p.dubbed
                findings.append(QCFinding(
                    id=f"rule_sensitive_{p.id}_{word.replace(' ', '_')}",
                    segment_id=p.id, category="localization", severity="high",
                    issue_type=f"민감어({category})",
                    start_time=anchor.start, end_time=anchor.end, speaker=anchor.speaker,
                    description=f"금칙어 사전에 등록된 표현이 감지되었습니다 (분류: {category}). "
                                "해당 표현의 사용 맥락과 등급 영향을 검토하세요.",
                    original_text=p.korean.text if p.korean else "",
                    current_translation=p.dubbed.text,
                    recommendation="해당 표현을 검토하고 필요 시 수정하세요.",
                    confidence=1.0, axis="언어 적합성", source="rule",
                    finding_type="sensitive",
                ))
                break
    return findings
