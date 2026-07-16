import math
import struct
import wave
import pytest
from app.core.rule_checks import (
    read_wav_mono, check_audio_quality, check_srt_audio_match, _token_similarity,
)
from app.providers.base import get_provider
from app.schemas import AlignedPair, SegmentText


def write_wav(path, samples, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"{len(samples)}h", *samples))


def sine(seconds, rate=16000, amp=8000, freq=440):
    n = int(seconds * rate)
    return [int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]


def pair_at(start, end, kr="대사", en="line", pid="p1"):
    return AlignedPair(
        id=pid,
        korean=SegmentText(start=start, end=end, speaker="A", text=kr),
        dubbed=SegmentText(start=start, end=end, speaker="A", text=en),
    )


def test_read_wav_mono(tmp_path):
    p = tmp_path / "a.wav"
    write_wav(p, sine(1.0))
    samples, rate = read_wav_mono(str(p))
    assert rate == 16000
    assert len(samples) == 16000


def test_clipping_detected(tmp_path):
    p = tmp_path / "clip.wav"
    write_wav(p, [32760, -32760] * 8000)  # 전부 클리핑
    findings = check_audio_quality(str(p), [pair_at(0.0, 1.0)])
    assert any(f.issue_type == "클리핑" and f.axis == "음질" for f in findings)


def test_dropout_detected_inside_segment(tmp_path):
    p = tmp_path / "drop.wav"
    write_wav(p, sine(1.0) + [0] * 16000 + sine(1.0))  # 1~2초 무음
    findings = check_audio_quality(str(p), [pair_at(0.5, 2.5, pid="p1")])
    assert any(f.issue_type == "드롭아웃" for f in findings)


def test_clean_audio_passes(tmp_path):
    p = tmp_path / "ok.wav"
    # 실제 대사 스템처럼 발화 사이에 무음 휴지가 있는 형태 (무음이 노이즈 플로어 역할)
    write_wav(p, sine(1.0) + [0] * 8000 + sine(1.0))
    findings = check_audio_quality(str(p), [pair_at(0.0, 1.0)])  # 세그먼트는 발화 구간만
    assert findings == []


def test_token_similarity():
    assert _token_similarity("did you eat rice", "did you eat rice") == 1.0
    assert _token_similarity("hello world", "completely different words") < 0.4


@pytest.mark.asyncio
async def test_srt_audio_match_flags_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    provider = get_provider()
    # MockProvider.transcribe는 한국어 고정 대사를 반환 → 영어 SRT와 불일치
    pairs = [pair_at(0.0, 2.0, en="totally unrelated english line", pid="p1")]

    def fake_clip(src, start, end):
        return src  # 실제 ffmpeg 호출 없이 원본 경로 반환

    findings = await check_srt_audio_match(
        pairs, "/tmp/stem.wav", provider, extract_clip_fn=fake_clip, sample_every=1,
    )
    assert len(findings) == 1
    assert findings[0].issue_type == "자막-음성 불일치"


@pytest.mark.asyncio
async def test_srt_audio_match_retries_once_on_transcribe_failure(monkeypatch):
    """STT가 한 번 실패(깨진 JSON 등)해도 재시도로 회복해야 한다."""
    monkeypatch.setenv("QC_PROVIDER", "mock")
    inner = get_provider()
    calls = {"n": 0}

    class FlakyProvider:
        async def transcribe(self, audio_path, lang):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("Expecting ',' delimiter: line 8 column 1")
            return await inner.transcribe(audio_path, lang)

        async def judge(self, *a, **kw):
            return []

    pairs = [pair_at(0.0, 2.0, en="totally unrelated english line", pid="p1")]
    findings = await check_srt_audio_match(
        pairs, "/tmp/stem.wav", FlakyProvider(),
        extract_clip_fn=lambda src, s, e: src, sample_every=1,
    )
    assert calls["n"] == 2  # 1회 실패 후 재시도했음
    assert len(findings) == 1  # 재시도 성공 → 정상적으로 불일치 검출


@pytest.mark.asyncio
async def test_srt_audio_match_skips_segment_after_persistent_failure(monkeypatch):
    """2회 모두 실패한 세그먼트는 건너뛰고, 전체 검사는 죽지 않아야 한다."""
    monkeypatch.setenv("QC_PROVIDER", "mock")
    inner = get_provider()

    class BrokenFirstProvider:
        async def transcribe(self, audio_path, lang):
            raise ValueError("Expecting ',' delimiter: line 8 column 1")

        async def judge(self, *a, **kw):
            return []

    pairs = [pair_at(0.0, 2.0, en="totally unrelated english line", pid="p1")]
    findings = await check_srt_audio_match(
        pairs, "/tmp/stem.wav", BrokenFirstProvider(),
        extract_clip_fn=lambda src, s, e: src, sample_every=1,
    )
    assert findings == []  # 예외 전파 없이 해당 세그먼트만 건너뜀
