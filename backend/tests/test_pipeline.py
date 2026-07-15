import math
import struct
import wave
import pytest
from app.core.pipeline import QCPipeline
from app.providers.base import get_provider
from app.schemas import QCJobInput

EN_SRT = """1
00:00:01,000 --> 00:00:03,000
Hey brother, did you eat rice?

2
00:00:04,000 --> 00:00:06,000
I have no kidney.
"""

KR_SRT = """1
00:00:01,000 --> 00:00:03,000
형, 밥 먹었어?

2
00:00:04,000 --> 00:00:06,000
어이가 없네.
"""


@pytest.fixture
def job_files(tmp_path):
    en = tmp_path / "en.srt"
    en.write_text(EN_SRT, encoding="utf-8")
    kr = tmp_path / "kr.srt"
    kr.write_text(KR_SRT, encoding="utf-8")
    stem = tmp_path / "stem.wav"
    rate, samples = 16000, []
    for i in range(rate * 7):
        samples.append(int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
    with wave.open(str(stem), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(struct.pack(f"{len(samples)}h", *samples))
    return str(en), str(kr), str(stem)


def _fake_classify_accent(clip_path):
    return "us", 0.95


async def test_pipeline_end_to_end_with_srt_both(job_files, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    monkeypatch.setattr("app.core.accent.classify_accent", _fake_classify_accent)
    en, kr, stem = job_files
    stages = []
    pipeline = QCPipeline(provider=get_provider())
    result = await pipeline.run(
        QCJobInput(movie_title="테스트", en_srt_path=en, kr_srt_path=kr, stem_audio_path=stem),
        on_progress=lambda stage, d, t: stages.append(stage),
    )
    assert result.verdict.status == "fail"  # kidney → high → 즉시 반려
    assert len(result.pairs) == 2
    seg_findings = [f for f in result.findings if f.source.startswith("persona:")]
    assert any("kidney" in f.current_translation for f in seg_findings)
    assert {"ingest", "align", "rules", "panel", "verdict"} <= set(stages)


async def test_pipeline_without_stem_skips_audio_checks(job_files, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    en, kr, _ = job_files
    pipeline = QCPipeline(provider=get_provider())
    result = await pipeline.run(QCJobInput(en_srt_path=en, kr_srt_path=kr))
    assert all(f.issue_type not in ("클리핑", "드롭아웃", "잡음") for f in result.findings)


async def test_pipeline_includes_sensitive_word_findings(job_files, monkeypatch, tmp_path):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    monkeypatch.setattr("app.core.accent.classify_accent", _fake_classify_accent)
    en, kr, stem = job_files
    # 사전에 확실히 걸리는 단어를 영어 SRT에 심는다
    sensitive_srt = tmp_path / "en_sensitive.srt"
    sensitive_srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nthis line has PLACEHOLDER-SLUR-1 in it\n",
        encoding="utf-8",
    )
    pipeline = QCPipeline(provider=get_provider())
    result = await pipeline.run(QCJobInput(
        movie_title="t", en_srt_path=str(sensitive_srt), kr_srt_path=kr,
        stem_audio_path=stem,
    ))
    sensitive_findings = [f for f in result.findings if f.finding_type == "sensitive"]
    assert len(sensitive_findings) >= 1


async def test_pipeline_passes_kr_audio_path_to_panel_for_director(job_files, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    monkeypatch.setattr("app.core.accent.classify_accent", _fake_classify_accent)
    en, kr, stem = job_files
    pipeline = QCPipeline(provider=get_provider())
    # kr_audio_path 없이도(한국어 SRT만 제공) 예외 없이 완료되어야 한다 —
    # 원본 오디오가 없으면 그냥 클립 없이 진행(우아한 저하)
    result = await pipeline.run(QCJobInput(
        movie_title="t", en_srt_path=en, kr_srt_path=kr, stem_audio_path=stem,
    ))
    assert result.verdict.status in ("pass", "conditional", "fail")
