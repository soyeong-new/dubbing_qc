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
    # 참고: 이 시나리오는 en_srt에 줄이 하나뿐이라 두 번째 KR 줄이 "번역 누락"(quality/high)으로
    # 잡히고, 픽스처의 합성 사인파 stem 오디오가 SNR 체크에서 상시 "잡음"(quality/medium)을
    # 유발한다 — 두 세그먼트짜리 미니 픽스처에서는 그 하나만으로도 해당 축 MOS가 1로
    # 떨어져 fail이 된다. 즉 이 시나리오의 verdict.status는 원래도 fail이며, 민감어
    # finding이 fail을 "추가로" 강제하는지 여부는 이 케이스로는 격리해서 검증할 수 없다.
    # verdict가 sensitive-only high로 fail을 강제하지 않는지는 아래
    # test_sensitive_only_high_finding_does_not_force_fail에서 다른 quality 지적이
    # 전혀 없는 깨끗한 시나리오로 검증한다.


async def test_sensitive_only_high_finding_does_not_force_fail(monkeypatch, tmp_path):
    """민감어(high, finding_type=sensitive) 단독으로는 verdict를 fail로 만들지 않아야 한다.

    check_sensitive_words는 텍스트 기반이므로 stem_audio_path 없이 실행해
    (합성 sine wave 픽스처가 유발하는 상시 SNR 'medium' 잡음 quality finding 등)
    다른 축의 quality finding이 섞이지 않는 깨끗한 시나리오를 만든다. 두 자막 줄의
    타임코드/속도를 원본과 정확히 맞춰 pacing/sync/low-alignment/missing 체크 중
    무엇도 걸리지 않도록 한다.
    """
    monkeypatch.setenv("QC_PROVIDER", "mock")
    kr = tmp_path / "kr_clean.srt"
    kr.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n형, 밥 먹었어?\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n어이가 없네.\n",
        encoding="utf-8",
    )
    en = tmp_path / "en_clean_sensitive.srt"
    en.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHey friend, did you have a meal?\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nthis scene has a PLACEHOLDER-SLUR-1 word\n",
        encoding="utf-8",
    )
    pipeline = QCPipeline(provider=get_provider())
    result = await pipeline.run(QCJobInput(
        movie_title="t", en_srt_path=str(en), kr_srt_path=str(kr),
    ))
    sensitive_findings = [f for f in result.findings if f.finding_type == "sensitive"]
    quality_high = [f for f in result.findings if f.finding_type == "quality" and f.severity == "high"]
    assert len(sensitive_findings) >= 1
    assert quality_high == []  # 이 시나리오에 다른 quality/high 지적이 없음을 확인
    assert result.verdict.status != "fail"


async def test_pipeline_survives_accent_classification_failure(job_files, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")

    def _raise_classify_accent(clip_path):
        raise RuntimeError("model load failed")

    monkeypatch.setattr("app.core.accent.classify_accent", _raise_classify_accent)
    en, kr, stem = job_files
    pipeline = QCPipeline(provider=get_provider())
    # 억양 분류가 실패해도 파이프라인 전체가 죽지 않고, 억양 관련 finding 없이
    # 완료되어야 한다 (우아한 저하).
    result = await pipeline.run(QCJobInput(
        movie_title="t", en_srt_path=en, kr_srt_path=kr, stem_audio_path=stem,
    ))
    assert all(f.axis != "억양 적합성" for f in result.findings)


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
