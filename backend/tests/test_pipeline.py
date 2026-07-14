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


async def test_pipeline_end_to_end_with_srt_both(job_files, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
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
