import pytest
from app.core.ingest import parse_srt, load_text_source
from app.schemas import SegmentText

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,500
Hey man, did you eat rice?

2
00:00:05,200 --> 00:00:07,800
This is ridiculous.
It really is.

"""


def test_parse_srt_basic():
    segments = parse_srt(SAMPLE_SRT)
    assert len(segments) == 2
    assert segments[0].start == 1.0
    assert segments[0].end == 4.5
    assert segments[0].text == "Hey man, did you eat rice?"
    assert segments[1].text == "This is ridiculous. It really is."


def test_parse_srt_skips_malformed_blocks():
    segments = parse_srt("garbage\n\n1\n00:00:01,000 --> 00:00:02,000\nok\n")
    assert len(segments) == 1
    assert segments[0].text == "ok"


async def test_load_text_source_prefers_srt(tmp_path):
    srt = tmp_path / "en.srt"
    srt.write_text(SAMPLE_SRT, encoding="utf-8")
    segments = await load_text_source("en", str(srt), "/tmp/audio.wav")
    assert segments[0].text == "Hey man, did you eat rice?"  # 로컬 STT가 아닌 SRT 결과


async def test_load_text_source_falls_back_to_local_stt_for_korean(monkeypatch):
    def fake_transcribe_korean(audio_path):
        return [SegmentText(start=0.0, end=1.0, text="눈치 좀 봐라")]

    monkeypatch.setattr("app.core.local_stt.transcribe_korean", fake_transcribe_korean)
    segments = await load_text_source("ko", None, "/tmp/audio.wav")
    assert "눈치" in segments[0].text


async def test_load_text_source_requires_some_input():
    with pytest.raises(ValueError):
        await load_text_source("ko", None, None)


async def test_load_text_source_runs_stt_on_separated_vocals(monkeypatch, tmp_path):
    # 원본은 대사+음악+효과음이 섞인 전체 믹스라 Whisper 환각의 원인이 된다 —
    # STT에는 원본이 아니라 분리된 보컬 파일이 들어가야 한다.
    captured = {}

    def fake_separate_vocals(audio_path, out_dir, model="htdemucs"):
        captured["separate_audio_path"] = audio_path
        vocals = tmp_path / "vocals.wav"
        vocals.write_bytes(b"v")
        return vocals, tmp_path / "no_vocals.wav"

    def fake_transcribe_korean(audio_path):
        captured["stt_audio_path"] = audio_path
        return [SegmentText(start=0.0, end=1.0, text="분리된 보컬")]

    monkeypatch.setattr("app.core.vocal_separation.separate_vocals", fake_separate_vocals)
    monkeypatch.setattr("app.core.local_stt.transcribe_korean", fake_transcribe_korean)

    segments = await load_text_source("ko", None, "/tmp/original.wav")

    assert captured["separate_audio_path"] == "/tmp/original.wav"
    assert captured["stt_audio_path"] == str(tmp_path / "vocals.wav")
    assert segments[0].text == "분리된 보컬"


async def test_load_text_source_falls_back_to_original_audio_when_separation_fails(monkeypatch):
    # 분리 실패(예: demucs 오류)가 전체 STT 작업을 막으면 안 된다 — 원본으로 계속 진행한다.
    def boom(audio_path, out_dir, model="htdemucs"):
        raise RuntimeError("demucs 실패")

    def fake_transcribe_korean(audio_path):
        return [SegmentText(start=0.0, end=1.0, text=f"stt:{audio_path}")]

    monkeypatch.setattr("app.core.vocal_separation.separate_vocals", boom)
    monkeypatch.setattr("app.core.local_stt.transcribe_korean", fake_transcribe_korean)

    segments = await load_text_source("ko", None, "/tmp/original.wav")
    assert segments[0].text == "stt:/tmp/original.wav"
