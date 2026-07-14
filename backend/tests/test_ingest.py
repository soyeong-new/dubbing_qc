import pytest
from app.core.ingest import parse_srt, load_text_source
from app.providers.base import get_provider

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


async def test_load_text_source_prefers_srt(tmp_path, monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    srt = tmp_path / "en.srt"
    srt.write_text(SAMPLE_SRT, encoding="utf-8")
    segments = await load_text_source("en", str(srt), "/tmp/audio.wav", get_provider())
    assert segments[0].text == "Hey man, did you eat rice?"  # STT가 아닌 SRT 결과


async def test_load_text_source_falls_back_to_stt(monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    segments = await load_text_source("ko", None, "/tmp/audio.wav", get_provider())
    assert "눈치" in segments[0].text  # MockProvider.transcribe 결과


async def test_load_text_source_requires_some_input(monkeypatch):
    monkeypatch.setenv("QC_PROVIDER", "mock")
    with pytest.raises(ValueError):
        await load_text_source("ko", None, None, get_provider())
