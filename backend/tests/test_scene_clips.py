import os
import subprocess

import pytest

from app.core.scene_clips import extract_scene_clip, scene_time_range
from app.schemas import AlignedPair, SegmentText


def _tone_wav(path: str, seconds: int = 40) -> None:
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", "16000", "-ac", "1", "-y", path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def test_extract_scene_clip_includes_context(tmp_path):
    wav = str(tmp_path / "src.wav")
    _tone_wav(wav)
    out = extract_scene_clip(wav, start=30.0, end=32.0, out_dir=str(tmp_path))
    assert os.path.exists(out) and out.endswith(".mp3")
    # 25초 맥락 + 2초 본문 + 0.3초 패드 = 27.3초
    assert abs(_duration(out) - 27.3) < 0.5


def test_extract_scene_clip_clamps_at_zero(tmp_path):
    wav = str(tmp_path / "src.wav")
    _tone_wav(wav, seconds=10)
    out = extract_scene_clip(wav, start=3.0, end=5.0, out_dir=str(tmp_path))
    # 시작이 0 밑으로 내려가지 않는다: 0~5.3초 = 5.3초
    assert abs(_duration(out) - 5.3) < 0.5


def test_scene_time_range_uses_dubbed_anchor():
    pairs = [
        AlignedPair(id="pair_1", dubbed=SegmentText(start=10.0, end=12.0, text="a")),
        AlignedPair(id="pair_2", dubbed=SegmentText(start=15.0, end=17.5, text="b")),
    ]
    assert scene_time_range(pairs) == (10.0, 17.5)


def test_scene_time_range_empty_returns_none():
    assert scene_time_range([]) is None
