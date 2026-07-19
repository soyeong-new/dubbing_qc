from pathlib import Path
from unittest.mock import patch

from app.core.vocal_separation import separate_vocals


def test_separate_vocals_builds_demucs_command_and_returns_paths(tmp_path):
    audio = tmp_path / "original.wav"
    audio.write_bytes(b"fake")
    out_dir = tmp_path / "sep"
    produced = out_dir / "htdemucs" / "original"
    produced.mkdir(parents=True)
    (produced / "vocals.wav").write_bytes(b"v")
    (produced / "no_vocals.wav").write_bytes(b"b")

    with patch("app.core.vocal_separation.subprocess.run") as run:
        vocals, bgm = separate_vocals(str(audio), str(out_dir))

    cmd = run.call_args[0][0]
    assert "demucs" in cmd
    assert "--two-stems" in cmd and "vocals" in cmd
    assert str(audio) in cmd
    assert vocals == produced / "vocals.wav"
    assert bgm == produced / "no_vocals.wav"


def test_separate_vocals_custom_model(tmp_path):
    audio = tmp_path / "original.wav"
    audio.write_bytes(b"fake")
    out_dir = tmp_path / "sep"
    produced = out_dir / "htdemucs_ft" / "original"
    produced.mkdir(parents=True)
    (produced / "vocals.wav").write_bytes(b"v")
    (produced / "no_vocals.wav").write_bytes(b"b")

    with patch("app.core.vocal_separation.subprocess.run") as run:
        vocals, _ = separate_vocals(str(audio), str(out_dir), model="htdemucs_ft")

    cmd = run.call_args[0][0]
    assert "htdemucs_ft" in cmd
    assert vocals == produced / "vocals.wav"
