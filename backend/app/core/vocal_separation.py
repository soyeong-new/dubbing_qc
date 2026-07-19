"""한국어 원본 오디오의 보컬(대사)/배경음 분리 — Demucs(htdemucs) 래퍼.

로컬 STT는 원본 파일 전체 믹스(대사+음악+효과음)를 그대로 받는데, 효과음·음악
구간에서 Whisper가 환각(반복, 지어내기)을 일으키는 것이 실측으로 확인되었다.
이 모듈은 STT에 넣기 전에 보컬만 분리해 그 문제를 줄이기 위한 전처리다.
"""
import subprocess
import sys
from pathlib import Path
from typing import Tuple

DEMUCS_MODEL = "htdemucs"


def separate_vocals(audio_wav: str, out_dir: str, model: str = DEMUCS_MODEL) -> Tuple[Path, Path]:
    """Demucs --two-stems로 보컬/배경음 분리. (vocals, no_vocals) 경로 반환.

    model: 'htdemucs'(기본, 빠름) 또는 'htdemucs_ft'(fine-tuned, 더 깨끗·약 4배 느림).
    """
    audio_wav_p = Path(audio_wav)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-o", str(out_dir_p),
        str(audio_wav_p),
    ]
    subprocess.run(cmd, check=True)
    produced = out_dir_p / model / audio_wav_p.stem
    return produced / "vocals.wav", produced / "no_vocals.wav"
