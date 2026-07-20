"""씬 단위 오디오 클립 추출 — 판정 모델이 직접 청취할 mp3를 만든다.

맥락(ctx_seconds)을 앞에 붙이는 이유: 맥락 없는 초단문 클립은 두 모델이 같은
오류를 공유하는 상관 오류를 낳지만, 맥락을 주면 틀릴 때 서로 다르게 틀려 합의
필터가 작동한다(2026-07-20 벤치마크 실측 — 설계 스펙 §4).
"""
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

from app.schemas import AlignedPair

CTX_SECONDS = 25.0
PAD_SECONDS = 0.3


def scene_time_range(pairs: List[AlignedPair]) -> Optional[Tuple[float, float]]:
    # 영어 SRT가 타임코드 기준(주체)이다 — assign_scenes와 같은 앵커 규칙.
    anchors = [(p.dubbed or p.korean) for p in pairs if (p.dubbed or p.korean)]
    if not anchors:
        return None
    return anchors[0].start, anchors[-1].end


def extract_scene_clip(audio_path: str, start: float, end: float,
                       ctx_seconds: float = CTX_SECONDS, pad: float = PAD_SECONDS,
                       out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or tempfile.gettempdir()
    clip_start = max(0.0, start - ctx_seconds)
    duration = (end + pad) - clip_start
    out = os.path.join(out_dir, f"qc_scene_{clip_start:.3f}_{duration:.3f}.mp3")
    subprocess.run(
        ["ffmpeg", "-ss", f"{clip_start:.3f}", "-t", f"{duration:.3f}", "-i", audio_path,
         "-acodec", "libmp3lame", "-b:a", "32k", "-ar", "16000", "-ac", "1", "-y", out],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out
